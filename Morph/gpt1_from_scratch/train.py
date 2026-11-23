import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from config import (
    ensureDirectories,
    getConfig,
    getWeightsFilePath,
    listCheckpoints,
    listPresets,
    resolveCheckpointPath,
    validateConfig,
)
from dataset import getDataLoaders
from model import buildGpt1, countParameters, estimateParameterCount
from tokenizer import getTokenId, loadTokenizer


def getDevice(preferred=None):
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def getLearningRate(step: int, config, totalSteps: int):
    warmupSteps = config["warmupSteps"]
    maxLearningRate = config["lr"]

    if warmupSteps > 0 and step < warmupSteps:
        return maxLearningRate * (step + 1) / warmupSteps

    decaySteps = max(1, totalSteps - warmupSteps)
    progress = min(1.0, max(0.0, (step - warmupSteps) / decaySteps))
    return maxLearningRate * 0.5 * (1.0 + math.cos(math.pi * progress))


def setLearningRate(optimizer, learningRate: float):
    for parameterGroup in optimizer.param_groups:
        parameterGroup["lr"] = learningRate


def saveCheckpoint(model, optimizer, config, epoch: int, globalStep: int):
    modelFilename = Path(getWeightsFilePath(config, globalStep))
    modelFilename.parent.mkdir(parents=True, exist_ok=True)
    temporaryFilename = Path(str(modelFilename) + ".tmp")

    torch.save(
        {
            "epoch": epoch,
            "globalStep": globalStep,
            "modelStateDict": model.state_dict(),
            "optimizerStateDict": optimizer.state_dict(),
            "config": config,
        },
        temporaryFilename,
    )
    temporaryFilename.replace(modelFilename)
    print(f"Saved checkpoint {modelFilename}")


def pruneCheckpoints(config):
    checkpoints = listCheckpoints(config)
    keepCount = max(1, config["keepCheckpoints"])

    for oldCheckpoint in checkpoints[:-keepCount]:
        oldCheckpoint.unlink()
        print(f"Removed old checkpoint {oldCheckpoint}")


def loadCheckpoint(model, optimizer, config, device):
    preload = config.get("preload")
    if preload is None:
        return 0, 0

    modelFilename = resolveCheckpointPath(config, preload)
    print(f"Loading checkpoint {modelFilename}")
    checkpoint = torch.load(modelFilename, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["modelStateDict"])
    optimizer.load_state_dict(checkpoint["optimizerStateDict"])

    startEpoch = int(checkpoint["epoch"]) + 1
    globalStep = int(checkpoint["globalStep"])
    print(f"Resuming after epoch {checkpoint['epoch']} at step {globalStep:,}")
    return startEpoch, globalStep


@torch.no_grad()
def runValidation(model, valDataloader, device, maxBatches=None):
    wasTraining = model.training
    model.eval()

    totalLoss = 0.0
    totalTokens = 0

    for batchIndex, (inputs, labels) in enumerate(valDataloader):
        if maxBatches is not None and batchIndex >= maxBatches:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        _, loss = model(inputs, labels)
        totalTokens += labels.numel()
        totalLoss += loss.item() * labels.numel()

    model.train(wasTraining)

    averageLoss = totalLoss / max(1, totalTokens)
    perplexity = math.exp(min(averageLoss, 20.0))
    return averageLoss, perplexity


@torch.no_grad()
def generateSample(model, tokenizer, device, prompt: str, config):
    wasTraining = model.training
    model.eval()

    if prompt:
        promptTokens = tokenizer.encode(prompt).ids
    else:
        promptTokens = []

    if not promptTokens:
        promptTokens = [getTokenId(tokenizer, "[BOS]")]

    idx = torch.tensor([promptTokens], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        maxNewTokens=config["sampleTokens"],
        temperature=config["sampleTemperature"],
        topK=config["sampleTopK"],
    )
    text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    model.train(wasTraining)
    return text


def trainModel(config):
    setSeed(config["seed"])
    device = getDevice(config.get("device"))
    print(f"Using device: {device}")

    ensureDirectories(config)

    tokenizer = loadTokenizer(config)
    vocabSize = tokenizer.get_vocab_size()
    trainDataloader, valDataloader = getDataLoaders(config)

    model = buildGpt1(config, vocabSize).to(device)
    actualParameters = countParameters(model)
    estimatedParameters = estimateParameterCount(
        vocabSize,
        config["contextLen"],
        config["dModel"],
        config["numLayers"],
        config["dFf"],
    )
    print(f"Model parameters: {actualParameters:,} (estimated {estimatedParameters:,})")
    print(
        "Effective batch size: "
        f"{config['microBatchSize'] * config['gradAccumSteps']} sequences "
        f"({config['microBatchSize']} micro-batch x "
        f"{config['gradAccumSteps']} accumulation)"
    )

    optimizer = Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
        weight_decay=config["weightDecay"],
    )

    startEpoch, globalStep = loadCheckpoint(model, optimizer, config, device)

    stepsPerEpoch = len(trainDataloader) // config["gradAccumSteps"]
    if stepsPerEpoch == 0:
        raise ValueError(
            "Not enough training batches for one optimizer step. "
            "Reduce microBatchSize or gradAccumSteps."
        )

    remainingEpochSteps = max(0, config["numEpochs"] - startEpoch) * stepsPerEpoch
    totalSteps = globalStep + remainingEpochSteps
    if config.get("maxSteps") is not None:
        totalSteps = min(totalSteps, config["maxSteps"])

    print(f"Optimizer steps per epoch: {stepsPerEpoch:,}")

    if startEpoch >= config["numEpochs"]:
        print("Checkpoint is already at the final epoch. Nothing to train.")
        return
    if config.get("maxSteps") is not None and globalStep >= config["maxSteps"]:
        print("Checkpoint is already at maxSteps. Nothing to train.")
        return

    totalSteps = max(globalStep + 1, totalSteps)
    print(f"Planned optimizer steps: {totalSteps:,}")

    bestValidationLoss = float("inf")
    lastValidatedStep = None
    lastSampledStep = None
    lastSavedStep = None
    stopTraining = False

    for epoch in range(startEpoch, config["numEpochs"]):
        model.train()
        batchIterator = iter(trainDataloader)
        remainingSteps = totalSteps - globalStep
        stepsThisEpoch = min(stepsPerEpoch, remainingSteps)
        progress = tqdm(
            range(stepsThisEpoch),
            desc=f"Epoch {epoch:02d}",
            unit="step",
        )

        for _ in progress:
            learningRate = getLearningRate(globalStep, config, totalSteps)
            setLearningRate(optimizer, learningRate)

            optimizer.zero_grad(set_to_none=True)
            accumulatedLoss = 0.0

            for _ in range(config["gradAccumSteps"]):
                try:
                    inputs, labels = next(batchIterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "Training DataLoader ran out of batches during "
                        "gradient accumulation"
                    ) from error

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                _, loss = model(inputs, labels)
                (loss / config["gradAccumSteps"]).backward()
                accumulatedLoss += loss.item()

            clip_grad_norm_(model.parameters(), config["gradClip"])
            optimizer.step()
            globalStep += 1

            averageLoss = accumulatedLoss / config["gradAccumSteps"]
            progress.set_postfix(
                {
                    "loss": f"{averageLoss:.4f}",
                    "lr": f"{learningRate:.2e}",
                }
            )

            if globalStep % config["logEvery"] == 0:
                print(
                    f"step {globalStep:7d} | loss {averageLoss:.4f} | "
                    f"lr {learningRate:.2e}"
                )

            if globalStep % config["valEvery"] == 0:
                validationLoss, validationPerplexity = runValidation(
                    model,
                    valDataloader,
                    device,
                    config.get("maxValBatches"),
                )
                bestValidationLoss = min(bestValidationLoss, validationLoss)
                lastValidatedStep = globalStep
                print(
                    f"step {globalStep:7d} | val loss {validationLoss:.4f} | "
                    f"val perplexity {validationPerplexity:.2f}"
                )

            if globalStep % config["sampleEvery"] == 0:
                sampleText = generateSample(
                    model,
                    tokenizer,
                    device,
                    config["samplePrompt"],
                    config,
                )
                lastSampledStep = globalStep
                print(f"sample: {sampleText}")

            if globalStep % config["saveEvery"] == 0:
                saveCheckpoint(model, optimizer, config, epoch, globalStep)
                pruneCheckpoints(config)
                lastSavedStep = globalStep

            if globalStep >= totalSteps:
                stopTraining = True
                break

        progress.close()
        model.eval()

        if lastValidatedStep != globalStep:
            validationLoss, validationPerplexity = runValidation(
                model,
                valDataloader,
                device,
                config.get("maxValBatches"),
            )
            bestValidationLoss = min(bestValidationLoss, validationLoss)
            lastValidatedStep = globalStep
            print(
                f"epoch {epoch:02d} | val loss {validationLoss:.4f} | "
                f"val perplexity {validationPerplexity:.2f}"
            )

        if config.get("sampleDuringTraining", True) and lastSampledStep != globalStep:
            sampleText = generateSample(
                model,
                tokenizer,
                device,
                config["samplePrompt"],
                config,
            )
            lastSampledStep = globalStep
            print(f"sample: {sampleText}")

        if lastSavedStep != globalStep:
            saveCheckpoint(model, optimizer, config, epoch, globalStep)
            pruneCheckpoints(config)
            lastSavedStep = globalStep

        if stopTraining or globalStep >= totalSteps:
            break

    bestValidationText = (
        f"{bestValidationLoss:.4f}" if math.isfinite(bestValidationLoss) else "n/a"
    )
    print(f"Training complete. Best validation loss: {bestValidationText}")


def parseArgs():
    parser = argparse.ArgumentParser(description="Pretrain GPT-1 on TinyStories")
    parser.add_argument("--preset", choices=listPresets(), default="paper")
    parser.add_argument(
        "--resume",
        default=None,
        help='"latest", a step number, or a checkpoint path',
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--no-sample", action="store_true")
    args = parser.parse_args()

    config = getConfig(
        args.preset,
        dataDir=args.data_dir,
        weightsDir=args.weights_dir,
    )

    if args.epochs is not None:
        config["numEpochs"] = args.epochs
    if args.max_steps is not None:
        config["maxSteps"] = args.max_steps
    if args.micro_batch_size is not None:
        config["microBatchSize"] = args.micro_batch_size
    if args.grad_accum_steps is not None:
        config["gradAccumSteps"] = args.grad_accum_steps
    if args.lr is not None:
        config["lr"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    if args.resume is not None:
        config["preload"] = args.resume

    config["device"] = args.device
    config["sampleDuringTraining"] = not args.no_sample

    validateConfig(config)
    return config


if __name__ == "__main__":
    trainModel(parseArgs())
