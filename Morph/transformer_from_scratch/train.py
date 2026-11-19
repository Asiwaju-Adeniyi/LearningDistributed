from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import getConfig, getWeightsFilePath, getLatestWeightsFilePath
from dataset import causalMask, getDs
from model import build_transformer


def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def getModel(config, vocabSrcSize, vocabTgtSize):
    return build_transformer(
        vocabSrcSize,
        vocabTgtSize,
        config["seqLen"],
        config["seqLen"],
        config["dModel"],
        config["N"],
        config["h"],
        config["dropout"],
        config["dFf"],
    )


def getDataloader(trainDs, valDs, config):
    trainDataloader = DataLoader(
        trainDs,
        batch_size=config["batchSize"],
        shuffle=True,
        num_workers=0,
    )
    valDataloader = DataLoader(
        valDs,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return trainDataloader, valDataloader


def greedyDecode(model, source, sourceMask, tokenizerTgt, maxLen, device):
    model.eval()

    with torch.no_grad():
        encoderOutput = model.encode(source, sourceMask)
        decoderInput = torch.empty(1, 1).fill_(tokenizerTgt.token_to_id("[SOS]")).type_as(source).to(device)

        while decoderInput.size(1) < maxLen:
            decoderMask = causalMask(decoderInput.size(1)).type_as(sourceMask).to(device)
            out = model.decode(encoderOutput, sourceMask, decoderInput, decoderMask)
            prob = model.project(out[:, -1, :])
            _, nextWord = torch.max(prob, dim=1)

            decoderInput = torch.cat(
                [decoderInput, torch.empty(1, 1).fill_(nextWord.item()).type_as(source).to(device)],
                dim=1,
            )

            if nextWord.item() == tokenizerTgt.token_to_id("[EOS]"):
                break

    return tokenizerTgt.decode(decoderInput.squeeze(0).detach().cpu().tolist(), skip_special_tokens=True)


def runValidation(model, valDataloader, tokenizerTgt, lossFn, device):
    model.eval()
    totalLoss = 0.0

    with torch.no_grad():
        for batch in valDataloader:
            encoderInput = batch["encoderInput"].to(device)
            decoderInput = batch["decoderInput"].to(device)
            encoderMask = batch["encoderMask"].to(device)
            decoderMask = batch["decoderMask"].to(device)
            label = batch["label"].to(device)

            encoderOutput = model.encode(encoderInput, encoderMask)
            decoderOutput = model.decode(encoderOutput, encoderMask, decoderInput, decoderMask)
            projectionOutput = model.project(decoderOutput)

            loss = lossFn(projectionOutput.view(-1, tokenizerTgt.get_vocab_size()), label.view(-1))
            totalLoss += loss.item()

    return totalLoss / len(valDataloader)


def saveCheckpoint(model, optimizer, epoch, globalStep, config):
    modelFilename = getWeightsFilePath(config, f"{epoch:02d}")
    torch.save(
        {
            "epoch": epoch,
            "globalStep": globalStep,
            "modelStateDict": model.state_dict(),
            "optimizerStateDict": optimizer.state_dict(),
        },
        modelFilename,
    )
    print(f"Saved checkpoint {modelFilename}")


def loadCheckpoint(model, optimizer, config, device):
    preload = config["preload"]
    if preload is None:
        return 0, 0

    if preload == "latest":
        modelFilename = getLatestWeightsFilePath(config)
    else:
        modelFilename = getWeightsFilePath(config, preload)

    if modelFilename is None:
        print("No checkpoint found to preload")
        return 0, 0

    print(f"Preloading model {modelFilename}")
    checkpoint = torch.load(modelFilename, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["modelStateDict"])
    optimizer.load_state_dict(checkpoint["optimizerStateDict"])

    return checkpoint["epoch"] + 1, checkpoint["globalStep"]


def trainModel(config):
    device = getDevice()
    print(f"Using device: {device}")

    Path(config["modelFolder"]).mkdir(parents=True, exist_ok=True)

    trainDs, valDs, tokenizerSrc, tokenizerTgt = getDs(config)
    trainDataloader, valDataloader = getDataloader(trainDs, valDs, config)

    model = getModel(config, tokenizerSrc.get_vocab_size(), tokenizerTgt.get_vocab_size()).to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    lossFn = nn.CrossEntropyLoss(
        ignore_index=tokenizerTgt.token_to_id("[PAD]"),
        label_smoothing=config["labelSmoothing"],
    ).to(device)

    initialEpoch, globalStep = loadCheckpoint(model, optimizer, config, device)

    for epoch in range(initialEpoch, config["numEpochs"]):
        model.train()
        epochLoss = 0.0

        batchIterator = tqdm(trainDataloader, desc=f"Epoch {epoch:02d}", unit="batch")
        for batch in batchIterator:
            encoderInput = batch["encoderInput"].to(device)
            decoderInput = batch["decoderInput"].to(device)
            encoderMask = batch["encoderMask"].to(device)
            decoderMask = batch["decoderMask"].to(device)
            label = batch["label"].to(device)

            encoderOutput = model.encode(encoderInput, encoderMask)
            decoderOutput = model.decode(encoderOutput, encoderMask, decoderInput, decoderMask)
            projectionOutput = model.project(decoderOutput)

            loss = lossFn(projectionOutput.view(-1, tokenizerTgt.get_vocab_size()), label.view(-1))
            batchIterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epochLoss += loss.item()
            globalStep += 1

        epochLoss /= len(trainDataloader)
        print(f"Epoch {epoch:02d} training loss: {epochLoss:.4f}")

        valLoss = runValidation(model, valDataloader, tokenizerTgt, lossFn, device)
        print(f"Epoch {epoch:02d} validation loss: {valLoss:.4f}")

        saveCheckpoint(model, optimizer, epoch, globalStep, config)

        model.eval()
        for i in range(min(config["numExamples"], len(valDs))):
            sample = valDs[i]
            source = sample["encoderInput"].unsqueeze(0).to(device)
            sourceMask = sample["encoderMask"].unsqueeze(0).to(device)

            translated = greedyDecode(model, source, sourceMask, tokenizerTgt, config["seqLen"], device)

            print(f"SOURCE: {sample['srcText']}")
            print(f"TARGET: {sample['tgtText']}")
            print(f"PREDICTED: {translated}\n")


if __name__ == "__main__":
    config = getConfig()
    trainModel(config)
