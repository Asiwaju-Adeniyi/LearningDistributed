import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# GPT-1 architecture and optimization hyperparameters from the paper.
COMMON_GPT1 = {
    "contextLen": 512,
    "dModel": 768,
    "numLayers": 12,
    "numHeads": 12,
    "dFf": 3072,
    "dropout": 0.1,
    "vocabSize": 40000,
    "tokenizerMinFrequency": 1,
    "microBatchSize": 8,
    "gradAccumSteps": 8,
    "valBatchSize": 16,
    "lr": 2.5e-4,
    "warmupSteps": 2000,
    "weightDecay": 0.01,
    "gradClip": 1.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "seed": 42,
    "numWorkers": 4,
    "logEvery": 10,
    "valEvery": 500,
    "saveEvery": 1000,
    "keepCheckpoints": 2,
    "sampleEvery": 1000,
    "samplePrompt": "Once upon a time",
    "sampleTokens": 80,
    "sampleTemperature": 0.8,
    "sampleTopK": 40,
    "maxValBatches": None,
    "tokenizerSampleSize": 200000,
    "maxTrainStories": None,
    "maxValStories": None,
}

PRESETS = {
    # Full paper schedule on the paper architecture. Intentionally long:
    # the paper trained for 100 epochs on BooksCorpus.
    "paper": {
        **COMMON_GPT1,
        "numEpochs": 100,
        "maxSteps": None,
        "valEvery": 1000,
        "saveEvery": 2000,
        "sampleEvery": 2000,
    },
    # Same architecture and optimizer, but a practical TinyStories run.
    "run": {
        **COMMON_GPT1,
        "numEpochs": 3,
        "maxSteps": 20000,
        "valEvery": 500,
        "saveEvery": 500,
        "sampleEvery": 1000,
    },
    # Tiny model/data for local checks.
    "smoke": {
        "contextLen": 128,
        "dModel": 256,
        "numLayers": 4,
        "numHeads": 4,
        "dFf": 1024,
        "dropout": 0.0,
        "vocabSize": 2000,
        "tokenizerMinFrequency": 1,
        "microBatchSize": 4,
        "gradAccumSteps": 4,
        "valBatchSize": 8,
        "lr": 1e-3,
        "warmupSteps": 5,
        "weightDecay": 0.01,
        "gradClip": 1.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "seed": 42,
        "numWorkers": 0,
        "logEvery": 1,
        "valEvery": 10,
        "saveEvery": 10,
        "keepCheckpoints": 1,
        "sampleEvery": 10,
        "samplePrompt": "Once upon a time",
        "sampleTokens": 20,
        "sampleTemperature": 0.8,
        "sampleTopK": 40,
        "maxValBatches": 10,
        "tokenizerSampleSize": 2000,
        "maxTrainStories": 2000,
        "maxValStories": 200,
        "numEpochs": 1,
        "maxSteps": 20,
    },
}


def listPresets():
    return list(PRESETS.keys())


def validateConfig(config):
    if config["dModel"] % config["numHeads"] != 0:
        raise ValueError("dModel must be divisible by numHeads")
    if config["contextLen"] <= 0:
        raise ValueError("contextLen must be positive")
    if config["microBatchSize"] <= 0 or config["gradAccumSteps"] <= 0:
        raise ValueError("microBatchSize and gradAccumSteps must be positive")
    if config["numLayers"] <= 0 or config["numHeads"] <= 0:
        raise ValueError("numLayers and numHeads must be positive")
    if config["vocabSize"] > 65536:
        raise ValueError("vocabSize must fit in uint16 token files")
    if config["numEpochs"] <= 0:
        raise ValueError("numEpochs must be positive")
    if config.get("maxSteps") is not None and config["maxSteps"] <= 0:
        raise ValueError("maxSteps must be positive")
    if config["keepCheckpoints"] <= 0:
        raise ValueError("keepCheckpoints must be positive")
    if config["tokenizerSampleSize"] is not None and config["tokenizerSampleSize"] <= 0:
        raise ValueError("tokenizerSampleSize must be positive")
    if config["tokenizerMinFrequency"] <= 0:
        raise ValueError("tokenizerMinFrequency must be positive")
    for intervalKey in ["logEvery", "valEvery", "saveEvery", "sampleEvery"]:
        if config[intervalKey] <= 0:
            raise ValueError(f"{intervalKey} must be positive")
    if config["warmupSteps"] < 0:
        raise ValueError("warmupSteps must be non-negative")


def getConfig(preset="run", dataDir=None, weightsDir=None):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {listPresets()}")

    config = dict(PRESETS[preset])
    config["preset"] = preset

    if dataDir is None:
        # paper and run use the same tokenizer/token files. Keep the tiny
        # smoke-test dataset separate so it cannot replace a real run.
        defaultDataDir = BASE_DIR / "data"
        if preset == "smoke":
            defaultDataDir = defaultDataDir / "smoke"
        resolvedDataDir = defaultDataDir
    else:
        resolvedDataDir = Path(dataDir)

    if weightsDir is None:
        resolvedWeightsDir = BASE_DIR / "weights" / preset
    else:
        resolvedWeightsDir = Path(weightsDir)

    config["dataDir"] = str(resolvedDataDir)
    config["weightsDir"] = str(resolvedWeightsDir)
    config["tokenizerFile"] = str(resolvedDataDir / "tokenizer.json")
    config["trainBin"] = str(resolvedDataDir / "train.bin")
    config["valBin"] = str(resolvedDataDir / "val.bin")
    config["metaFile"] = str(resolvedDataDir / "meta.json")
    config["modelBasename"] = "gpt1_step_"
    config["preload"] = None
    config["device"] = None
    config["sampleDuringTraining"] = True

    validateConfig(config)
    return config


def ensureDirectories(config):
    Path(config["dataDir"]).mkdir(parents=True, exist_ok=True)
    Path(config["weightsDir"]).mkdir(parents=True, exist_ok=True)


def getPrepareSignature(config):
    # The preset name is deliberately excluded: "paper" and "run" use the
    # same tokenizer and token files. Only data-affecting settings matter.
    return {
        "vocabSize": config["vocabSize"],
        "tokenizerMinFrequency": config.get("tokenizerMinFrequency", 2),
        "tokenizerSampleSize": config.get("tokenizerSampleSize"),
        "maxTrainStories": config.get("maxTrainStories"),
        "maxValStories": config.get("maxValStories"),
    }


def checkPreparedData(config):
    metaFile = Path(config["metaFile"])
    if not metaFile.exists():
        raise FileNotFoundError(
            f"Prepared data metadata not found at {metaFile}. Run prepare_data.py first."
        )

    try:
        metadata = json.loads(metaFile.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read prepared data metadata at {metaFile}") from error

    if metadata.get("prepareSignature") != getPrepareSignature(config):
        raise ValueError(
            "Prepared data does not match this preset/config. "
            f"Run: python prepare_data.py --preset {config['preset']}"
        )


def getWeightsFilePath(config, step):
    modelFolder = Path(config["weightsDir"])
    modelFilename = f"{config['modelBasename']}{int(step):07d}.pt"
    return str(modelFolder / modelFilename)


def listCheckpoints(config):
    modelFolder = Path(config["weightsDir"])
    if not modelFolder.exists():
        return []

    def checkpointStep(path):
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return sorted(modelFolder.glob(f"{config['modelBasename']}*.pt"), key=checkpointStep)


def getLatestWeightsFilePath(config):
    checkpoints = listCheckpoints(config)
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def resolveCheckpointPath(config, checkpoint):
    if checkpoint is None or checkpoint == "latest":
        modelFilename = getLatestWeightsFilePath(config)
    else:
        checkpointPath = Path(str(checkpoint))
        if checkpointPath.suffix == ".pt":
            modelFilename = str(checkpointPath)
        else:
            modelFilename = getWeightsFilePath(config, int(checkpoint))

    if modelFilename is None or not Path(modelFilename).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {modelFilename}. "
            f"Looked in {config['weightsDir']}."
        )

    return modelFilename
