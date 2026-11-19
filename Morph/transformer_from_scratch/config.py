from pathlib import Path


def getConfig():
    return {
        "batchSize": 8,
        "numEpochs": 20,
        "lr": 1e-4,
        "seqLen": 350,
        "dModel": 512,
        "N": 6,
        "h": 8,
        "dropout": 0.1,
        "dFf": 2048,
        "labelSmoothing": 0.1,
        "numExamples": 2,
        "langSrc": "en",
        "langTgt": "fr",
        "modelFolder": "weights",
        "modelBasename": "tmodel_",
        "tokenizerFile": "tokenizer_{0}.json",
        # Set to "latest" to resume from the most recent checkpoint,
        # or to an epoch string such as "03" to resume from that checkpoint.
        "preload": None,
    }


def getWeightsFilePath(config, epoch: str):
    modelFolder = Path(config["modelFolder"])
    modelFilename = f"{config['modelBasename']}{epoch}.pt"
    return str(modelFolder / modelFilename)


def getLatestWeightsFilePath(config):
    modelFolder = Path(config["modelFolder"])
    if not modelFolder.exists():
        return None

    checkpoints = list(modelFolder.glob(f"{config['modelBasename']}*.pt"))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
    return str(latest)
