import argparse
import json
from array import array
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from config import ensureDirectories, getConfig, getPrepareSignature, listPresets
from tokenizer import getTokenId, loadTokenizer, trainTokenizer


DATASET_NAME = "roneneldan/TinyStories"
TOKEN_BYTES = 2
TOKEN_BUFFER_SIZE = 1_000_000
TOKENIZE_BATCH_SIZE = 1024


def getDatasetSplits(config):
    trainSplit = load_dataset(DATASET_NAME, split="train")
    valSplit = load_dataset(DATASET_NAME, split="validation")

    maxTrainStories = config.get("maxTrainStories")
    if maxTrainStories is not None:
        trainSplit = trainSplit.select(range(min(maxTrainStories, len(trainSplit))))

    maxValStories = config.get("maxValStories")
    if maxValStories is not None:
        valSplit = valSplit.select(range(min(maxValStories, len(valSplit))))

    return trainSplit, valSplit


def textIterator(dataset, limit=None, batchSize: int = 1000):
    if limit is None:
        limit = len(dataset)
    else:
        limit = min(limit, len(dataset))

    for start in range(0, limit, batchSize):
        end = min(start + batchSize, limit)
        yield from dataset[start:end]["text"]


def tokenizeSplit(dataset, tokenizer, outputPath, eosTokenId: int):
    outputPath = Path(outputPath)
    temporaryPath = Path(str(outputPath) + ".tmp")
    temporaryPath.unlink(missing_ok=True)

    totalStories = len(dataset)
    tokenCount = 0
    buffer = array("H")
    if buffer.itemsize != TOKEN_BYTES:
        raise RuntimeError("This platform does not use 2-byte unsigned shorts")

    try:
        with open(temporaryPath, "wb") as outputFile:
            progress = tqdm(
                total=totalStories,
                desc=f"Tokenizing {outputPath.name}",
                unit="story",
            )

            try:
                for start in range(0, totalStories, TOKENIZE_BATCH_SIZE):
                    end = min(start + TOKENIZE_BATCH_SIZE, totalStories)
                    batch = dataset[start:end]["text"]
                    encodings = tokenizer.encode_batch(batch)

                    for encoding in encodings:
                        buffer.extend(encoding.ids)
                        buffer.append(eosTokenId)
                        tokenCount += len(encoding.ids) + 1

                    if len(buffer) >= TOKEN_BUFFER_SIZE:
                        buffer.tofile(outputFile)
                        buffer = array("H")

                    progress.update(len(batch))
            finally:
                progress.close()

            if len(buffer) > 0:
                buffer.tofile(outputFile)
    except Exception:
        temporaryPath.unlink(missing_ok=True)
        raise

    return tokenCount, temporaryPath


def writeMetadata(metaFile, metadata):
    metaFile = Path(metaFile)
    temporaryPath = Path(str(metaFile) + ".tmp")
    temporaryPath.write_text(json.dumps(metadata, indent=2) + "\n")
    temporaryPath.replace(metaFile)


def prepareData(config, force: bool = False):
    ensureDirectories(config)

    tokenizerPath = Path(config["tokenizerFile"])
    trainBin = Path(config["trainBin"])
    valBin = Path(config["valBin"])
    metaFile = Path(config["metaFile"])
    prepareSignature = getPrepareSignature(config)

    metadata = {}
    if metaFile.exists():
        try:
            metadata = json.loads(metaFile.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}

    dataIsReady = (
        tokenizerPath.exists()
        and trainBin.exists()
        and valBin.exists()
        and trainBin.stat().st_size >= TOKEN_BYTES * (config["contextLen"] + 1)
        and valBin.stat().st_size >= TOKEN_BYTES * (config["contextLen"] + 1)
        and metaFile.exists()
        and metadata.get("prepareSignature") == prepareSignature
    )
    if dataIsReady and not force:
        print(f"Data already prepared in {config['dataDir']}")
        return

    retrainTokenizer = (
        force
        or not tokenizerPath.exists()
        or metadata.get("prepareSignature") != prepareSignature
    )

    trainSplit, valSplit = getDatasetSplits(config)
    print(f"Train stories: {len(trainSplit):,}")
    print(f"Validation stories: {len(valSplit):,}")

    if retrainTokenizer:
        print(f"Training a {config['vocabSize']:,}-token BPE tokenizer...")
        tokenizer = trainTokenizer(
            textIterator(trainSplit, config.get("tokenizerSampleSize")),
            tokenizerPath,
            config["vocabSize"],
            config.get("tokenizerMinFrequency", 1),
        )
    else:
        print(f"Loading tokenizer from {tokenizerPath}")
        tokenizer = loadTokenizer(config)

    vocabSize = tokenizer.get_vocab_size()
    if vocabSize > 65536:
        raise ValueError(
            f"Vocabulary size {vocabSize} does not fit in uint16 token files"
        )
    if vocabSize < config["vocabSize"]:
        print(
            f"Note: the corpus only produced {vocabSize:,} tokens "
            f"(target was {config['vocabSize']:,})."
        )

    eosTokenId = getTokenId(tokenizer, "[EOS]")
    print(f"Vocabulary size: {vocabSize:,}")

    trainTemporaryPath = Path(str(trainBin) + ".tmp")
    valTemporaryPath = Path(str(valBin) + ".tmp")

    try:
        print("Tokenizing the training split...")
        trainTokens, trainTemporaryPath = tokenizeSplit(
            trainSplit,
            tokenizer,
            trainBin,
            eosTokenId,
        )
        print(f"Training tokens: {trainTokens:,}")

        print("Tokenizing the validation split...")
        valTokens, valTemporaryPath = tokenizeSplit(
            valSplit,
            tokenizer,
            valBin,
            eosTokenId,
        )
        print(f"Validation tokens: {valTokens:,}")

        trainTemporaryPath.replace(trainBin)
        valTemporaryPath.replace(valBin)
    except Exception:
        trainTemporaryPath.unlink(missing_ok=True)
        valTemporaryPath.unlink(missing_ok=True)
        raise

    metadata = {
        "dataset": DATASET_NAME,
        "tokenDtype": "uint16",
        "prepareSignature": prepareSignature,
        "tokenizerTargetVocabSize": config["vocabSize"],
        "vocabSize": vocabSize,
        "trainTokens": trainTokens,
        "valTokens": valTokens,
        "eosTokenId": eosTokenId,
        "trainStories": len(trainSplit),
        "valStories": len(valSplit),
        "contextLen": config["contextLen"],
        "tokenizerSampleSize": config.get("tokenizerSampleSize"),
        "tokenizerMinFrequency": config.get("tokenizerMinFrequency", 1),
        "maxTrainStories": config.get("maxTrainStories"),
        "maxValStories": config.get("maxValStories"),
    }
    writeMetadata(metaFile, metadata)
    print(f"Wrote metadata to {metaFile}")


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories data for GPT-1 pretraining"
    )
    parser.add_argument("--preset", choices=listPresets(), default="paper")
    parser.add_argument("--data-dir", default=None, help="Override the data directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain the tokenizer and rewrite the token files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    config = getConfig(args.preset, dataDir=args.data_dir)
    prepareData(config, force=args.force)
