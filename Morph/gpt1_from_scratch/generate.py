import argparse

import torch

from config import getConfig, listPresets, resolveCheckpointPath
from model import buildGpt1
from tokenizer import getTokenId, loadTokenizer


def getDevice(preferred=None):
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate text with a pretrained GPT-1 checkpoint"
    )
    parser.add_argument("--preset", choices=listPresets(), default="paper")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help='"latest", a step number, or a checkpoint path',
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main():
    args = parseArgs()
    config = getConfig(args.preset, dataDir=args.data_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = getDevice(args.device)
    tokenizer = loadTokenizer(config)
    vocabSize = tokenizer.get_vocab_size()

    modelFilename = resolveCheckpointPath(config, args.checkpoint)
    checkpoint = torch.load(modelFilename, map_location=device, weights_only=True)

    checkpointVocabSize = checkpoint["modelStateDict"]["tokenEmbedding.weight"].shape[0]
    if checkpointVocabSize != vocabSize:
        raise ValueError(
            f"Tokenizer vocabulary size {vocabSize} does not match checkpoint "
            f"vocabulary size {checkpointVocabSize}. Use the tokenizer that was "
            "trained with this checkpoint."
        )

    savedConfig = checkpoint.get("config")
    if savedConfig is not None:
        for key in [
            "contextLen",
            "dModel",
            "numLayers",
            "numHeads",
            "dFf",
            "dropout",
        ]:
            config[key] = savedConfig[key]

    model = buildGpt1(config, vocabSize).to(device)
    model.load_state_dict(checkpoint["modelStateDict"])
    model.eval()

    globalStep = checkpoint.get("globalStep", 0)
    print(f"Loaded {modelFilename} (step {globalStep:,})")

    promptTokens = tokenizer.encode(args.prompt).ids if args.prompt else []
    if not promptTokens:
        promptTokens = [getTokenId(tokenizer, "[BOS]")]

    temperature = (
        args.temperature
        if args.temperature is not None
        else config["sampleTemperature"]
    )
    topK = args.top_k if args.top_k is not None else config["sampleTopK"]
    if topK is not None and topK <= 0:
        topK = None

    idx = torch.tensor([promptTokens], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        maxNewTokens=args.max_new_tokens,
        temperature=temperature,
        topK=topK,
    )

    text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
