from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def trainTokenizer(texts, outputPath, vocabSize: int, minFrequency: int = 1):
    if vocabSize <= len(SPECIAL_TOKENS) + len(ByteLevel.alphabet()):
        raise ValueError("vocabSize is too small for a ByteLevel BPE tokenizer")

    outputPath = Path(outputPath)
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocabSize,
        min_frequency=minFrequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    temporaryPath = outputPath.with_suffix(outputPath.suffix + ".tmp")
    tokenizer.save(str(temporaryPath))
    temporaryPath.replace(outputPath)

    return tokenizer


def loadTokenizer(config):
    tokenizerPath = Path(config["tokenizerFile"])
    if not tokenizerPath.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizerPath}. Run prepare_data.py first."
        )
    return Tokenizer.from_file(str(tokenizerPath))


def getTokenId(tokenizer, token: str):
    tokenId = tokenizer.token_to_id(token)
    if tokenId is None:
        raise ValueError(f"Tokenizer is missing special token {token}")
    return tokenId
