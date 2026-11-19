from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def getAllSentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def getOrBuildTokenizer(config, ds, lang):
    tokenizerPath = Path(config["tokenizerFile"].format(lang))

    if tokenizerPath.exists():
        tokenizer = Tokenizer.from_file(str(tokenizerPath))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(getAllSentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizerPath))

    return tokenizer


def causalMask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizerSrc, tokenizerTgt, langSrc, langTgt, seqLen) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizerSrc = tokenizerSrc
        self.tokenizerTgt = tokenizerTgt
        self.langSrc = langSrc
        self.langTgt = langTgt
        self.seqLen = seqLen

        self.srcSosToken = torch.tensor([tokenizerSrc.token_to_id("[SOS]")], dtype=torch.int64)
        self.srcEosToken = torch.tensor([tokenizerSrc.token_to_id("[EOS]")], dtype=torch.int64)
        self.srcPadToken = torch.tensor([tokenizerSrc.token_to_id("[PAD]")], dtype=torch.int64)

        self.tgtSosToken = torch.tensor([tokenizerTgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.tgtEosToken = torch.tensor([tokenizerTgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.tgtPadToken = torch.tensor([tokenizerTgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        srcTargetPair = self.ds[index]
        srcText = srcTargetPair['translation'][self.langSrc]
        tgtText = srcTargetPair['translation'][self.langTgt]

        encInputTokens = self.tokenizerSrc.encode(srcText).ids
        decInputTokens = self.tokenizerTgt.encode(tgtText).ids

        encoderInput = torch.cat([
            self.srcSosToken,
            torch.tensor(encInputTokens[:self.seqLen - 2], dtype=torch.int64),
            self.srcEosToken,
        ])

        decoderInput = torch.cat([
            self.tgtSosToken,
            torch.tensor(decInputTokens[:self.seqLen - 2], dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(decInputTokens[:self.seqLen - 2], dtype=torch.int64),
            self.tgtEosToken,
        ])

        numPaddingTokensEncoder = self.seqLen - len(encoderInput)
        if numPaddingTokensEncoder > 0:
            encoderInput = torch.cat([encoderInput, self.srcPadToken.repeat(numPaddingTokensEncoder)])

        numPaddingTokensDecoder = self.seqLen - len(decoderInput)
        if numPaddingTokensDecoder > 0:
            decoderInput = torch.cat([decoderInput, self.tgtPadToken.repeat(numPaddingTokensDecoder)])
            label = torch.cat([label, self.tgtPadToken.repeat(numPaddingTokensDecoder)])

        assert encoderInput.size(0) == self.seqLen
        assert decoderInput.size(0) == self.seqLen
        assert label.size(0) == self.seqLen

        return {
            "encoderInput": encoderInput,
            "decoderInput": decoderInput,
            "encoderMask": (encoderInput != self.srcPadToken).unsqueeze(0).unsqueeze(0).int(),
            "decoderMask": (decoderInput != self.tgtPadToken).unsqueeze(0).unsqueeze(0).int() & causalMask(decoderInput.size(0)).int(),
            "label": label,
            "srcText": srcText,
            "tgtText": tgtText,
        }


def getDs(config):
    dsRaw = load_dataset("Helsinki-NLP/opus_books", f"{config['langSrc']}-{config['langTgt']}", split='train')
    dsRaw = dsRaw.train_test_split(test_size=0.1, seed=42)

    trainDsRaw = dsRaw['train']
    valDsRaw = dsRaw['test']

    tokenizerSrc = getOrBuildTokenizer(config, trainDsRaw, config["langSrc"])
    tokenizerTgt = getOrBuildTokenizer(config, trainDsRaw, config["langTgt"])

    trainDs = BilingualDataset(
        trainDsRaw,
        tokenizerSrc,
        tokenizerTgt,
        config["langSrc"],
        config["langTgt"],
        config["seqLen"],
    )
    valDs = BilingualDataset(
        valDsRaw,
        tokenizerSrc,
        tokenizerTgt,
        config["langSrc"],
        config["langTgt"],
        config["seqLen"],
    )

    maxLenSrc = 0
    maxLenTgt = 0
    for item in trainDsRaw:
        srcIds = tokenizerSrc.encode(item['translation'][config["langSrc"]]).ids
        tgtIds = tokenizerTgt.encode(item['translation'][config["langTgt"]]).ids
        maxLenSrc = max(maxLenSrc, len(srcIds))
        maxLenTgt = max(maxLenTgt, len(tgtIds))

    print(f"Max source sentence length: {maxLenSrc}")
    print(f"Max target sentence length: {maxLenTgt}")

    return trainDs, valDs, tokenizerSrc, tokenizerTgt
