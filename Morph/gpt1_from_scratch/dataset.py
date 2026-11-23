from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import checkPreparedData


class CausalLmDataset(Dataset):
    def __init__(self, binPath, contextLen: int) -> None:
        super().__init__()

        self.binPath = str(binPath)
        self.contextLen = contextLen

        tokenFile = Path(self.binPath)
        if not tokenFile.exists():
            raise FileNotFoundError(
                f"Token file not found at {tokenFile}. Run prepare_data.py first."
            )

        fileSize = tokenFile.stat().st_size
        if fileSize % 2 != 0:
            raise ValueError(
                f"Token file {tokenFile} does not contain complete uint16 tokens"
            )

        self.numTokens = fileSize // 2
        self.numSamples = (self.numTokens - 1) // contextLen
        if self.numSamples <= 0:
            raise ValueError(
                f"Not enough tokens in {tokenFile} to form a context of {contextLen}"
            )

        self.tokens = None

    def getTokens(self):
        # Open lazily so each DataLoader worker gets its own memory map.
        if self.tokens is None:
            self.tokens = np.memmap(self.binPath, dtype=np.uint16, mode="r")
        return self.tokens

    def __len__(self):
        return self.numSamples

    def __getitem__(self, index):
        start = index * self.contextLen
        chunk = np.asarray(
            self.getTokens()[start : start + self.contextLen + 1],
            dtype=np.int64,
        )

        inputs = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])
        return inputs, labels


def getDataLoaders(config):
    checkPreparedData(config)

    trainDataset = CausalLmDataset(config["trainBin"], config["contextLen"])
    valDataset = CausalLmDataset(config["valBin"], config["contextLen"])

    numWorkers = config.get("numWorkers", 0)
    pinMemory = torch.cuda.is_available()

    trainDataloader = DataLoader(
        trainDataset,
        batch_size=config["microBatchSize"],
        shuffle=True,
        drop_last=True,
        num_workers=numWorkers,
        pin_memory=pinMemory,
        persistent_workers=numWorkers > 0,
    )
    valDataloader = DataLoader(
        valDataset,
        batch_size=config.get("valBatchSize", config["microBatchSize"]),
        shuffle=False,
        drop_last=False,
        num_workers=numWorkers,
        pin_memory=pinMemory,
        persistent_workers=numWorkers > 0,
    )

    return trainDataloader, valDataloader
