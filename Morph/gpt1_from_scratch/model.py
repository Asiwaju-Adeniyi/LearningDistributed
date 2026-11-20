import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    # GPT-1 uses the tanh approximation of GELU.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.features = features
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, correction=0)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return (self.alpha * normalized) + self.beta


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dModel: int, numHeads: int, contextLen: int, dropout: float) -> None:
        super().__init__()

        if dModel % numHeads != 0:
            raise ValueError("dModel is not divisible by numHeads")

        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads

        self.wQ = nn.Linear(dModel, dModel)
        self.wK = nn.Linear(dModel, dModel)
        self.wV = nn.Linear(dModel, dModel)
        self.wO = nn.Linear(dModel, dModel)

        self.attentionDropout = nn.Dropout(dropout)
        self.residualDropout = nn.Dropout(dropout)

        causalMask = torch.tril(torch.ones(contextLen, contextLen, dtype=torch.bool))
        self.register_buffer(
            "causalMask",
            causalMask.view(1, 1, contextLen, contextLen),
            persistent=False,
        )

    def forward(self, x):
        batchSize, seqLen, _ = x.shape

        query = self.wQ(x)
        key = self.wK(x)
        value = self.wV(x)

        query = query.view(batchSize, seqLen, self.numHeads, self.dK).transpose(1, 2)
        key = key.view(batchSize, seqLen, self.numHeads, self.dK).transpose(1, 2)
        value = value.view(batchSize, seqLen, self.numHeads, self.dK).transpose(1, 2)

        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(self.dK)
        attentionScores = attentionScores.masked_fill(
            ~self.causalMask[:, :, :seqLen, :seqLen],
            torch.finfo(attentionScores.dtype).min,
        )
        attentionScores = F.softmax(attentionScores, dim=-1)
        attentionScores = self.attentionDropout(attentionScores)

        output = attentionScores @ value
        output = output.transpose(1, 2).contiguous().view(batchSize, seqLen, self.dModel)

        return self.residualDropout(self.wO(output))


class FeedForward(nn.Module):
    def __init__(self, dModel: int, dFf: int, dropout: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(dModel, dFf)
        self.linear2 = nn.Linear(dFf, dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(gelu(self.linear1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, dModel: int, numHeads: int, contextLen: int, dFf: int, dropout: float) -> None:
        super().__init__()

        self.norm1 = LayerNorm(dModel)
        self.attention = MultiHeadSelfAttention(dModel, numHeads, contextLen, dropout)
        self.norm2 = LayerNorm(dModel)
        self.feedForward = FeedForward(dModel, dFf, dropout)

    def forward(self, x):
        # Pre-LN residual blocks, as in GPT-1.
        x = x + self.attention(self.norm1(x))
        x = x + self.feedForward(self.norm2(x))
        return x


class GPT1(nn.Module):
    def __init__(
        self,
        vocabSize: int,
        contextLen: int = 512,
        dModel: int = 768,
        numLayers: int = 12,
        numHeads: int = 12,
        dFf: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocabSize = vocabSize
        self.contextLen = contextLen
        self.dModel = dModel

        self.tokenEmbedding = nn.Embedding(vocabSize, dModel)
        self.positionEmbedding = nn.Embedding(contextLen, dModel)
        self.embeddingDropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(dModel, numHeads, contextLen, dFf, dropout)
                for _ in range(numLayers)
            ]
        )
        self.finalNorm = LayerNorm(dModel)

        self.apply(self.initWeights)
        # GPT-1 used std 0.02 for token embeddings and std 0.01 for positions.
        nn.init.normal_(self.positionEmbedding.weight, mean=0.0, std=0.01)

    @staticmethod
    def initWeights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, seqLen = idx.shape
        if seqLen > self.contextLen:
            raise ValueError(f"Sequence length {seqLen} exceeds context length {self.contextLen}")

        positions = torch.arange(seqLen, device=idx.device, dtype=torch.long)
        x = self.tokenEmbedding(idx) + self.positionEmbedding(positions)
        x = self.embeddingDropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.finalNorm(x)

        # GPT-1 ties the output projection to the token embedding matrix.
        logits = F.linear(x, self.tokenEmbedding.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, maxNewTokens: int, temperature: float = 1.0, topK=None):
        wasTraining = self.training
        self.eval()

        try:
            for _ in range(maxNewTokens):
                idxConditioned = idx[:, -self.contextLen:]
                logits, _ = self(idxConditioned)
                logits = logits[:, -1, :]

                if temperature <= 0.0:
                    nextToken = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if topK is not None:
                        topK = min(topK, logits.size(-1))
                        topValues, _ = torch.topk(logits, topK)
                        threshold = topValues[:, [-1]]
                        logits = torch.where(
                            logits < threshold,
                            torch.full_like(logits, float("-inf")),
                            logits,
                        )

                    probabilities = F.softmax(logits, dim=-1)
                    nextToken = torch.multinomial(probabilities, num_samples=1)

                idx = torch.cat([idx, nextToken], dim=1)
        finally:
            self.train(wasTraining)

        return idx


def buildGpt1(config, vocabSize=None):
    if vocabSize is None:
        vocabSize = config["vocabSize"]

    return GPT1(
        vocabSize=vocabSize,
        contextLen=config["contextLen"],
        dModel=config["dModel"],
        numLayers=config["numLayers"],
        numHeads=config["numHeads"],
        dFf=config["dFf"],
        dropout=config["dropout"],
    )


def countParameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def estimateParameterCount(vocabSize: int, contextLen: int, dModel: int, numLayers: int, dFf: int):
    # The output projection is tied to the token embedding, so it contributes
    # no extra parameters.
    tokenEmbedding = vocabSize * dModel
    positionEmbedding = contextLen * dModel
    attention = numLayers * (4 * dModel * dModel + 4 * dModel)
    feedForward = numLayers * (2 * dModel * dFf + dFf + dModel)
    layerNorm = numLayers * 2 * (2 * dModel) + 2 * dModel
    return tokenEmbedding + positionEmbedding + attention + feedForward + layerNorm
