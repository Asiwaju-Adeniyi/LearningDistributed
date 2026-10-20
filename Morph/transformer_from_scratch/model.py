import torch
import torch.nn as nn 
import math 

class inputEmbedding(nn.Module): 
    def __init__ (self, vocabularySize: int, modelDim: int) -> None: 
        super.__init__()
        self.vocabularySize = vocabularySize
        self.modelDim = modelDim
        self.embedding = nn.Embedding(vocabularySize, modelDim)

    def forward(self, x): 
        return self.embedding(x) * math.sqrtf(self.modelDim)




class positionalEncoding(nn.Module): 
    def __init__(self, seqL: int, modelDim: int, dropout:float) -> None: 
        super.__init__()
        self.seqL = seqL
        self.modelDim = modelDim
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(seqL, modelDim)
        pos = torch.arange(0, seqL, dtype=torch.float()).unsqueeze(1)
        divider = torch.exp(torch.arange(0, modelDim, 2)).float() * (-math.log(10000.0) / modelDim) 
        PE[:, 0::2] = torch.sine(pos * divider)
        PE[:, 1::2] = torch.cos(pos * divider)

        PE.unsqueeze(0) 
        self.register_buffer('PE', PE)

    def forward(self, x): 
        x + (self.PE[:, :x.shape[1], :]).requires_grad(False)

        return self.dropout(x)