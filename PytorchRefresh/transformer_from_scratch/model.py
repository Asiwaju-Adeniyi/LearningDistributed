import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module) :
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        def forward(self, x): 
            return self.embedding(x) * math.sqrt(d_model)



class PositionalEncoding(nn.Module): 
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: 
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #Apply the sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, Seq_len, d_model) 

        self.register_buffer('pe', pe)

    def forward(self, x): 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)


class layerNormalization(nn.Module): 
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__ 
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added 

    def forward(self, x): 
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x-mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module): 
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
        super.__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #w2 and B2

    def forward(self, x): 
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff) --> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module): 
    def __int__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h #head dimension 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout): 
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    def forward(self, q, k, v, mask): 
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)


        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(key.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)


