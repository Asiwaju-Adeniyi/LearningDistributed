import torch
import torch.nn as nn 
import math 

class inputEmbedding(nn.Module): 
    def __init__ (self, vocabularySize: int, modelDim: int) -> None: 
        super().__init__()
        self.vocabularySize = vocabularySize
        self.modelDim = modelDim
        self.embedding = nn.Embedding(vocabularySize, modelDim)

    def forward(self, x): 
        return self.embedding(x) * math.sqrt(self.modelDim)


class positionalEncoding(nn.Module): 
    def __init__(self, seqL: int, modelDim: int, dropout:float) -> None: 
        super().__init__()
        self.seqL = seqL
        self.modelDim = modelDim
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(seqL, modelDim)
        pos = torch.arange(0, seqL, dtype=torch.float32).unsqueeze(1)
        divider = torch.exp(torch.arange(0, modelDim, 2).float() * (-math.log(10000.0) / modelDim)) 
        PE[:, 0::2] = torch.sin(pos * divider)
        PE[:, 1::2] = torch.cos(pos * divider)

        PE = PE.unsqueeze(0) 
        self.register_buffer('PE', PE)

    def forward(self, x): 
        x = x + self.PE[:, :x.shape[1], :]

        return self.dropout(x)



class FeedForward(nn.Module): 
    def __init__(self, FFd: int, modelDim: int, dropout: float) -> None: 
        super().__init__()
        self.linear1 = nn.Linear(modelDim, FFd) 
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(FFd, modelDim)

    def forward(self, x): 
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class LayerNorm(nn.Module): 
    def __init__(self, features: int, eps:float=10**-6)-> None:
        super().__init__()

        self.features = features
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward (self, x): 
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, correction = 0)
        x = (x - mean) / torch.sqrt(variance + self.eps)

        return (self.alpha * x) + self.beta

class ResidualConnection(nn.Module): 
    def __init__(self, features: int, dropout: float) -> None: 
        super().__init__()

        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): 
        return x + self.dropout(sublayer(self.norm(x)))


class ProjectionLayer(nn.Module):
    def __init__ (self, d_model: int, vocab_size: int) -> None: 
        super().__init__()

        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x): 
        return self.project(x)


class MhaBlock(nn.Module): 
    def __init__ (self, d_model: int, h : int, dropout: float) -> None: 

        super().__init__()
        self.h = h
        self.d_model = d_model

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model, bias = False) 
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False) 
        self.w_o = nn.Linear(d_model, d_model, bias = False) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod 
    def attention(query, key, value, mask, dropout: nn.Dropout): 
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: 
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None: 
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask): 
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MhaBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MhaBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MhaBlock, cross_attention_block: MhaBlock, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: inputEmbedding, tgt_embed: inputEmbedding, src_pos: positionalEncoding, tgt_pos: positionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = inputEmbedding(src_vocab_size, d_model)
    tgt_embed = inputEmbedding(tgt_vocab_size, d_model)

    # Create the positional encoding layers
    src_pos = positionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos = positionalEncoding(tgt_seq_len, d_model, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MhaBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_ff, d_model, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MhaBlock(d_model, h, dropout)
        decoder_cross_attention_block = MhaBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_ff, d_model, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer