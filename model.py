import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, 
                 emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int, 
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, 
                src_mask: torch.Tensor, trg_mask: torch.Tensor, 
                src_padding_mask: torch.Tensor, trg_padding_mask: torch.Tensor, 
                memory_key_padding_mask: torch.Tensor):
        
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        
        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None, 
                                src_padding_mask, trg_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)),
                                        src_key_padding_mask=src_padding_mask)

    def decode(self, trg: torch.Tensor, memory: torch.Tensor, trg_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(trg)), memory,
                                        trg_mask=trg_mask)