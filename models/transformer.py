import torch
import torch.nn as nn
from .BasicModule import BasicModule

class transformer(BasicModule):
    def __init__(self, cfg, predict_type):
        super(transformer, self).__init__(predict_type)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_in)

        encoder_layer = nn.TransformerEncoderLayer(cfg.model_in, cfg.enc_n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(cfg.model_in, cfg.dec_n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.dec_layers)
        self.tail_predictor = nn.Linear(cfg.model_in, cfg.vocab_size)

    def forward(self, x, rel=None):
        enc_input, dec_input, relation = x['word'], x['tail_cont'], x['relation']
        if rel is None:
            enc_input = torch.cat((torch.unsqueeze(relation, dim=1), enc_input), dim=1)
        else:
            enc_input = torch.cat((torch.unsqueeze(rel, dim=1), enc_input), dim=1)
        src_key_padding_mask = enc_input.data.eq(0)
        tgt_key_padding_mask = dec_input.data.eq(0)

        # embedding
        enc_embed = self.embedding(enc_input)
        dec_embed = self.embedding(dec_input)

        # encoding
        enc_output = self.encoder(enc_embed.transpose(0, 1), src_key_padding_mask = src_key_padding_mask)

        # decoding
        dec_output = self.decoder(dec_embed.transpose(0, 1), enc_output, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

        # predicting
        tail = self.tail_predictor(dec_output.transpose(0, 1))
        return tail