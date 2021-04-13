import os
import torch
import torch.nn as nn

from transformers import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .BasicModule import BasicModule


class bert(BasicModule):
    def __init__(self, cfg, predict_type):
        super(bert, self).__init__(predict_type)
        pretrained_model = os.path.join(cfg.cwd, cfg.pretrained_path, "pytorch_model.bin")
        model_config = BertConfig.from_json_file(os.path.join(cfg.cwd, cfg.pretrained_path, "config.json"))
        self.bert = BertModel.from_pretrained(pretrained_model, config=model_config)
        self.bilstm = nn.LSTM(cfg.lstm_in, cfg.lstm_hid, cfg.lstm_layers, batch_first=True, dropout=cfg.lstm_drop, bidirectional=cfg.bidirectional)

        self.dropout_label = nn.Dropout(cfg.lstm_drop)
        self.fc_rel = nn.Linear(cfg.lstm_hid * cfg.lstm_layers * (1 + cfg.bidirectional), cfg.num_relations)
    
    def forward(self, x):
        word, lens = x['word'], x['lens']
        b, _= word.size()
        last_hidden_state = self.bert(word).last_hidden_state
        last_hidden_state = pack_padded_sequence(last_hidden_state, lens.cpu(), batch_first=True, enforce_sorted=True)
        out, (out_pool, _) = self.bilstm(last_hidden_state)
        out_pool = self.dropout_label(out_pool.transpose(0, 1).reshape(b, -1))
        output = self.fc_rel(out_pool)
        return output