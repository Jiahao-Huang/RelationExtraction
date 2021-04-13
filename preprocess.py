import os
import logging
from collections import OrderedDict
from typing import List, Dict
from transformers import BertTokenizer
from utils import save_pkl, load_csv
import hydra

logger = logging.getLogger(__name__)


def _handle_relation_data(relation_data: List[Dict]) -> Dict:
    rels = OrderedDict()
    relation_data = sorted(relation_data, key=lambda i: int(i['index']))
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }

    return rels


def _add_relation_data(rels: Dict, data: List) -> None:
    for d in data:
        d['rel2idx'] = rels[d['relation']]['index']
        d['head_type'] = rels[d['relation']]['head_type']
        d['tail_type'] = rels[d['relation']]['tail_type']

def _lm_serialize(data: List[Dict], cfg):
    logger.info('use bert tokenizer...')
    vocab = os.path.join(cfg.cwd, cfg.tokenizer_path, 'vocab.txt')
    special_tokens_dict = {'cls_token': '<CLS>', 'bos_token': '<s>', 'unk_token': '<UNK>'}
    tokenizer = BertTokenizer.from_pretrained(vocab, add_special_tokens=special_tokens_dict)
    cfg.vocab_size = tokenizer.vocab_size
    for d in data:
        sent = d['sentence'].strip()
        sent += '[SEP]' + d['head']
        
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)
        tail_token2idx = tokenizer.encode(d['tail'], add_special_tokens=False)
        d['tail_cont'] = tokenizer.encode(d['tail'], add_special_tokens=True)

        d['seq_len'] = len(d['token2idx'])
        d['tail_len'] = len(d['tail_cont'])

@hydra.main(config_path='config.yaml')
def preprocess(cfg):

    logger.info('===== start preprocess data =====')
    train_fp = os.path.join(cfg.cwd, cfg.data_path, 'train.csv')
    valid_fp = os.path.join(cfg.cwd, cfg.data_path, 'valid.csv')
    test_fp = os.path.join(cfg.cwd, cfg.data_path, 'test.csv')
    relation_fp = os.path.join(cfg.cwd, cfg.data_path, 'relation.csv')

    logger.info('load raw files...')
    train_data = load_csv(train_fp)
    valid_data = load_csv(valid_fp)
    test_data = load_csv(test_fp)
    relation_data = load_csv(relation_fp)

    logger.info('convert relation into index...')
    rels = _handle_relation_data(relation_data)
    _add_relation_data(rels, train_data)
    _add_relation_data(rels, valid_data)
    _add_relation_data(rels, test_data)

    logger.info('use pretrained language models serialize sentence...')
    _lm_serialize(train_data, cfg)
    _lm_serialize(valid_data, cfg)
    _lm_serialize(test_data, cfg)

    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)
    logger.info('===== end preprocess data =====')


if __name__ == "__main__":
    preprocess()