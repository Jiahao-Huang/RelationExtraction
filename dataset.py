import torch
from torch.utils.data import Dataset
from utils import load_pkl


def collate_fn(cfg):
    def collate_fn_intra(batch):
        batch.sort(key = lambda data: data['tail_len'], reverse=True)
        max_tail_len = batch[0]['tail_len']
        
        batch.sort(key=lambda data: data['seq_len'], reverse=True)
        max_len = batch[0]['seq_len']

        def _padding(x, max_len):
            return x + [0] * (max_len - len(x))

        x, relation = dict(), []
        word, word_len = [], []
        tail_pos, tail_cont = [], []
        pcnn_mask = []

        for data in batch:
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(data['seq_len'])
            relation.append(int(data['rel2idx']))
            
            tail_cont.append(_padding(data['tail_cont'], max_tail_len))
        
        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        x['tail_cont'] = torch.tensor(tail_cont)
        x['relation'] = torch.tensor(relation)
        return x

    return collate_fn_intra


class CustomDataset(Dataset):
    """默认使用 List 存储数据"""
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)


if __name__ == '__main__':
    pass
