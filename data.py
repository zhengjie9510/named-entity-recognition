import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, datas, tags, word_2_index, tag_2_index, device):
        self.datas = datas
        self.tags = tags
        self.word_2_index: dict = word_2_index
        self.tag_2_index: dict = tag_2_index
        self.device = device

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word_2_index.get(
            i, self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]

        data_index = torch.tensor(
            data=data_index, dtype=torch.long, device=self.device)
        tag_index = torch.tensor(
            data=tag_index, dtype=torch.long, device=self.device)

        return data_index, tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.datas)

    def batch_data_pro(self, batch_datas):
        data, tag, mask, da_len = [], [], [], []
        for da, ta in batch_datas:
            l = len(da)
            data.append(da)
            tag.append(ta)
            da_len.append(l)
            mask.append(torch.tensor(
                [1] * l, dtype=torch.bool, device=self.device))

        data = nn.utils.rnn.pad_sequence(
            data, batch_first=True, padding_value=self.word_2_index["<PAD>"])
        tag = nn.utils.rnn.pad_sequence(
            tag, batch_first=True, padding_value=self.tag_2_index["<PAD>"])
        mask = nn.utils.rnn.pad_sequence(
            mask, batch_first=True)
        return data, tag, da_len, mask


class DataProcess():
    def __init__(self, filepath) -> None:
        self.filepath = filepath

    def get_samples(self):
        word_lists = []
        label_lists = []
        data_name = os.path.basename(os.path.dirname(self.filepath))
        if data_name == 'resume':
            with open(self.filepath, 'r', encoding='utf-8') as f:
                word_list = []
                label_list = []
                for line in f:
                    if line != '\n':
                        word, label = line.strip('\n').split()
                        word_list.append(word)
                        label_list.append(label)
                    else:
                        word_lists.append(word_list)
                        label_lists.append(label_list)
                        word_list = []
                        label_list = []
        if data_name == 'cluener':
            with open(self.filepath, 'r', encoding='utf-8') as f:
                # 先读取到内存中，然后逐行处理
                for line in f.readlines():
                    # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                    json_line = json.loads(line.strip())

                    text = json_line['text']
                    words = list(text)
                    # 如果没有label，则返回None
                    label_entities = json_line.get('label', None)
                    labels = ['O'] * len(words)

                    if label_entities is not None:
                        for key, value in label_entities.items():
                            for sub_name, sub_index in value.items():
                                for start_index, end_index in sub_index:
                                    assert ''.join(
                                        words[start_index:end_index + 1]) == sub_name
                                    if start_index == end_index:
                                        labels[start_index] = 'S-' + key
                                    else:
                                        labels[start_index] = 'B-' + key
                                        labels[start_index + 1:end_index +
                                               1] = ['I-' + key] * (len(sub_name) - 1)
                    word_lists.append(words)
                    label_lists.append(labels)

        word_lists = self.sort_by_length(word_lists)
        label_lists = self.sort_by_length(label_lists)
        return word_lists, label_lists

    def sort_by_length(self, lists):
        lists = sorted(lists, key=lambda x: len(x), reverse=True)
        return lists

    def build_map(self, lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps
