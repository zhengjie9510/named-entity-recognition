import os
import json
import torch
import torch.nn as nn
from itertools import chain
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, datas, tags, word_2_index, tag_2_index, device):
        self.device = device
        self.datas = datas
        self.tags = tags
        self.word_2_index: dict = word_2_index
        self.tag_2_index: dict = tag_2_index

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

    def build_maps(self, lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps


class BiLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, bidirectional, class_num):
        """
        num_embeddings: size of the dictionary of embeddings  
        embedding_dim: the size of each embedding vector
        hidden_size: The number of features in the hidden state `h`
        bidirectional: If ``True``, becomes a bidirectional LSTM
        class_num: class number
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size,
                            batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            self.classifier = nn.Linear(hidden_size * 2, class_num)
        else:
            self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, data_index, data_len):
        em = self.embedding(data_index)
        pack = nn.utils.rnn.pack_padded_sequence(
            em, data_len, batch_first=True)
        output, _ = self.lstm(pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        pre = self.classifier(output)

        return pre

    def fit(self,  train_dataloader, epoch, dev_dataloader, word_2_index):
        """
        训练模型
        """
        lr = 0.001
        loss_fn = nn.CrossEntropyLoss(ignore_index=word_2_index["<PAD>"])
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for e in range(epoch):
            print("Epoch", f"{e+1}/{epoch}")
            self.train()
            for data, tag, da_len, mask in train_dataloader:
                pred = self.forward(data, da_len)
                loss = loss_fn(pred.transpose(1, 2), tag)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'loss: {round(loss.item(), 2)}', end='\r')

            self.eval()
            for data, tag, da_len, mask in dev_dataloader:
                tag = nn.utils.rnn.unpad_sequence(
                    tag, da_len, batch_first=True)
                tag = [list(x.cpu().numpy()) for x in tag]
                pred = self.forward(data, da_len)
                pred = nn.utils.rnn.unpad_sequence(
                    torch.argmax(pred, dim=-1), da_len, batch_first=True)
                pred = [list(x.cpu().numpy()) for x in pred]

                y_pred = list(chain.from_iterable(pred))
                y_true = list(chain.from_iterable(tag))
                f1 = f1_score(y_true, y_pred, average="micro")
            print(f"loss: {round(loss.item(),2)}\tf1: {round(f1,3)}")

    def predict(self,  word_2_index, index_2_label, filepath):
        self.load_state_dict(torch.load(filepath))
        text = input("请输入：")
        text_index = [
            [word_2_index.get(i, word_2_index["<UNK>"]) for i in text]]
        text_index = torch.tensor(text_index, device=device)
        text_len = [len(text)]
        pred = self.forward(text_index, text_len)
        pred = torch.argmax(pred, dim=-1).reshape(-1).cpu().numpy()
        pred = [index_2_label[i] for i in pred]
        print([f'{w}_{s}' for w, s in zip(text, pred)])


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    mode = "predict"
    model_path = "models/bilstm.pth"
    # 准备数据
    # 训练数据
    filepath = 'data/cluener/train.json'
    process = DataProcess(filepath=filepath)
    train_word_lists, train_label_lists = process.get_samples()
    word_2_index = process.build_maps(train_word_lists)
    label_2_index = process.build_maps(train_label_lists)
    word_2_index['<UNK>'] = len(word_2_index)
    word_2_index['<PAD>'] = len(word_2_index)
    label_2_index['<PAD>'] = len(label_2_index)
    # 验证数据
    filepath = 'data/cluener/dev.json'
    process = DataProcess(filepath=filepath)
    dev_word_lists, dev_label_lists = process.get_samples()
    # 设置模型参数
    num_embeddings = len(word_2_index)
    embedding_dim = 128
    hidden_size = 129
    bidirectional = True
    class_num = len(label_2_index)
    # 建立模型
    model = BiLSTM(num_embeddings, embedding_dim,
                   hidden_size, bidirectional, class_num)
    model = model.to(device)
    # 设置数据集参数
    train_batch_size = 64
    dev_batch_size = len(dev_word_lists)
    # 构建数据集
    train_dataset = MyDataset(
        train_word_lists, train_label_lists, word_2_index, label_2_index, device)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=False, collate_fn=train_dataset.batch_data_pro)
    dev_dataset = MyDataset(dev_word_lists, dev_label_lists,
                            word_2_index, label_2_index, device)
    dev_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size,
                                shuffle=False, collate_fn=dev_dataset.batch_data_pro)
    # 构建index转label字典
    index_2_label = {i: c for i, c in enumerate(label_2_index)}

    if mode == 'fit':
        # 模型训练
        model.fit(train_dataloader, 20, dev_dataloader, word_2_index)
        # 保存模型
        torch.save(model.state_dict(), model_path)
    # 预测
    else:
        while True:
            model.predict(word_2_index, index_2_label, model_path)
