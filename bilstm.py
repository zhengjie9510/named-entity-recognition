import json
import torch
import numpy as np
import torch.nn as nn
from itertools import chain
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader


class Tokenizer():
    def __init__(self, word_lists, label_lists, pad='<PAD>', unknown='<UNK>') -> None:
        self.word_lists = word_lists
        self.label_lists = label_lists
        self.pad = pad
        self.unknown = unknown
        self.vocab = self._build_map(self.word_lists, unknown=True)
        self.label_vocab = self._build_map(self.label_lists, unknown=False)
        self.decode_vocab = {value: key for key, value in self.vocab.items()}
        self.decode_label_vocab = {value: key for key,
                                   value in self.label_vocab.items()}

    def encode(self, data):
        data_index = [self.vocab.get(i, self.vocab["<UNK>"]) for i in data]
        return data_index

    def decode(self, data):
        words = [self.decode_vocab[i] for i in data]
        return words

    def encode_label(self, data):
        data_index = [self.label_vocab.get(i) for i in data]
        return data_index

    def decode_label(self, data):
        labels = [self.decode_label_vocab[i] for i in data]
        return labels

    def _build_map(self, lists, unknown=False):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        maps[self.pad] = len(maps)
        if unknown:
            maps[self.unknown] = len(maps)
        return maps


class DataProcess():
    """
    提供数据处理功能
    """

    def __init__(self, filepath) -> None:
        """
        数据处理
        filepath: 原始文件路径
        """
        self.filepath = filepath

    def _get_samples(self):
        """
        获取word以及对应的label
        """
        word_lists = []
        label_lists = []
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

        return word_lists, label_lists

    def _sort_by_length(self, word_lists, label_lists):
        """
        对数据进行排序
        """
        word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
        label_lists = sorted(label_lists, key=lambda x: len(x), reverse=True)
        return word_lists, label_lists

    def get_processed_samples(self, sort=True):
        word_lists, label_lists = self._get_samples()
        if sort:
            word_lists, label_lists = self._sort_by_length(
                word_lists, label_lists)
        return word_lists, label_lists


class MyDataset(Dataset):
    def __init__(self, word_lists, label_lists, tokenizer, device):
        self.word_lists = word_lists
        self.label_lists = label_lists
        self.tokenizer: Tokenizer = tokenizer
        self.device = device

    def __getitem__(self, index):
        words = self.word_lists[index]
        labels = self.label_lists[index]

        words_index = self.tokenizer.encode(words)
        labels_index = self.tokenizer.encode_label(labels)
        words_index = torch.tensor(
            words_index, dtype=torch.long, device=self.device)
        labels_index = torch.tensor(
            labels_index, dtype=torch.long, device=self.device)

        l = len(words_index)
        mask = torch.tensor([1] * l, dtype=torch.bool, device=self.device)

        return words_index, labels_index, mask, l

    def __len__(self):
        assert len(self.word_lists) == len(self.label_lists)
        return len(self.label_lists)

    def batch_data_pro(self, batch_datas):
        word_lists, label_lists, mask_lists, len_list = [], [], [], []
        for words, labels, mask, l in batch_datas:
            word_lists.append(words)
            label_lists.append(labels)
            mask_lists.append(mask)
            len_list.append(l)

        word_lists = nn.utils.rnn.pad_sequence(
            word_lists, batch_first=True, padding_value=self.tokenizer.vocab[self.tokenizer.pad])
        label_lists = nn.utils.rnn.pad_sequence(
            label_lists, batch_first=True, padding_value=self.tokenizer.label_vocab[self.tokenizer.pad])
        mask_lists = nn.utils.rnn.pad_sequence(mask_lists, batch_first=True)
        return word_lists, label_lists, mask_lists, len_list


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

    def forward(self, X, X_len):
        em = self.embedding(X)
        pack = nn.utils.rnn.pack_padded_sequence(
            em, X_len, batch_first=True)
        output, _ = self.lstm(pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        pre = self.classifier(output)
        return pre

    def _compute_matrix(self, X, y, mask, X_len):
        pred = self.forward(X,  X_len)
        pred = nn.utils.rnn.unpad_sequence(
            torch.argmax(pred, dim=-1), X_len, batch_first=True)

        y = nn.utils.rnn.unpad_sequence(
            y,  X_len, batch_first=True)
        y = [list(i.cpu().numpy()) for i in y]

        y_pred = list(chain.from_iterable(pred))
        y_true = list(chain.from_iterable(y))
        f1 = f1_score(y_true, y_pred, average="micro")

        return f1

    def fit(self, train_dataloader, epoch, tokenizer: Tokenizer, dev_dataloader=None):
        """
        训练模型
        """
        lr = 0.001
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.label_vocab[tokenizer.pad])
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epoch):
            print("Epoch", f"{e+1}/{epoch}")
            train_loss, train_f1 = [], []
            train_num = 0
            self.train()
            for X, y, mask, X_len in train_dataloader:
                pred = self.forward(X, X_len)
                loss = loss_fn(pred.transpose(1, 2), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.cpu().detach().numpy().sum())
                f1 = self._compute_matrix(X, y, mask, X_len)
                train_f1.append(f1)
                train_num = train_num + X.__len__()
                print('train_loss: %.4f' % (np.sum(train_loss)/train_num),
                      '\ttrain_f1: %.4f' % (np.mean(train_f1)), end='\r')
            if dev_dataloader:
                dev_loss, dev_f1 = [], []
                dev_num = 0
                self.eval()
                for X, y, mask, X_len in dev_dataloader:
                    pred = self.forward(X, X_len)
                    loss = loss_fn(pred.transpose(1, 2), y)
                    dev_loss.append(loss.cpu().detach().numpy().sum())
                    f1 = self._compute_matrix(X, y, mask, X_len)
                    dev_f1.append(f1)
                    dev_num = dev_num + X.__len__()
                print('train_loss: %.4f' % (np.sum(train_loss)/train_num), '\ttrain_f1: %.4f' % (np.mean(train_f1)),
                      '\tdev_loss: %.4f' % (np.sum(dev_loss)/dev_num), '\tdev_f1: %.4f' % (np.mean(dev_f1)))

    def predict(self, tokenizer: Tokenizer, text, device):
        text_index = [tokenizer.encode(text)]
        text_index = torch.tensor(text_index, device=device)
        text_len = [len(text)]
        pred = self.forward(text_index, text_len)
        pred = torch.argmax(pred, dim=-1)
        pred = tokenizer.decode_label(pred.cpu().numpy()[0])
        print([f'{w}_{s}' for w, s in zip(text, pred)])


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("使用设备：", device)
    # ///////////////////
    process = DataProcess(filepath='cluener/train.json')
    train_word_lists, train_label_lists = process.get_processed_samples(
        sort=True)
    process = DataProcess(filepath='cluener/dev.json')
    dev_word_lists, dev_label_lists = process.get_processed_samples(
        sort=True)
    tokenizer = Tokenizer(train_word_lists, train_label_lists)
    # ///////////////////
    num_embeddings = len(tokenizer.vocab)
    class_num = len(tokenizer.label_vocab)
    embedding_dim = 128
    hidden_size = 129
    bidirectional = True
    train_batch_size = 64
    epoch = 10
    model_path = "models/bilstm.pth"
    # ///////////////////
    train_dataset = MyDataset(
        train_word_lists, train_label_lists, tokenizer, device)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=False, collate_fn=train_dataset.batch_data_pro)
    dev_dataset = MyDataset(
        dev_word_lists, dev_label_lists, tokenizer, device)
    dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_word_lists),
                                shuffle=False, collate_fn=dev_dataset.batch_data_pro)
    # ///////////////////
    model = BiLSTM(num_embeddings, embedding_dim,
                   hidden_size, bidirectional, class_num)
    model = model.to(device)
    # ///////////////////
    # model.fit(train_dataloader, epoch, tokenizer,
    #           dev_dataloader=dev_dataloader)
    # torch.save(model.state_dict(), model_path)
    # ///////////////////
    model.load_state_dict(torch.load(model_path, map_location=device))
    while True:
        text = input("请输入：")
        model.predict(tokenizer, text, device)
