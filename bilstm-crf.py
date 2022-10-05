import os
import torch
import torch.nn as nn
from torchcrf import CRF
from itertools import chain
from sklearn.metrics import f1_score, accuracy_score, precision_score
from torch.utils.data import Dataset, DataLoader


def build_corpus(split, make_vocab=True, data_dir="data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=True)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=True)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)

        tag2id['<PAD>'] = len(tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps


class MyDataset(Dataset):
    def __init__(self, datas, tags, word_2_index, tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index: dict = word_2_index
        self.tag_2_index: dict = tag_2_index

    def __getitem__(self, index):
        global device
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word_2_index.get(
            i, self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]

        data_index = torch.tensor(
            data=data_index, dtype=torch.long, device=device)
        tag_index = torch.tensor(
            data=tag_index, dtype=torch.long, device=device)

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
            mask.append(torch.tensor([1] * l, dtype=torch.bool, device=device))

        data = nn.utils.rnn.pad_sequence(
            data, batch_first=True, padding_value=self.word_2_index["<PAD>"])
        tag = nn.utils.rnn.pad_sequence(
            tag, batch_first=True, padding_value=self.tag_2_index["<PAD>"])
        mask = nn.utils.rnn.pad_sequence(
            mask, batch_first=True)
        return data, tag, da_len, mask


class BiLSTMCRF(nn.Module):
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
        self.crf = CRF(class_num, batch_first=True)

    def forward(self, data_index, data_len):
        em = self.embedding(data_index)
        pack = nn.utils.rnn.pack_padded_sequence(
            em, data_len, batch_first=True)
        output, _ = self.lstm(pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        pre = self.classifier(output)

        return pre

    def loss(self, emissions, tags, mask):
        loss = self.crf(emissions, tags, mask)
        return -loss

    def decode(self, emissions, mask=None):
        out = self.crf.decode(emissions, mask)
        return out

    def fit(self, train_dataloader, epoch, dev_dataloader):
        """
        训练模型
        """
        lr = 0.001
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epoch):
            print("Epoch", f"{e+1}/{epoch}")
            self.train()
            for data, tag, da_len, mask in train_dataloader:
                pred = self.forward(data, da_len)
                loss = self.loss(pred, tag, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'loss:{round(loss.item(), 2)}', end='\r')
            self.eval()
            for data, tag, da_len, mask in dev_dataloader:
                tag = nn.utils.rnn.unpad_sequence(
                    tag, da_len, batch_first=True)
                tag = [list(x.cpu().numpy()) for x in tag]
                pred = self.forward(data, da_len)
                pred = self.decode(pred, mask)

                y_pred = list(chain.from_iterable(pred))
                y_true = list(chain.from_iterable(tag))
                f1 = f1_score(y_true, y_pred, average="micro")
            print(f"loss:{round(loss.item(),2)}\tf1:{round(f1,3)}")

    def predict(self,  word_2_index, index_2_tag, filepath):
        self.load_state_dict(torch.load(filepath))
        text = input("请输入：")
        text_index = [
            [word_2_index.get(i, word_2_index["<UNK>"]) for i in text]]
        text_index = torch.tensor(text_index, device=device)
        text_len = [len(text)]
        index_2_tag = {i: c for i, c in enumerate(tag_2_index)}
        pred = self.forward(text_index, text_len)
        pred = self.decode(pred)
        pred = [index_2_tag[i] for i in pred[0]]
        print([f'{w}_{s}' for w, s in zip(text, pred)])


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    type = "predict"
    model_path = "models/bilstm-crf.pth"

    # 准备数据
    train_word_lists, train_tag_lists, word_2_index, tag_2_index = build_corpus(
        "train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    # 设置模型参数
    num_embeddings = len(word_2_index)
    embedding_dim = 128
    hidden_size = 129
    bidirectional = True
    class_num = len(tag_2_index)
    # 建立模型
    model = BiLSTMCRF(num_embeddings, embedding_dim,
                      hidden_size, bidirectional, class_num)
    model = model.to(device)
    # 设置数据集参数
    train_batch_size = 64
    dev_batch_size = len(dev_word_lists)
    # 构建数据集
    train_dataset = MyDataset(
        train_word_lists, train_tag_lists, word_2_index, tag_2_index)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=False, collate_fn=train_dataset.batch_data_pro)
    dev_dataset = MyDataset(dev_word_lists, dev_tag_lists,
                            word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size,
                                shuffle=False, collate_fn=dev_dataset.batch_data_pro)
    # 构建index转tag字典
    index_2_tag = {i: c for i, c in enumerate(tag_2_index)}

    if type == 'fit':
        # 模型训练
        model.fit(train_dataloader, 20, dev_dataloader)
        # 保存模型
        torch.save(model.state_dict(), model_path)
    # 预测
    else:
        while True:
            model.predict(word_2_index, index_2_tag, model_path)
