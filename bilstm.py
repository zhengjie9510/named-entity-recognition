import torch
import torch.nn as nn
from itertools import chain
from sklearn.metrics import f1_score
from data import DataProcess, MyDataset
from torch.utils.data import DataLoader


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
    filepath = 'data/resume/train.char.bmes'
    process = DataProcess(filepath=filepath)
    train_word_lists, train_label_lists = process.get_samples()
    word_2_index = process.build_map(train_word_lists)
    label_2_index = process.build_map(train_label_lists)
    word_2_index['<UNK>'] = len(word_2_index)
    word_2_index['<PAD>'] = len(word_2_index)
    label_2_index['<PAD>'] = len(label_2_index)
    # 验证数据
    filepath = 'data/resume/test.char.bmes'
    process = DataProcess(filepath=filepath)
    dev_word_lists, dev_label_lists = process.get_samples()
    dev_word_lists = process.sort_by_length(dev_word_lists)
    dev_label_lists = process.sort_by_length(dev_label_lists)
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
