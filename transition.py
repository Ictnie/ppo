import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def trans_learn(current_time,file,num=200):

    print("trans start")
    np.random.shuffle(file)
    train_data = file[0:70000]
    test_data = file[70000:]

    x_train = []
    y_train = []
    for i in train_data:
        p = [i[0], i[1], i[2],i[3]]
        x_train.append(p)
        y_train.append([i[5],i[6],i[7]])
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    x_test = []
    y_test = []
    for i in test_data:
        p = [i[0], i[1], i[2], i[3]]
        x_test.append(p)
        y_test.append([i[5], i[6], i[7]])
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset)

    ## LSTM 网络定义
    class CustomLSTMModel(nn.Module):
        def __init__(self, input_size=4, hidden_size1=16, hidden_size2=256, output_size=3):
            super(CustomLSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, num_layers=1, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = self.fc(x)
            return x

    model = CustomLSTMModel(input_size=4, output_size=3)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    res=[]
    for epoch in range(num):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % (num//10) == 0:
            print(f'Epoch [{epoch + 1}/{num}], Loss: {loss.item():.8f}')
            with torch.no_grad():
                total_loss=0
                for inputs, targets in test_dataloader:
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                res.append(total_loss)

    file_model='transition.pth'

    torch.save(model.state_dict(), f'{current_time}/{file_model}')

    model.load_state_dict(torch.load(f'{current_time}/{file_model}'))

    # 预测
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(test_dataloader)
    print(f'Average MSE on test set: {average_loss:.10f}')

    x = [i for i in range(len(res))]
    plt.figure()
    plt.plot(x, res)
    plt.savefig(f"{current_time}/trans_train_loss.jpg")
    res = np.array(res)
    np.save(f'{current_time}/trans_train_loss.npy', res)
