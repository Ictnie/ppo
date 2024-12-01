import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def bc_learn(current_time, file, num=200):
    """
    current_time
        确保数据时间
    file
        训练使用的数据
    num
        训练轮次
    """
    print('bc start')
    ##数据划分
    np.random.shuffle(file)
    train_data = file[0:70000]
    test_data = file[70000:]
    x_train = []
    y_train = []
    for i in train_data:
        p = [i[0], i[1], i[2]]
        x_train.append(p)
        y_train.append([i[3]])
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    x_test = []
    y_test = []
    for i in test_data:
        p = [i[0], i[1], i[2]]
        x_test.append(p)
        y_test.append([i[3]])
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset)

    ##BC网络
    class BehaviorCloningNN(nn.Module):
        def __init__(self):
            super(BehaviorCloningNN, self).__init__()
            self.fc1 = nn.Linear(3, 9)  # 输入3维，隐藏层9个节点
            self.fc2 = nn.Linear(9, 1)  # 输出1维
            self.output = nn.Tanh()

        def forward(self, x):
            x = torch.relu(self.fc1(x))  # 激活函数
            x = self.fc2(x)
            x = self.output(x) * 2
            return x

    model = BehaviorCloningNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    model.train()
    res = []
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
                total_loss = 0
                for inputs, targets in test_dataloader:
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                res.append(total_loss)

    file_model = 'bc_model.pth'
    torch.save(model.state_dict(), f'{current_time}/{file_model}')

    model.load_state_dict(torch.load(f'{current_time}/{file_model}'))
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(test_dataloader)
    print(f'Average MSE on test set: {average_loss:.4f}')

    x = [i for i in range(len(res))]
    plt.figure()
    plt.plot(x, res)
    plt.savefig(f"{current_time}/bc_train_loss.jpg")
    res = np.array(res)
    np.save(f'{current_time}/bc_train_loss.npy', res)
