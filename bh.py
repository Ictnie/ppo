import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

file = np.load('expert.npy')
np.random.shuffle(file)
train_data = file[0:70000]
test_data = file[70000:]
x_train = []
y_train = []
for i in train_data:
    p = [i[0], i[1], i[2]]
    x_train.append(p)
    y_train.append([i[4]])
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class BehaviorCloningNN(nn.Module):
    def __init__(self):
        super(BehaviorCloningNN, self).__init__()
        self.fc = nn.Linear(3, 9)  # 输入3维，隐藏层9个节点
        self.output = nn.Linear(9, 1)  # 输出1维

    def forward(self, x):
        x = torch.relu(self.fc(x))  # 激活函数
        x = self.output(x)
        return x


model = BehaviorCloningNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1000):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.6f}')

torch.save(model.state_dict(), 'behavior_cloning_model.pth')

x_test = []
y_test = []
for i in test_data:
    p = [i[0], i[1], i[2]]
    x_test.append(p)
    y_test.append([i[4]])
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset)

model.load_state_dict(torch.load('behavior_cloning_model.pth'))
model.eval()

total_loss = 0
print(len(test_dataloader))
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
average_loss = total_loss / len(test_dataloader)
print(f'Average MSE on test set: {average_loss:.4f}')
