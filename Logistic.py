import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time


with open('data.txt', 'r') as f:
    data_list = f.readlines()
    data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
    x_in = [(float(i[0]), float(i[1])) for i in data_list]
    y_in = [(float(i[2])) for i in data_list]

x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))
plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]
plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]

plt.plot(plot_x0_0, plot_x0_1, 'ro', label='x_0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label='x_1')
plt.legend(loc='best')

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2,1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x

logistic_model = LogisticRegression()
if torch.cuda.is_available():
    logistic_model.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)


x_np = np.array(x_in)
y_np = np.array(y_in)
print("start!!!")
x_data = torch.from_numpy(x_np)
y_data = torch.from_numpy(y_np)
for epoch in range(1000):
    if torch.cuda.is_available():
        x = Variable(x_data).float().cuda()
        y = Variable(y_data).float().cuda()
    else:
        x = Variable(x_data).float()
        y = Variable(y_data).float()

    ### forward
    out = logistic_model(x)
    loss = criterion(out, y)
    print_loss = loss.data
    mask = out.ge(0.5).float()
    correct = (mask == y).sum()
    acc = correct.data / x.size(0)

    ### backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch+1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

w0, w1 = logistic_model.lr.weight[0]
w0 = w0.data[0].float().numpy()
w1 = w1.data[0].float().numpy()
b = logistic_model.lr.bias.data[0].float().numpy()
plot_x = np.arange(30, 100, 0.1)
plot_y = (-w0*plot_x - b) / w1
plt.plot(plot_x, plot_y)
plt.show()