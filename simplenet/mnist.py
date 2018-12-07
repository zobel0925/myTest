import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import mnist
import numpy as np

import net

# 超参数 (Hyperparameters)
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

# 预处理, Compose将参数的模型进行组合
# ToTensor, 将图像转换为Tensor, 范围0~1
# Normalize, 标准化(均值, 方差), 表示 (value-均值)/方差
#data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

# 下载训练集,MNIST手写数字训练集
train_dataset = mnist.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = mnist.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = net.Batch_Net(28*28, 300, 100, 10)

if torch.cuda.is_available():
    model = model.cuda()
a, a_label = next(iter(train_loader))
a2, a2_label = next(iter(train_loader))

print(a.shape)
print(a_label.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []
acces = []
eval_losses = []
eval_acces = []
epoch = 0

for e in range(2):
    train_loss = 0
    train_acc = 0
    model.train()
    im_num = 0
    for im, label in train_loader:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = model(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct / im.shape[0]
        train_acc += acc
        im_num += 1
    
    print('im_num: {}'.format(im_num))
    losses.append(float(train_loss) / len(train_loader))
    acces.append(float(train_acc) / len(train_loader))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    model.eval() # 将模型改为预测模式
    for im, label in test_loader:
        im = Variable(im)
        label = Variable(label)
        out = model(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.data
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(float(eval_loss) / len(test_loader))
    eval_acces.append(float(eval_acc) / len(test_loader))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_loader), train_acc , 
                     eval_loss / len(test_loader), eval_acc))
print('train_loader len: {}, test_loader len: {}'
          .format(len(train_loader), len(test_loader)))
import matplotlib.pyplot as plt

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.show()

plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.show()

plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.show()

plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()