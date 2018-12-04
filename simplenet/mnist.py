import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import net

# 超参数 (Hyperparameters)
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

# 预处理, Compose将参数的模型进行组合
# ToTensor, 将图像转换为Tensor, 范围0~1
# Normalize, 标准化(均值, 方差), 表示 (value-均值)/方差
data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# 下载训练集,MNIST手写数字训练集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = net.simpleNet(28*28, 300, 100, 10)

if torch.cuda.is_available():
    model = model.cuda()
a, a_label = next(iter(train_loader))

print(a.shape)
print(a_label.shape)

exit
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []
acces = []
eval_loss = 0
eval_acc = 0
epoch = 0

while epoch < num_epoches:
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('*'*10)
    print('epoch {}'.format(epoch))
    print('loss is {:.4f}'.format(eval_loss))
    print('acc is {:.4f}'.format(eval_acc))
    epoch += 1
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))

print("end!")