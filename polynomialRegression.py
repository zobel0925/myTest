import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def make_features(x):
    """builds featrues i.e. a matrix with columns [x, x^2, x^3]. """
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    """ Approximated function. """
    return x.mm(w_target) + b_target[0]

def get_batch(batch_size=32):
    """ Build a batch i.e. (x, f(x)) pari. """
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda(), random
    else:
        print("no cuda here!!!!")
        return Variable(x), Variable(y)
# define model
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    print("no cuda here!!!!")
    
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
# get data
batch_x, batch_y, x_data = get_batch()
#x_data = batch_x.data.cpu()
y_data = batch_y.data.cpu()
#plt.plot(x_data.numpy(), y_data.numpy(), 'ro', label='Original data1')
while True:
    #batch_x, batch_y = get_batch()
    # forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data
    #reset gradients
    optimizer.zero_grad()
    #backward pass
    loss.backward()
    #update parameters
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

plt.plot(x_data.numpy(), output.data.cpu().numpy(), 'b*', label='Original data')
#plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()
print("epoch is %d" %(epoch))