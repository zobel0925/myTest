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

def f_out(x, w_out, b_out):
    return x.mm(w_out.t()) + b_out

def get_batch(batch_size=32):
    """ Build a batch i.e. (x, f(x)) pari. """
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda(), random
    else:
        return Variable(x), Variable(y), random

def get_line_batch(model):
    x_data = torch.arange(-3, 3, step=1e-2)
    x = make_features(x_data)
    y_data = f_out(x, model.poly.weight.data, model.poly.bias.data)
    return x_data, y_data

def get_origin_batch():
    x_data = torch.arange(-3, 3, step=1e-2)
    x = make_features(x_data)
    y = f(x)
    return x_data, y
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
    model = poly_model()
    
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
# get data
batch_x, batch_y, x_data = get_batch()
#x_data = batch_x.data.cpu()
#plt.plot(x_data.numpy(), y_data.numpy(), 'ro', label='Original data1')
while True:
    #batch_x, batch_y, x_data = get_batch()
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
    if print_loss < 1e-1:
        break

y_data = batch_y.data.cpu()
xx_data, yy_data = get_line_batch(model)
xx_origin, yy_origin = get_origin_batch()
plt.plot(xx_origin.numpy(), yy_origin.numpy(), color="red", linewidth=1.5, label='Original data')
plt.plot(xx_data.numpy(), yy_data.numpy(), color="blue", linewidth=1, label='Original data')
plt.plot(x_data.numpy(), output.data.cpu().numpy(), 'r*', label='Original data')
plt.show()
print("epoch is %d\n" %(epoch))
print(model.poly.weight.data)
print(model.poly.bias.data)