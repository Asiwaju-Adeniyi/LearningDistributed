import torch 
import numpy as np

x = torch.rand(2,2)
y = torch.rand(2,2)


z = torch.sub(x, y)
print(z)
y.mul_(x)



a = np.ones(5)
print(a)
b = a.numpy()
print(b)