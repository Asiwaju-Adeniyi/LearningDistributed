import torch 
import numpy as np

x = torch.rand(2,2)
y = torch.rand(2,2)
z = torch.sub(x, y)
print(z)
y.mul_(x)



a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

if torch.mps.is_available(): 
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    y = torch.ones(5, device=device)
    q = torch.zeros(5)
    q = q.to(device)
    z = x + y
    print(z)