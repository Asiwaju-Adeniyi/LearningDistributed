import torch 

x = torch.rand(2,2)
y = torch.rand(2,2)


z = torch.sub(x, y)
print(z)
y.mul_(x)