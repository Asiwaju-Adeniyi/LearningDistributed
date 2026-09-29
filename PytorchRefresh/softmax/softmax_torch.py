import torch

x = torch.tensor([2.0, 1.0, 3.0, 4.0])
outputs = torch.softmax(x, dim = 0)

print(outputs)