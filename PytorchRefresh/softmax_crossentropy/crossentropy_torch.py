import torch 
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# nsamples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.4, 0.3, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_good, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions2)