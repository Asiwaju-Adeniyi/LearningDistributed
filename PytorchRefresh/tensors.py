import torch 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
print(torch.__version__)

# x = torch.rand(2,2)
# y = torch.rand(2,2)
# z = torch.sub(x, y)
# print(z)
# y.mul_(x)



# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# #print(b)

# if torch.mps.is_available(): 
#     device = torch.device("mps")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5, device=device)
#     q = torch.zeros(5)
#     q = q.to(device)
#     z = x + y
#     print(z)

#     scalar = torch.tensor(7)
#     #print(scalar.item())

#     vector = torch.tensor([7,7])



# tensor = torch.tensor([1,2,3])
# print(tensor + 10)


# tensor = torch.tensor([3,4,5,0,98])
# print(tensor * 100)


###MatMuls: Two main ways of performing multplication in neural networks and deep learning. 

# tensor = torch.tensor([4,4,4,0])
# print(tensor, "*", tensor)
# print(f"Equals: {tensor * tensor}")

# print(torch.matmul(tensor, tensor))


# tensor1 = torch.tensor([[1,2,3,5,6,4,5],
#                          [45,4,50,9,89,3,4], 
#                          [45,4,50,9,89,3,4]])
# tensor2 = torch.tensor([[1,2,3,5,6,4,5],
#                          [45,4,50,9,89,3,4], 
#                          [45,4,50,9,89,3,4]])

# tensor = tensor1 @ tensor2.permute(1,0)
# print(tensor)


# Mean, Max, Min, Sum, etc. (Tensor Aggregation)

x = torch.arange(0, 100, 10)
y = torch.min(x)
print(y)

print(torch.mean(x.type(torch.float32)))