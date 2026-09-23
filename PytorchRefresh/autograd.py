import torch 

a = torch.randn(3, requires_grad=True)
print(a)

b = a + 2
print(b)
c = b * b * 3
#c = c.mean()
print(c)

q = torch.tensor([0.3, 0.7, 0.5], dtype = torch.float16)
c.backward(q) 

