import torch

a = torch.tensor(1.0)
b = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#forward pass to compute the loss 

y_hat = w * a 
loss = (y_hat - b) ** 2

print(loss)

#backward pass to compute local gradients and chain rule to compute the gradient of the loss with respect to w
loss.backward()
print(w.grad)