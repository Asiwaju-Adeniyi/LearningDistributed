#1) Design model (input, output size, forward pass) 
#2) Construct loss and optimizer
#3) Training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

import torch
import torch.nn as nn 

# f = w * x
# f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)


in_features = n_features
out_features = n_features

#model = nn.Linear(in_features=1, out_features=1)

class LinearRegression(nn.Module): 
    def __init__(self, in_features, out_features): 
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x): 
        return self.lin(x)

model = LinearRegression(in_features=1, out_features=1)

print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

#Training
learning_rate = 0.01
n_iters = 10

#loss and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

for epoch in range(n_iters): 
    #prediction = forward pass 
    y_pred = model(X)

    #loss 
    l = loss(Y, y_pred) 

    #gradients = backward pass 
    l.backward() #dl/dw

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0: 
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, b = {b.item():.3f}, loss = {l:.8f}')

print(f'prediction after training: f(5) = {model(X_test).item():.3f}')