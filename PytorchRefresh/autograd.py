import torch 

#a = torch.randn(3, requires_grad=True)
#print(a)

#b = a + 2
#print(b)
#c = b * b * 3
#c = c.mean()
#print(c)
#q = torch.tensor([0.3, 0.7, 0.5], dtype = torch.float16)
#c.backward(q) 
#a.requires_grad_(False)
#print(a)
#y = a.detach() 

#with torch.no_grad(): 
    #b = a + 2
    #print(b)

weights = torch.ones(4, requires_grad=True) 

for epoch in range(5): 
    model_output = (weights * 3).sum() 

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()



