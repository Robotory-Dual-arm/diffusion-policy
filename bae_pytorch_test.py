import torch

torch.set_printoptions(precision=4, sci_mode=False)

class Model():
    def __init__(self):
        super().__init__() 
        
    def forward(self, x):
        return x**2 

# random action     0 - 10
a = torch.rand(3,3) * 10
a = a.requires_grad_(True)
print('noise action', a)

y = torch.tensor([[1., 2., 3.], [4., 5., 6.], [0., 0., 0.]])

model = Model()
output = model.forward(a)

e = y - output



for i in range(100):
    vjp, = torch.autograd.grad(outputs=output,
                            inputs=a,
                            grad_outputs=e,
                            retain_graph=True,
                            create_graph=False)
    
    a = a + 0.01 * vjp
    output = model.forward(a)
    print(i)
    print(output)
    e = y - output


