import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], requires_grad=True)

print(x)

x_pred = x**2 

print(x_pred)
print(x_pred.grad_fn)


y = torch.tensor([[5., 5., 5.], [5., 5., 5.], [0., 0., 0.]])
e = y - x_pred


vjp, = torch.autograd.grad(outputs=x_pred,
                           inputs=x,
                           grad_outputs=e,
                           retain_graph=True,
                           create_graph=False)
print(vjp)

final = x - 0.1 * vjp
final = final**2
print(final)