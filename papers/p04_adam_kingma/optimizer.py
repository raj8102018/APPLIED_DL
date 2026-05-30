import torch

class CustomAdam(torch.optim.Optimizer):
    def __init__(self,params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params,defaults)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                betas = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                for p in group['params']:

                    if p.grad is None: #generally use p.grad.data to ensure the optimizer never interacts with autograd medata but the current implementation works because we are wrapping it inside with torch.no_grad()
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1
                    grad = p.grad
                    if weight_decay > 0:
                        grad = grad + weight_decay * p
                    state['exp_avg'].mul_(betas[0]).add_((1-betas[0])*grad)
                    state['exp_avg_sq'].mul_(betas[1]).add_((1-betas[1])*(grad**2))
                    exp_avg_bias_corrn = state['exp_avg']/(1-betas[0]**state['step'])
                    exp_avg_sq_bias_corrn = state['exp_avg_sq']/(1-betas[1]**state['step'])
                    p -= (lr*exp_avg_bias_corrn)/(torch.sqrt(exp_avg_sq_bias_corrn)+eps)



