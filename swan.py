import os
import torch

# this is what algorithm 3 in paper shows but something seems off here...
# def gradnorm(grad):
#     """
#     Implements the GradNorm operator as described in Algorithm 3.
    
#     Args:
#         grad: Input matrix G of shape (m x n)
        
#     Returns:
#         Normalized gradient according to the formula:
#         (G - row_mean) / sqrt(1/n * sum((G - sum(G))^2))
#     """
#     n = grad.size(1)
#     row_sums = grad.sum(dim=1, keepdim=True)
#     row_means = row_sums / n
#     centered_grad = grad - row_means
#     diff_from_sum = grad - row_sums
#     denominator = torch.sqrt((diff_from_sum.pow(2).sum(dim=1, keepdim=True) / n) + 1e-8)
    
#     # Return normalized gradient
#     return centered_grad / denominator


def gradnorm(grad):
    """
    Implements the GradNorm operator as described in Algorithm 3.

    basically layernorm for a gradient, center and divide by the stddev

    Args:
        grad: Input matrix G of shape (m x n)
        
    Returns:
        Normalized gradient according to the formula:
        (G - row_mean) / sqrt(1/n * sum((G - sum(G))^2))
    """
    row_mean = grad.mean(dim=1, keepdim=True)
    centered_grad = grad - row_mean
    denominator = torch.sqrt((centered_grad.pow(2).mean(dim=1, keepdim=True)) + 1e-8)
    return centered_grad / denominator


def gradwhiten(grad, ns_steps=6, beta=0.5):
    """
    Implements the GradWhitening operator as described in Algorithm 2.
    
    Args:
        grad: Input matrix G of shape (m x n) where m <= n
        ns_steps: Number of Newton-Schulz iterations (default: 6)
        beta: Step size for Newton-Schulz iterations (default: 0.5)
        
    Returns:
        Whitened gradient ZG where Z approximates (GG^T)^(-1/2)
    """
    # initialize
    grad = grad / grad.norm()
    Y = grad @ grad.T #+ 1e-6 * torch.eye(grad.size(0), device=grad.device, dtype=grad.dtype)
    Z = torch.eye(Y.size(0), device=grad.device, dtype=grad.dtype)
    I3 = 3 * Z
    
    #  Newton-Schulz iterations
    for _ in range(ns_steps):
        ZY = Z @ Y
        I3_minus_ZY = I3 - ZY
        Y = beta * (Y @ I3_minus_ZY)
        Z = beta * (I3_minus_ZY @ Z)

    out = Z @ grad
    return out


class SWAN(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=None, nesterov=True, ns_steps=6, weight_decay=0.0, beta=0.5, post_norm=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay, beta=beta, post_norm=post_norm)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group['params']
            lr = group['lr']
            momentum = group['momentum']

            for i, param in enumerate(params):
                g = param.grad
                if g is None:
                    continue

                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # momentum
                # compared to muon, momentum is disabled typically
                state = self.state[param]
                if 'momentum_buffer' not in state and momentum is not None:
                    state['momentum_buffer'] = torch.zeros_like(g)

                if momentum is not None:        
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf

                # norm
                # effectively layernorm for gradient
                # this should more or less be centered around 0 as is, but still need to rescale
                g = gradnorm(g)

                # get prewhitened gradient norm
                g_norm = g.norm()
                
                # whiten
                # OG form of newton schulz is used here
                w_del = gradwhiten(g, ns_steps=group['ns_steps'], beta=group['beta'])

                # post norm
                # instead of rescaling by dimensions of matrix which is a constant value, we rescale such that the 
                # norm of the whitened gradient is the same as the norm of the prewhitened gradient
                # However, since this is the norm after having applied layer norm, this should probably be a fairly constant value anyway...
                if group['post_norm']:
                    w_del *= (g_norm / w_del.norm())

                param.data.mul_(1 - lr * group['weight_decay'])
                param.data.add_(w_del, alpha=-lr)

        return loss
