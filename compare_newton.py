
import torch 

def zeropower_via_newtonschulz5(G, steps=6, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

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


def gradnorm(grad):
    """
    Implements the GradNorm operator as described in Algorithm 3.

    basically layernorm for a radient, center and divide by the stddev

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


# test whitening with both methods

# create a non-whitened matrix
dim = 512
X = torch.randn(dim, dim)
scales = torch.logspace(-1, 1, dim)  # Creates scales from 0.1 to 10
non_whitened = X * scales.unsqueeze(0)  # Scale each column differently
X = non_whitened
X = gradnorm(X)

print(X.norm())

# whiten with both methods
whitened_1 = gradwhiten(X, ns_steps=6, beta=0.5)
whitened_2 = zeropower_via_newtonschulz5(X, steps=6)

# normalize both and take xxt
whitened_1 = whitened_1 / whitened_1.norm(dim=1, keepdim=True)
whitened_2 = whitened_2 / whitened_2.norm(dim=1, keepdim=True)  

sims1 = whitened_1 @ whitened_1.T
sims2 = whitened_2 @ whitened_2.T

# disable scientific notation
torch.set_printoptions(precision=4, sci_mode=False)

# sum absolute value of off diagonal elements
sims1[torch.arange(dim), torch.arange(dim)] = 0
sims2[torch.arange(dim), torch.arange(dim)] = 0

print(sims1.abs().sum())
print(sims2.abs().sum())






