import torch

def mix_rbf_mmd2(x, y, gammas=[1]):
    xi_xj, yi_yj, xi_yj = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    xj_xj = xi_xj.diag().unsqueeze(0).expand_as(xi_xj)
    yj_yj = yi_yj.diag().unsqueeze(0).expand_as(yi_yj)
    xi_xi = xj_xj.t()
    yi_yi = yj_yj.t()

    K_XX, K_YY, K_XY = (torch.zeros(xi_xj.shape),
                        torch.zeros(yi_yj.shape),
                        torch.zeros(xi_yj.shape))

    for gamma in gammas:
        K_XX += torch.exp((-1/gamma) * (xi_xi + xj_xj - (2.*xi_xj)))
        K_YY += torch.exp((-1/gamma) * (yi_yi + yj_yj - (2.*yi_yj)))
        K_XY += torch.exp((-1/gamma) * (xi_xi + yj_yj - (2.*xi_yj)))

    m = K_XX.shape[0]
    n = K_YY.shape[0]
    k_xx_sum = torch.sum(K_XX) / (m*m)
    k_yy_sum = torch.sum(K_YY) / (n*n)
    k_xy_sum = torch.sum(K_XY) / (m*n)
    return (k_xx_sum + k_yy_sum - (2*k_xy_sum))
