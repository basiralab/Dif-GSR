import torch
import torch.nn as nn
from tqdm import tqdm


def ddpm_schedules(beta1, beta2, T):

    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """

    assert beta1 < beta2 < 1.0
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    def __init__(self,
                 denoising_model,
                 beta1,
                 beta2,
                 n_T,
                 drop_prob=0.1,
                 lr=1e-4):
        super(DDPM, self).__init__()
        self.nn_model = denoising_model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        for k, v in ddpm_schedules(self.beta1, self.beta2, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(torch.zeros_like(_ts) + self.drop_prob).to(self.device)

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, context_i_1, guide_w=0.0):

        x_i = torch.randn(n_sample, *size).to(device)
        c_i = context_i_1.view(x_i.shape[0], -1)
        context_mask = torch.zeros((2 * c_i.shape[0])).to(device)
        c_i = c_i.repeat(2, 1)

        context_mask[n_sample:] = 1.  # makes second half of batch context free
        print()
        for i in tqdm(range(self.n_T, 0, -1)):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )

        return x_i


