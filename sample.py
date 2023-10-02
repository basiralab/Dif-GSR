import torch
import random
import numpy as np
from models.DDPM_Schedule import DDPM
from models.unet import ContextUnet
from utils import connectomes_data_loader
import yaml
from tqdm import tqdm
import pickle


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

torch.manual_seed(config["Seed"]['seed'])
random.seed(config["Seed"]['seed'])
np.random.seed(config["Seed"]['seed'])

t_dl = connectomes_data_loader(source_data=config["Data"]["morphological_data"],
                               target_data=config["Data"]["functional_data"],
                               batch_size=config["Sampling"]["batch_size"])[1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for fn in range(3):
    g = config["Sampling"]["guidance"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_dim = config["Sampling"]["source_dim"]**2
    n_feat = config["Sampling"]["n_feat"]
    save_dir = config["Sampling"]["save_dir"]

    ddpm = DDPM(denoising_model=ContextUnet(in_channels=1,
                                            n_feat=n_feat,
                                            n_classes=source_dim),
                beta1=config["Diffusion"]["beta1"],
                beta2=config["Diffusion"]["beta2"],
                n_T=config["Diffusion"]["n_T"],
                drop_prob=config["Diffusion"]["dropout_prob"])
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(f"results/Diffusion_Train/fold_{fn+1}.ckpt", map_location=device))
    ddpm.eval()
    dl = t_dl[fn]
    fin = []
    real = []
    with torch.no_grad():
        for c in tqdm(dl):
            real.append(c[0].detach().cpu())
            c = c[1].to(device)
            x_gen = ddpm.sample(c.shape[0], (1, 160, 160), device, c, guide_w=g)
            fin.append(x_gen.detach().cpu())


    with open(f'./results/Diffusion_Sample/gen_{fn}_{g}.pkl', 'wb') as f:
        pickle.dump(fin, f)

    with open(f'./results/Diffusion_Sample/target_{fn}_{g}.pkl', 'wb') as f:
        pickle.dump(real, f)

