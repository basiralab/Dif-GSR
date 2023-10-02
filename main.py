import torch
from tqdm import tqdm
from utils import connectomes_data_loader
import yaml
import wandb
from models.DDPM_Schedule import DDPM
from models.unet import ContextUnet
import random
import numpy as np


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config["Wandb"]["use_wandb"]:

    wandb.login(key=config["Wandb"]["key"])

torch.manual_seed(config["Seed"]['seed'])
random.seed(config["Seed"]['seed'])
np.random.seed(config["Seed"]['seed'])


t_dl, v_dl = connectomes_data_loader(source_data=config["Data"]["morphological_data"],
                                     target_data=config["Data"]["functional_data"],
                                     batch_size=config["Sampling"]["batch_size"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for fold in range(3):

    if config["Wandb"]["use_wandb"]:

        if wandb.run is not None:
            wandb.finish()

        wandb.init(project="ConDiff", name=f"fold_{fold + 1}_Diffusion_Image", config=config)

    print(f"Fold {fold + 1}")

    train_loader = t_dl[fold]
    val_loader = v_dl[fold]

    nn = ContextUnet(in_channels=1,
                     n_feat=config["Diffusion"]["n_feat"],
                     n_classes=config["Diffusion"]["source_dim"]**2)

    model = DDPM(denoising_model=nn,
                 beta1=config["Diffusion"]["beta1"],
                 beta2=config["Diffusion"]["beta2"],
                 n_T=config["Diffusion"]["n_T"],
                 drop_prob=config["Diffusion"]["dropout_prob"],
                 lr=config["Diffusion"]["lr"])

    n_epoch = config["Diffusion"]["epochs"]

    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config["Diffusion"]["lr"])

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        model.train()

        pbar = tqdm(train_loader)
        loss_ema = None
        optim.param_groups[0]['lr'] = config["Diffusion"]["lr"] * (1 - ep / n_epoch)

        for x, c in pbar:
            optim.zero_grad()
            x = x.unsqueeze(1).to(device)
            c = c.to(device)
            loss = model(x, c)
            if config["Wandb"]["use_wandb"]:
                wandb.log({"train_loss": loss,
                           "epoch": ep})
            loss.backward()
            loss_ema = loss.item()

            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        model.eval()
        pbar = tqdm(val_loader)
        with torch.no_grad():
            for x, c in pbar:
                x = x.unsqueeze(1).to(device)
                c = c.to(device)
                loss = model(x, c)
                if config["Wandb"]["use_wandb"]:
                    wandb.log({"val_loss": loss,
                               "epoch": ep})
                pbar.set_description(f"val_loss: {loss:.4f}")

    with torch.no_grad():
        with open(config["Diffusion"][f"load_dir_{fold+1}"], "wb") as f:
            torch.save(model.state_dict(), f)
