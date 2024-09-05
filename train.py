"""
@author supermantx
@date 2024/8/29 16:56
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import SegmentationNet
from dataset import get_data_loader, CelebAMaskHQ_224x224


def train_one_epoch():
    pass


def train():
    # init dataloader
    loader = get_data_loader(CelebAMaskHQ_224x224())
    # init model
    # model = SegmentationNet(3,
    #                         [18, 36, 72, 144, 288],
    #                         2,
    #                         2,
    #                         num_groups=[6, 6, 12, 24])
    model = SegmentationNet(3,
                 [18, 36, 72, 144, 288, 576],
                 2, 2,
                 num_groups=[12, 12, 24])
    # init optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    # init lr scheduler
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    model.cuda()
    for epoch in range(100):
        model.train()
        t = tqdm(loader, ncols=100)
        for step, (imgs, masks) in enumerate(t):
            optimizer.zero_grad()
            loss = model(imgs.cuda(), masks.cuda())
            loss.backward()
            optimizer.step()
            t.desc = f"({epoch}/100) loss: {loss.item():.6f}"
            if step % 500 == 0:
                torch.save(model.state_dict(), f"model_{epoch}_{step / len(t):.2f}.pth")
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pass
        torch.save(model.state_dict(), f"model_{epoch}.pth")

def main():
    train()


if __name__ == '__main__':
    main()
