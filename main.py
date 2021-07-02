import os
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from dataset.dataset import ImageFolderDataset
from model.unet import UNet as Model


def operate(phase):
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader
    for idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output=lastactivation(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if phase != 'train':
        torchvision.utils.save_image(torch.cat([data, target,output], dim=2), f'result/{args.savefolder}/{e}_{idx}.jpg')

        print(f'{e}/{args.epoch},{idx}/{len(loader)},loss:{loss.item():.4f}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--datasetpath', default='../data')
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--savefolder', default='tmp')
    args = parser.parse_args()

    savefolder = f'result/{args.savefolder}'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder, exist_ok=True)

    device = args.device
    criterion = nn.L1Loss()
    model = Model().to(device)
    lastactivation=nn.Sigmoid()
    optimizer = torch.optim.Adam(model.parameters())

    s, m = 0.5, 0.5
    dataset = ImageFolderDataset('../data/celeba/img_align_celeba',
                                 transform=T.Compose(
                                     [T.Resize((args.size, args.size)), T.GaussianBlur(kernel_size=51,sigma=(0.1,100.0)), T.ToTensor()]),
                                 target_transform=T.Compose([T.Resize((args.size, args.size)), T.ToTensor()]))
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                                 len(dataset) - int(len(dataset) * 0.8)])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batchsize, shuffle=True,
                                              num_workers=cpu_count())
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize, shuffle=True, num_workers=cpu_count())
    for e in range(args.epoch):
        operate('train')
        operate('val')
