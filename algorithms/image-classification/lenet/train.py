from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append('.')
import datasets
from model import LeNet
from config import Config

def main(conf: Config):

    dataset = datasets.CIFAR10()


    train_dataset = datasets.PlainImgDataset(
        dataset.train_image,
        dataset.train_label,
        transforms.ToTensor(),
    )
    test_dataset = datasets.PlainImgDataset(
        dataset.test_image,
        dataset.test_label,
        transforms.ToTensor(),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
    )


    device = conf.device
    device_cpu = conf.device_cpu

    model = LeNet()
    # model = nn.DataParallel(model)
    model.to(device)

    n_epoch = conf.n_epoch

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = conf.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    def train(dataloader, model, loss_fn, optimizer):
        # vanilla classification train one epoch
        size = len(dataloader.dataset)
        batchsize = dataloader.batch_size
        tbar = tqdm(total=size)
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_loss = loss.to(device_cpu).item()
            tbar.update(batchsize)
            tbar.set_postfix_str(f'loss = {cur_loss:.6f}')

    def test(dataloader, model, loss_fn):
        # vanilla classification test one epoch
        size = len(dataloader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).to(device_cpu).item()
                correct += (pred.argmax(1)==y).type(torch.float).sum().to(device_cpu).item()
        test_loss /= len(dataloader)
        lr_scheduler.step(test_loss)
        accuracy = correct / size
        print(f'learning rate = {optimizer.state_dict()["param_groups"][0]["lr"]}')
        print(f'test acc: {accuracy*100:.4f}%, test loss: {test_loss:.6f}.\n')


    for epoch in range(n_epoch):
        print(f'Epoch {epoch+1}/{n_epoch}')
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
