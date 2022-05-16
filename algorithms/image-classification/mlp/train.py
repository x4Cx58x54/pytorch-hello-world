import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('.')
import datasets
from model import MLP
from config import Config

def main(conf: Config):

    dataset = datasets.MNIST()


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
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
    )


    device = conf.device
    device_cpu = conf.device_cpu

    model = MLP()
    # model = nn.DataParallel(model)
    model.to(device)

    n_epoch = conf.n_epoch

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = conf.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    writer = SummaryWriter('./log/' + str(datetime.now()) + conf.__repr__())

    def train(epoch):
        # vanilla classification train one epoch
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_loss = loss.to(device_cpu).item()
            writer.add_scalar('train loss', cur_loss, epoch)

    def test(epoch):
        # vanilla classification test one epoch
        size = len(test_dataloader.dataset)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).to(device_cpu).item()
                correct += (pred.argmax(1)==y).type(torch.float).sum().to(device_cpu).item()
        test_loss /= len(test_dataloader)
        lr_scheduler.step(test_loss)
        accuracy = correct / size
        writer.add_scalar('learning rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
        writer.add_scalar('test acc', accuracy, epoch)
        writer.add_scalar('test loss', test_loss, epoch)


    for epoch in range(n_epoch):
        train(epoch)
        test(epoch)
