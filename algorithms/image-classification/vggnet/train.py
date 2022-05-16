from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter

import sys
sys.path.append('.')
import datasets
from model import VGGNet
from config import Config


def main(conf: Config):

    dataset = datasets.CIFAR10()

    train_dataset = datasets.PlainImgDataset(
        dataset.train_image,
        dataset.train_label,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            # transforms.Resize((64, 64)),
        ])
    )
    test_dataset = datasets.PlainImgDataset(
        dataset.test_image,
        dataset.test_label,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            # transforms.Resize((64, 64)),
        ])
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


    device = 'cuda'
    device_cpu = 'cpu'

    model = conf.model
    # model = nn.DataParallel(model)
    model.to(device)

    n_epoch = conf.n_epoch

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = conf.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    writer = SummaryWriter('./log/' + str(datetime.now()) + conf.name)
    writer.add_text('comment', conf.__repr__())

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
        print(f'Epoch {epoch+1}/{n_epoch}')
        time_start = datetime.now()
        train(epoch)
        test(epoch)
        time_end = datetime.now()
        time_delta = time_end - time_start
        writer.add_text('epoch duration (train+test)', str(time_delta), epoch)
