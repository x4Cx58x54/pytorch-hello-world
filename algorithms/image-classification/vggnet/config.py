from dataclasses import dataclass
from model import VGGNet

@dataclass
class Config:
    name: str
    model: VGGNet
    device: ...
    device_cpu: ...
    batch_size: int
    n_epoch: int
    learning_rate: float

configs = [
    Config(
        name='config1',
        model=VGGNet(16, 10),
        device='cuda:0',
        device_cpu='cpu',
        batch_size=256,
        n_epoch=100,
        learning_rate=1e-3,
    ),
]
