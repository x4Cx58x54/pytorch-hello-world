from dataclasses import dataclass

@dataclass
class Config:
    device: ...
    device_cpu: ...
    batch_size: int
    n_epoch: int
    learning_rate: float

configs = [
    Config(
        device='cuda:0',
        device_cpu='cpu',
        batch_size=50,
        n_epoch=5000,
        learning_rate=1e-3,
    ),
    Config(
        device='cuda:1',
        device_cpu='cpu',
        batch_size=100,
        n_epoch=5000,
        learning_rate=1e-3,
    ),
]
