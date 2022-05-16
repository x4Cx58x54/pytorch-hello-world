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
        batch_size=256,
        n_epoch=100,
        learning_rate=1e-3,
    ),
]
