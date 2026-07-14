# import torch
# import torchvision
# import torchvision.transforms as transforms


# def get_loader(data_path, batch_size, mode, num_workers=4):

#     transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     if mode == 'train':
#         dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
#                                             download=True, transform=transform)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                               shuffle=True, num_workers=num_workers)

#     else:
#         dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
#                                                download=True, transform=transform)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                  shuffle=False, num_workers=num_workers)

#     return dataloader

import numpy as np
import torch

class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, n_points=10000, radius=1.0, noise=0.01):
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts = np.stack([x, y], axis=1) + np.random.normal(0, noise, (n_points, 2))
        self.data = torch.tensor(pts, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_loader(data_path=None, batch_size=64, mode='train', num_workers=0, n_points=10000):
    dataset = CircleDataset(n_points=n_points)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=(mode == 'train'), num_workers=num_workers)
