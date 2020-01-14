from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import DataLoader

dataTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     ]
)

train_dataset = datasets.MNIST(
    root='./MNIST_data', train=True, transform=dataTransform, download=True
)
test_dataset = datasets.MNIST(
    root='./MNIST_data', train=False, transform=dataTransform, download=True
)


def get_train_data(batch_size):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=4
                      )


def get_test_data(batch_size):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4
                      )
