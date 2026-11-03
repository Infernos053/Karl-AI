import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2




def main():
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    batchsize = 64
    loaded_training_data = DataLoader(training_data,batch_size=batchsize)
    for x,y in loaded_training_data:
        print(f"Shape of X [N,C,H,W]: {x.shape}")
        print(f"Shape of Y : {y.shape} {y.dtype}")
        break

    device = torch.accelerator.current_accelerator if torch.accelerator.is_available() else "cpu"
    print(f"Using: {device}")

    return



if __name__ == "__main__":
    main()