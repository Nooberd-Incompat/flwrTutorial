import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

def get_mnist(data_path: str = "./data"):
    """
    Download and preprocess the MNIST dataset.
    Args:
        data_path (str): Path to store/download the dataset.
    Returns:
        Tuple[Dataset, Dataset]: Train and Test datasets.
    """
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset


def prepare_dataset(
    num_partitions: int,
    batch_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    data_path: str = "./data",
    test_batch_size: int = 128
):
    """
    Prepare train, validation, and test loaders for federated learning.
    
    Args:
        num_partitions (int): Number of partitions (clients).
        batch_size (int): Batch size for train/validation loaders.
        val_ratio (float): Proportion of validation data from each partition.
        seed (int): Random seed for reproducibility.
        data_path (str): Path to store/download the dataset.
        test_batch_size (int): Batch size for test loader.
        
    Returns:
        Tuple[List[DataLoader], List[DataLoader], DataLoader]: 
        Train loaders, validation loaders, and test loader.
    """
    train_set, test_set = get_mnist(data_path)
    
    # Split dataset into IID partitions for clients
    num_images = len(train_set) // num_partitions
    partition_lengths = [num_images] * num_partitions
    trainsets = random_split(train_set, partition_lengths, torch.Generator().manual_seed(seed))

    trainloaders = []
    valloaders = []

    # Split each client's data into training and validation sets
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(seed))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    # Test loader
    testloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader
