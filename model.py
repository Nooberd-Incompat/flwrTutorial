import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes: int, kernel_size: int = 5, pool_size: int = 2) -> None:
        """
        Initialize the CNN model.
        Args:
            num_classes (int): Number of output classes.
            kernel_size (int): Size of the convolutional kernel. Default is 5.
            pool_size (int): Size of the pooling kernel. Default is 2.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        
        # Use adaptive calculation for fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Dynamically compute the flattening dimension
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))  # Dropout applied here
        x = self.fc3(x)
        return x

def train(net, trainloader, optimizer, epochs, device: str = None):
    """
    Train the model on the given dataset.
    Args:
        net (nn.Module): The model to train.
        trainloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        epochs (int): Number of epochs to train.
        device (str): Device to use ('cuda' or 'cpu'). Default is None (auto-detect).
    """
    criterion = nn.CrossEntropyLoss()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net.train()
    net.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

def test(net, testloader, device: str = None):
    """
    Evaluate the model on the test dataset.
    Args:
        net (nn.Module): The model to evaluate.
        testloader (DataLoader): DataLoader for test data.
        device (str): Device to use ('cuda' or 'cpu'). Default is None (auto-detect).
    Returns:
        Tuple[float, float]: Loss and accuracy of the model.
    """
    criterion = nn.CrossEntropyLoss()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total_loss = 0.0
    net.eval()
    net.to(device)
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")
    return avg_loss, accuracy
