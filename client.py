from collections import OrderedDict
from typing import Dict, Tuple
import ray
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes)

        # Check for CUDA support
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Copy parameters sent by the server into the client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}


# Define Ray-enabled Flower Client
@ray.remote
class RayFlowerClient(FlowerClient):
    """
    This class wraps the FlowerClient to enable Ray-based distributed execution.
    """
    pass


def launch_ray_clients(num_clients, trainloaders, valloaders, num_classes):
    clients = []
    for i in range(num_clients):
        train_data = trainloaders[i].dataset  # Access underlying dataset
        val_data = valloaders[i].dataset

        client = RayFlowerClient.remote(
            train_data,  # Pass dataset objects or their paths
            val_data,
            num_classes
        )
        clients.append(client)
    return clients




def generate_client_fn(trainloaders, valloaders, num_classes, use_ray=False):
    """
    Generates a client creation function, optionally using Ray-enabled clients.
    """
    if use_ray:
        def ray_client_fn(cid: str):
            # Ensure Ray actors are handled externally, e.g., in `main`
            raise ValueError(
                "Ray client function must be handled externally. "
                "Use `launch_ray_clients` to create Ray clients and pass them directly."
            )
        return ray_client_fn

    else:
        def standard_client_fn(cid: str):
            return FlowerClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_classes=num_classes
            )
        return standard_client_fn
