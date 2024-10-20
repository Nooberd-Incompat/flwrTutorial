import pickle
from pathlib import Path
import hydra 
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
import flwr as fl
from server import get_on_fit_config
from server import get_evaluate_fn

@hydra.main(config_path="conf", config_name="base" , version_base=None)
def main(cfg:DictConfig):
    ## 1. Environment Setup 
        ## Install Python 3.8
        ## Install miniconda (https://docs.anaconda.com/miniconda/)
        ## Activate miniconda (source miniconda3/bin/activate)
        ## Create Virtual environment using Conda (conda create -n flower_tutorial python=3.8 -y)
        ## Activate FLTutorial (conda activate flower_tutorial)
        ## Select Framework (Pytorch) (  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda==11.6 -c pytorch -c nvidia -y)
        ## Install Flower Framework (pip install flwr==1.4.0)
        ## Install Hydra (pip install hydra-core)
        ## Install Ray (pip install ray==1.11.1)

        ## Configuration setup 
            ## Import Hydra, DictConfig and OmegaConfig
            ## Create a YAML file 
    print(OmegaConf.to_yaml(cfg))


    ## 2. Dataset preparation
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size )
    print(len(trainloaders),len(trainloaders[0].dataset))

    ## 3. Client Setup
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. FL Strategy and Server Setup
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader))


    ## 5. FL Simulation
    history = fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy = strategy
    )


    ## 6. Saving results

    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) /'results.pkl'

    results = {'history':history}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol = pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
