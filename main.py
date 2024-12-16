import pickle
from pathlib import Path
import hydra 
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn, RayFlowerClient, launch_ray_clients
import flwr as fl
import ray
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    # Step 1: Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Step 2: Dataset preparation
        trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
        # print(len(trainloaders), len(trainloaders[0].dataset))

        # Step 3: Launch Flower clients as Ray actors
# Initialize Ray clients
        ray_clients = launch_ray_clients(cfg.num_clients, trainloaders, validationloaders, cfg.num_classes)


        # Step 4: Define the client function to interact with Ray clients
        def ray_client_fn(cid: str):
            return ray_clients[int(cid)]

        # Step 5: FL Strategy and Server Setup
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=cfg.fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=cfg.fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=get_on_fit_config(cfg.config_fit),
            evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
        )

        # Step 6: FL Simulation with Ray clients
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: ray_clients[int(cid)],
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': cfg.num_cpus, 'num_gpus': cfg.num_gpus}  # Adjust as per setup
        )

        # Step 7: Save results
        save_path = HydraConfig.get().runtime.output_dir
        results_path = Path(save_path) / 'results.pkl'
        results = {'history': history}
        with open(str(results_path), 'wb') as h:
            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    finally:
        # Step 8: Shutdown Ray
        ray.shutdown()

if __name__ == "__main__":
    main()
