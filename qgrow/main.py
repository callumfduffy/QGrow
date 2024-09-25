from __future__ import annotations

import argparse

import torch
from config import ConfigCompiler
from data import create_dataloaders, setup_dataset
from qadence.draw import savefig
from trainers import DQCTrainer, Trainer
from utils import model_selector, set_random_seed


torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default="/Users/callum/Desktop/pasqal/growing_circuit/refactored/configs/2d/block_grow_laplace.yaml",
    help="path to config file",
)


def main(config):
    train_config = config.train_config
    grow_config = config.grow_config

    # set random seed
    set_random_seed(train_config.seed)
    # set device
    device = torch.device("cpu")

    # define model determine whether growing circuit or not
    model = model_selector(train_config=train_config, grow_config=grow_config)
    savefig(model, f"{train_config.save_dir}/model_viz.pdf")

    # choosing of opt from config
    opt_fn = train_config.opt_fn
    opt = opt_fn(model.parameters(), lr=train_config.lr)

    dataset = setup_dataset(
        train_config.dataset_name,
        train_config.data_filepath,
        train_config.num_train_points,
        train_config.num_test_points,
        train_config.noise,
        model,
        train_config.dim,
        device,
    )

    if train_config.dataset_name != "laplace":
        train_dataloader, test_dataloader = create_dataloaders(
            dataset,
            train_config.num_train_points,
            train_config.num_test_points,
            train_config.batch_size,
        )

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            opt=opt,
            dim=train_config.dim,
            log_interval=train_config.log_interval,
            device=device,
            save_dir=train_config.save_dir,
            grow=grow_config,
        )

    else:
        trainer = DQCTrainer(
            model=model,
            sol=dataset,
            opt=opt,
            dim=train_config.dim,
            log_interval=train_config.log_interval,
            device=device,
            save_dir=train_config.save_dir,
            grow=grow_config,
        )

    trainer.train(train_config.epochs)
    trainer.vis_loss()


if __name__ == "__main__":
    args = parser.parse_args()
    config_compiler = ConfigCompiler(args.config_path)
    config = config_compiler.compile()
    main(config)
