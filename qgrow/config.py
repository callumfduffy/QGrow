from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import qadence as q
import torch
import yaml


@dataclass
class GrowConfig:
    grow_type: str
    rate: int
    step: int
    final_depth: int
    depth: int
    n_qubits: int
    inputs: list[str]
    fm_gates: list[Callable]
    ansatz_gates: list[Callable]
    init_func: Callable
    fm_init: Callable | str
    opt_fn: Callable
    lr: float


@dataclass
class TrainConfig:
    seed: int
    save_dir: str
    n_qubits: int
    depth: int
    inputs: list[str]
    dim: int
    batch_size: int
    num_train_points: int
    num_test_points: int
    dataset_name: str
    data_filepath: str
    noise: bool
    epochs: int
    opt_fn: Callable
    lr: float
    fm_gates: list[Callable]
    ansatz_gates: list[Callable]
    init_func: Callable
    fm_init: Callable | str
    log_interval: int


@dataclass
class Config:
    train_config: TrainConfig
    grow_config: GrowConfig



class ConfigCompiler:
    def __init__(self, fileapth):
        self.filepath = fileapth

        with open(self.filepath, "r") as f:
            self.raw_config = yaml.load(f, Loader=yaml.FullLoader)

        self.init_func, self.fm_init = self.get_init_funcs()

    def create_save_dir(self):
        save_dir = self.raw_config["base_dir"]
        depth = self.raw_config["depth"]
        seed = self.raw_config["seed"]
        new_path = f"{save_dir}_depth_{depth}/{seed}"
        Path(new_path).mkdir(parents=True, exist_ok=True)
        self.raw_config["save_dir"] = new_path

    def get_init_funcs(self):
        # perhaps move
        if self.raw_config["grow"] is not None:
            init_func = partial(
                np.random.uniform, low=0, high=np.pi
            )  # partial(np.random.uniform, low=0, high=0.1)
            fm_init = partial(np.random.uniform, low=0, high=np.pi)
        else:
            # init_func = (
            #    partial(np.random.uniform, low=0, high=np.pi),
            #    "identity",
            # )

            # fm_init = (
            #    partial(np.random.uniform, low=0, high=np.pi),
            #    "identity",
            # )

            init_func = partial(
                np.random.uniform, low=0, high=np.pi
            )  # partial(np.random.uniform, low=0, high=0.1)
            fm_init = "equal_spacing"  #
            # fm_init = partial(np.random.uniform, low=0, high=np.pi/9)

        return init_func, fm_init

    def get_gates(self):
        fm_gates = []
        for fm_gate in self.raw_config["fm_gates"]:
            try:
                fm_gates.append(getattr(q,fm_gate))
            except AttributeError:
                print("gate not found!")

        ansatz_gates = []
        for a_gate in self.raw_config["ansatz_gates"]:
            try:
                ansatz_gates.append(getattr(q,a_gate))
            except AttributeError:
                print("gate not found!")
        
        self.raw_config["fm_gates"] = fm_gates
        self.raw_config["ansatz_gates"] = ansatz_gates

    def compile_grow_config(self):
        if self.raw_config["grow"] is not None:
            grow_config = GrowConfig(
                grow_type=self.raw_config["grow"]["grow_type"],
                rate=self.raw_config["grow"]["rate"],
                step=self.raw_config["grow"]["step"],
                final_depth=self.raw_config["grow"]["final_depth"],
                depth=self.raw_config["grow"]["depth"],
                n_qubits=self.raw_config["n_qubits"],
                inputs=self.raw_config["inputs"],
                fm_gates=self.raw_config["fm_gates"],
                ansatz_gates=self.raw_config["ansatz_gates"],
                init_func=self.init_func,
                fm_init=self.fm_init,
                opt_fn=getattr(torch.optim, self.raw_config["optimizer"]),
                lr=self.raw_config["lr"],
            )
            return grow_config
        else:
            return None

    def compile_train_config(self):
        train_config = TrainConfig(
            seed=self.raw_config["seed"],
            save_dir=self.raw_config["save_dir"],
            n_qubits=self.raw_config["n_qubits"],
            depth=self.raw_config["depth"],
            inputs=self.raw_config["inputs"],
            dim=self.raw_config["dim"],
            batch_size=self.raw_config["batch_size"],
            num_train_points=self.raw_config["num_train_points"],
            num_test_points=self.raw_config["num_test_points"],
            dataset_name=self.raw_config["dataset_name"],
            data_filepath=self.raw_config["data_filepath"],
            noise=self.raw_config["noise"],
            epochs=self.raw_config["epochs"],
            opt_fn=getattr(torch.optim, self.raw_config["optimizer"]),
            lr=self.raw_config["lr"],
            fm_gates=self.raw_config["fm_gates"],
            ansatz_gates=self.raw_config["ansatz_gates"],
            init_func=self.init_func,
            fm_init=self.fm_init,
            log_interval=self.raw_config["log_interval"]
        )
        return train_config

    def compile(self):
        self.create_save_dir()
        self.get_gates()
        self.grow_config = self.compile_grow_config()
        self.train_config = self.compile_train_config()
        return Config(
            train_config=self.train_config, grow_config=self.grow_config
        )
