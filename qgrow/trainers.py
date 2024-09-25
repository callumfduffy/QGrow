from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models import growth_circ, growth_fm_circ
from qadence.draw import savefig

class BaseTrainer:
    """
    Base class for training and evaluating models.

    Parameters:
    ----------
    model : nn.Module
        The model to be trained.
    opt : torch.optim.Optimizer
        Optimizer for training the model.
    dim : int
        Dimension of the input data.
    log_interval : int
        Number of epochs between logging training and validation losses.
    device : torch.device
        The device on which to run the training (e.g., 'cpu' or 'cuda').
    save_dir : Optional[str], default=None
        Directory to save model checkpoints and visualizations.
    grow : Optional[object], default=None
        Object containing parameters for model growth during training.
    vis_bootstrap : Optional[object], default=None
        Object for visualizing bootstrapped data at different frequencies.
    """

    def __init__(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        dim: int,
        log_interval: int,
        device: torch.device,
        save_dir: Optional[str] = None,
        grow: Optional[object] = None,
        vis_bootstrap: Optional[object] = None,
    ):
        self.model = model
        self.opt = opt
        self.dim = dim
        self.log_interval = log_interval
        self.device = device
        self.save_dir = save_dir
        self.grow = grow
        self.vis_bootstrap = vis_bootstrap

        # define metrics to track during training
        self.train_loss_log = []
        self.test_loss_log = []
        self.best_train_loss = np.inf
        self.best_test_loss = np.inf
        self.grow_intervals = []

    def param_transfer(
        self, old_model: nn.Module, new_model: nn.Module
    ) -> nn.Module:
        """
        Transfers common parameters between two models.

        Parameters:
        ----------
        old_model : nn.Module
            The model from which parameters will be transferred.
        new_model : nn.Module
            The model to which parameters will be transferred.

        Returns:
        -------
        new_model : nn.Module
            The new model with transferred parameters.
        """
        old_params = old_model._params
        old_keys = old_params.keys()
        new_params = new_model._params
        new_keys = new_params.keys()
        for i, old_key in enumerate(old_keys):
            for j, new_key in enumerate(new_keys):
                if old_key == new_key:
                    old_param = old_params[old_key]
                    new_params[old_key] = old_param

        new_model._params = new_params
        return new_model

    def _check_and_grow_model(self, epoch: int) -> None:
        """Checks and grows the model if necessary based on epoch and grow parameters"""
        if (
            (self.grow.depth < self.grow.final_depth)
            and (epoch > 0)
            and (epoch % self.grow.rate == 0)
        ):
            # before growing circuit we can vis the bootstrap
            if self.vis_bootstrap is not None:
                self.bootstrap(self.grow.depth)

            self._grow_model(epoch)

    def _grow_model(self, epoch: int) -> None:
        """Grows the model and transfers parameters from the previous model."""
        self.grow_intervals.append(epoch)

        new_depth = np.min(
            [self.grow.depth + self.grow.step, self.grow.final_depth]
        )

        if self.grow.depth != new_depth:
            print(
                f"Epoch: {epoch}, New Depth {new_depth} from {self.grow.depth}"
            )
            self.grow.depth = new_depth
            new_model = self._create_new_model()
            if self.save_dir is not None:
                savefig(
                    new_model,
                    filename=f"{self.save_dir}/model_{new_depth}.pdf",
                )
            self.model = self.param_transfer(
                old_model=self.model, new_model=new_model
            )
            self.model = self.model.to(self.device)
            self.opt = self.grow.opt_fn(
                self.model.parameters(), lr=self.grow.lr
            )

    def _create_new_model(self) -> nn.Module:
        """Creates a new model based on the growth parameters."""
        if self.grow.grow_type in ["serial", "interleave"]:
            return growth_fm_circ(
                n_qubits=self.grow.n_qubits,
                depth=self.grow.final_depth,
                fm_depth=self.grow.depth,
                inputs=self.grow.inputs,
                init_func=self.grow.init_func,
                fm_init=self.grow.init_func,
                growth_type=self.grow.grow_type,
                ansatz_gates=self.grow.ansatz_gates,
                fm_gates=self.grow.fm_gates,
            )
        else:
            return growth_circ(
                n_qubits=self.grow.n_qubits,
                depth=self.grow.depth,
                inputs=self.grow.inputs,
                init_func=self.grow.init_func,
                fm_init=self.grow.init_func,
                ansatz_gates=self.grow.ansatz_gates,
                fm_gates=self.grow.fm_gates,
            )

    def _save_best_losses(self) -> None:
        """Saves the best training and test losses."""
        print(f"best train loss: {self.best_train_loss:.4}")
        print(f"best test loss: {self.best_test_loss:.4}")

        if self.save_dir is not None:
            np.save(
                f"{self.save_dir}/best_train_loss.npy", self.best_train_loss
            )
            np.save(f"{self.save_dir}/best_test_loss.npy", self.best_test_loss)
            np.save(
                f"{self.save_dir}/train_loss.npy",
                np.array(self.train_loss_log),
            )
            np.save(
                f"{self.save_dir}/test_loss.npy", np.array(self.test_loss_log)
            )

    def vis_loss(self, log_scale=True) -> None:
        """
        Visualizes the training and testing loss over epochs.

        Creates a log-scale plot of the training and testing losses.
        Optionally includes vertical lines to indicate growth intervals.
        """
        fig, ax = plt.subplots()
        x = np.arange(
            int(len(self.train_loss_log) * self.log_interval),
            step=self.log_interval,
        )
        print(self.train_loss_log)
        ax.plot(x, self.train_loss_log, label="train")
        if len(self.test_loss_log) > 0:
            ax.plot(x, self.test_loss_log, label="test")
        if log_scale:
            ax.set_yscale("log")
        if self.grow is not None:
            ax.vlines(x=self.grow_intervals, ymin=0, ymax=1.0, color="r")
        plt.legend()
        if self.save_dir is not None:
            fig.savefig(f"{self.save_dir}/loss_history.pdf")
        plt.close()


class Trainer(BaseTrainer):
    """
    Trainer class for training and evaluating a neural network model.

    Parameters:
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    test_dataloader : torch.utils.data.DataLoader
        DataLoader for the testing/validation dataset.
    opt : torch.optim.Optimizer
        Optimizer for training the model.
    dim : int
        Dimension of the input data.
    log_interval : int
        Number of epochs between logging training and validation losses.
    device : torch.device
        The device on which to run the training (e.g., 'cpu' or 'cuda').
    save_dir : Optional[str], default=None
        Directory to save model checkpoints and visualizations.
    grow : Optional[object], default=None
        Object containing parameters for model growth during training.
    vis_bootstrap : Optional[object], default=None
        Object for visualizing bootstrapped data at different frequencies.
    """

    def __init__(
        self,
        *args,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.cost = nn.MSELoss()  # could provide as an argument

    def bootstrap(self, depth: int) -> None:
        """
        Visualize model performance on bootstrapped data at different frequencies.

        Parameters:
        ----------
        depth : int
            The current depth of the model.
        """

        for i in range(1, 4):
            data, label = self.vis_bootstrap.get_freq(i)
            data = torch.tensor(data).reshape(-1, self.dim).to(self.device)
            labels = torch.tensor(label).reshape(-1, 1).to(self.device)

            # want to extract outputs from model
            outputs = self.model(data).detach().cpu()

            # compare similarity between label and outputs
            bootstrap_loss = self.cost(outputs, labels).detach().cpu().numpy()
            # could also extract freq spectrum from model and results

            if self.dim == 1:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(
                    data.flatten(),
                    outputs.numpy().flatten(),
                    label=f"model mse: {bootstrap_loss:.4f}",
                )
                ax.plot(
                    data.flatten(),
                    labels.detach().cpu().numpy().flatten(),
                    label="true",
                )
                ax.legend()
                plt.tight_layout()
                fig.savefig(f"{self.save_dir}/bootstrap_{depth}_freqs_{i}.pdf")
                plt.close()

    def train(self, epochs: int) -> None:
        """
        The main training loop for the model.

        Parameters:
        ----------
        epochs : int
            The number of epochs to train the model.
        """

        for epoch in range(epochs):
            if self.grow is not None:
                self._check_and_grow_model(epoch)
            train_loss = self._train_single_epoch()

            if epoch % self.log_interval == 0:
                self._evaluate_and_log(epoch, train_loss)

        self._finialize_training()

    def _train_single_epoch(self) -> torch.Tensor:
        """Helper function to train the model for a single epoch."""
        self.model.train()
        for batch_id, (data_batch, label_batch) in enumerate(
            self.train_dataloader
        ):
            data_batch = data_batch.reshape(-1, self.dim).to(self.device)
            label_batch = label_batch.reshape(-1, 1).to(self.device)
            self.opt.zero_grad()
            loss = self.cost(self.model(data_batch), label_batch)
            loss.backward()
            self.opt.step()
        
        return loss

    def _evaluate_and_log(self, epoch: int, train_loss: torch.Tensor) -> None:
        """Evaluates the model on the test set and logs performance."""
        #train_loss = self._calculate_loss(self.train_dataloader)
        train_loss = train_loss.detach().cpu().numpy()
        test_loss = self._calculate_loss(self.test_dataloader)

        # update logs
        self.train_loss_log.append(train_loss)
        self.test_loss_log.append(test_loss)

        # update best losses
        self.best_train_loss = np.min([self.best_train_loss, train_loss])
        self.best_test_loss = np.min([self.best_test_loss, test_loss])

        # print
        print(
            f"epoch: {epoch}, train loss: {train_loss:.4}, test loss: {test_loss:.4}"
        )

    def _calculate_loss(
        self, dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Calculates the loss for the given dataloader"""
        self.model.eval()
        data, labels = next(iter(dataloader))
        data = data.reshape(-1, self.dim).to(self.device)
        labels = labels.reshape(-1, 1).to(self.device)
        with torch.no_grad():
            loss = self.cost(self.model(data), labels)
        return loss.detach().cpu().numpy()

    def _finialize_training(self) -> None:
        print("Training Finished!")
        if self.vis_bootstrap:
            self.bootstrap(self.grow.depth)
        if self.dim ==1:
            self._visualise_final_model_output()
        self._save_best_losses()

    def _save_best_losses(self) -> None:
        """Saves the best training and test losses."""
        print(f"best train loss: {self.best_train_loss:.4}")
        print(f"best test loss: {self.best_test_loss:.4}")

        if self.save_dir is not None:
            np.save(
                f"{self.save_dir}/best_train_loss.npy", self.best_train_loss
            )
            np.save(f"{self.save_dir}/best_test_loss.npy", self.best_test_loss)
            np.save(
                f"{self.save_dir}/train_loss.npy",
                np.array(self.train_loss_log),
            )
            np.save(
                f"{self.save_dir}/test_loss.npy", np.array(self.test_loss_log)
            )

    def _visualise_final_model_output(self) -> None:
        """Visualise the final model output on the test data."""
        test_data, test_labels = next(iter(self.test_dataloader))
        test_data = test_data.reshape(-1, self.dim).to(self.device)
        test_labels = test_labels.reshape(-1, 1).to(self.device)
        model_output = self.model(test_data).detach().cpu().numpy().flatten()

        #test_loss = (
        #    self.cost(self.model(test_data), test_labels)
        #    .detach()
        #    .cpu()
        #    .numpy()
        #)
        train_data, train_labels = next(iter(self.train_dataloader))
        train_data = train_data.flatten()
        test_labels = test_labels.flatten()
        test_data = test_data.detach().cpu().flatten()
        test_labels = test_labels.detach().cpu().flatten()

        indices = np.argsort(test_data)
        test_data = test_data[indices]
        test_labels = test_labels[indices]
        model_output = model_output[indices]

        plt.scatter(
            train_data, train_labels, alpha=0.4, label="train", color="r"
        )
        plt.plot(
            test_data.detach().cpu().numpy().flatten(),
            self.model(test_data).detach().cpu().numpy().flatten(),
            "-o",
            label="model",
            alpha=0.8,
        )
        plt.plot(
            test_data.detach().cpu().numpy().flatten(),
            test_labels.detach().cpu().numpy().flatten(),
            "o",
            label="test",
            alpha=0.4,
        )
        plt.tight_layout()
        plt.legend()
        if self.save_dir is not None:
            plt.savefig(f"{self.save_dir}/final_output.pdf")
        plt.close()


class DQCTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        sol,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sol = sol

    def train(self, epochs):
        for epoch in range(epochs):
            if self.grow:
                self._check_and_grow_model(epoch)
                self.sol.model = self.model
            loss = self._train_single_epoch()
            if epoch % self.log_interval == 0:
                train_loss = loss.detach().cpu().numpy()
                self.train_loss_log.append(train_loss)
                self.best_train_loss = np.min([self.best_train_loss, train_loss])
                # save best loss
                print(
                    f"epoch: {epoch}, train loss: {train_loss:.4}"
                )
    def _train_single_epoch(self) -> torch.Tensor:
        self.opt.zero_grad()
        loss = (
            self.sol.left_boundary()
            + self.sol.right_boundary()
            + self.sol.top_boundary()
            + self.sol.bottom_boundary()
            + self.sol.interior()
        )
        loss.backward()
        self.opt.step()
        return loss