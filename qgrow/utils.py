from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import qadence as q
import torch
import torch.nn as nn
from models import growth_circ, growth_fm_circ, trainable_freq_circ


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for repoducibility.

    Args:
        seed (int): seed value to be used.
        deterministic (bool): deterministic behaviour in pytorch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multiple GPUs

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")

def model_selector(train_config, grow_config=None) -> nn.Module:
    if grow_config is not None:
        if grow_config.grow_type == ("serial" or "interleave"):
            model = growth_fm_circ(
                n_qubits=train_config.n_qubits,
                depth=grow_config.final_depth,
                fm_depth=grow_config.depth,
                inputs=train_config.inputs,
                init_func=train_config.init_func,
                fm_init=train_config.init_func,
                growth_type=grow_config.grow_type,
                ansatz_gates=grow_config.ansatz_gates,
                fm_gates=grow_config.fm_gates,
            )

        else:
            model = growth_circ(
                n_qubits=train_config.n_qubits,
                depth=grow_config.depth,
                inputs=train_config.inputs,
                init_func=train_config.init_func,
                fm_init=train_config.init_func,
                ansatz_gates=grow_config.ansatz_gates,
                fm_gates=grow_config.fm_gates,
            )

    else:
        grow_config = None
        model = trainable_freq_circ(
            train_config.n_qubits,
            train_config.depth,
            train_config.inputs,
            train_config.init_func,
            train_config.fm_init,
            ansatz_gates=train_config.ansatz_gates,
            fm_gates=train_config.fm_gates,
        )
    return model


class ModelAnalysis:
    def __init__(self, results_dirs, labels, palette, name):
        self.results_dirs = results_dirs
        self.labels = labels
        self.palette = palette
        self.name = name

    def analyse(self, file="test_loss.npy"):
        labels = []
        mean_losses = []
        std_losses = []
        max_losses = []
        min_losses = []
        losses_all = []
        xs = []

        for idx, entry in enumerate(self.results_dirs):
            labels.append(entry)
            worst_loss = 0
            worst_idx = 0
            best_seed_losses = []
            for curr_seed in os.listdir(entry):
                seed_path = os.path.join(entry, curr_seed)
                loss_path = f"{seed_path}/{file}"
                if os.path.exists(loss_path):
                    losses = np.load(f"{seed_path}/{file}")
                    best_loss = np.min(losses)
                    best_seed_losses.append(best_loss)
                    if best_loss > worst_loss:
                        worst_loss = best_loss
                        worst_idx = curr_seed

            print(f"ENTRY: {entry}: {worst_idx}")
            # select top 30 best losses
            # best_seed_losses = np.sort(np.array(best_seed_losses))[:20]

            mean_losses.append(np.mean(best_seed_losses))
            std_losses.append(np.std(best_seed_losses))
            max_losses.append(np.max(best_seed_losses))
            min_losses.append(np.min(best_seed_losses))
            losses_all.append(np.array(best_seed_losses))
            xs.append(np.ones(len(best_seed_losses)) * (idx + 1))

        for i in range(len(labels)):
            print(f"model: {labels[i]}")
            print(
                f"mean_losses: {mean_losses[i]}+/-{std_losses[i]}, best loss: {min_losses[i]}, worst loss: {max_losses[i]}"
            )
            print("----")
        self.plot_boxplot(
            losses_all,
            xs,
            self.labels,
            self.palette,
            self.name,
        )

    def loss_curves(self, save_dir):
        labels = []
        mean_losses = []
        std_losses = []
        max_losses = []
        min_losses = []
        losses_all = []
        xs = []
        fig, ax = plt.subplots(2)

        for idx, entry in enumerate(self.results_dirs):
            labels.append(entry)
            seed_train_losses = []
            seed_test_losses = []
            for curr_seed in os.listdir(entry):
                seed_path = os.path.join(entry, curr_seed)
                train_losses = np.load(f"{seed_path}/test_loss.npy")
                test_losses = np.load(f"{seed_path}/train_loss.npy")

                min_train_losses = np.zeros_like(train_losses)
                min_test_losses = np.zeros_like(test_losses)
                curr_train_min = np.inf
                curr_test_min = np.inf

                for i in range(len(train_losses)):
                    curr_train_min = np.min([curr_train_min, train_losses[i]])
                    curr_test_min = np.min([curr_test_min, test_losses[i]])
                    min_train_losses[i] = curr_train_min
                    min_test_losses[i] = curr_test_min

                seed_train_losses.append(min_train_losses)
                seed_test_losses.append(min_test_losses)

            seed_train_losses = np.array(seed_train_losses)
            seed_test_losses = np.array(seed_test_losses)
            mean_train_losses = np.mean(seed_train_losses, axis=0)
            mean_test_losses = np.mean(seed_test_losses, axis=0)
            std_train_losses = np.std(seed_train_losses, axis=0)
            std_test_losses = np.std(seed_test_losses, axis=0)
            print(self.labels[idx])
            x = np.arange(seed_train_losses.shape[1]) * 5
            ax[0].plot(x, mean_train_losses, label=f"{self.labels[idx]}")
            ax[1].plot(x, mean_test_losses, label=f"{self.labels[idx]}")

        ax[0].set_title("train")
        ax[0].set_title("test")

        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].legend()
        fig.savefig(save_dir)
        plt.close()

    def plot_boxplot(self, vals, xs, labels, palette, name):
        plt.boxplot(vals, tick_labels=labels, showfliers=False, showmeans=True)
        for x, val, c in zip(xs, vals, palette):
            plt.scatter(x, val, alpha=0.2, color=c)
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.ylabel("Mean Square Error (MSE)")
        plt.tight_layout()
        plt.savefig(name)
        plt.close()

    def loss_analysis(self):
        pass

    def convergence_analysis(self, thresholds, name):
        labels = []
        results = []
        for idx, entry in enumerate(self.results_dirs):
            # print(entry)
            labels.append(entry)
            model_results = []
            for curr_seed in os.listdir(entry):
                # print("seed: ", curr_seed)
                seed_path = os.path.join(entry, curr_seed)
                losses = np.load(f"{seed_path}/test_loss.npy")
                seed_epochs = []
                for j, thresh in enumerate(thresholds):
                    thresh_epochs = np.where(losses < thresh)[0]
                    # print(thresh_epochs)
                    # print(np.min(losses))
                    if thresh_epochs.size > 0:
                        thresh_epoch = thresh_epochs[0]
                        # due to logging every 5
                        seed_epochs.append(thresh_epoch * 5)
                    else:
                        seed_epochs.append(np.nan)

                model_results.append(seed_epochs)

            results.append(np.array(model_results))
        print(np.array(results).shape)
        results = np.array(results)
        results = np.nanmean(results, axis=1)
        # for each we can find the mean and std and plot
        """
        means = []
        for i in range(len(labels)):
            model_results = results[i]
            print(self.labels[i])
            print(model_results.shape)
            model_means = []
            for j, t in enumerate(thresholds):
                results_at_t = model_results[:, j]
                # take the mean
                mean_results_at_t = np.mean(results_at_t)
                model_means.append(mean_results_at_t)
            means.append(np.array(model_means))
        print(means)
        """
        fig, ax = plt.subplots()
        for i in range(len(labels)):
            # print(results[i])
            ax.plot(
                thresholds, results[i], "-o", label=self.labels[i], alpha=0.4
            )
        ax.legend()
        ax.set_xscale("log")
        plt.savefig(name)
        plt.close()

