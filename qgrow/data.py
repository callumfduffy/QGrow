from __future__ import annotations

from functools import partial
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import cm
from models import trainable_freq_circ
from scipy.fft import fft, fftfreq
from torch import Tensor, ones_like, rand, sin
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def save_data(data, filename):
    d = {}
    for i in range(len(data)):
        d[f"dim_{i}"] = data[i]
    df = pd.DataFrame(data=d)
    df.to_csv(filename, index=False)
    return df


class StudentTeacherDataset(Dataset):
    def __init__(self, filepath: str, num_points: int, noise=None):
        """Pytorch dataset for the student teacher dataset.

        Args:
            filepath (str): filepath to dataset
            num_points (int): total number of points to load
            noise (float, optional): variance of noise. Defaults to None.
        """
        self.filepath = filepath
        self.num_points = num_points
        self.noise = noise

        print(f"loading {filepath}...")
        self.data = pd.read_csv(filepath).to_numpy()
        if num_points < 1000:
            indices = np.random.choice(
                np.arange(self.data.shape[0]), size=num_points, replace=False
            )
            self.data = self.data[indices]
        else:
            self.data = self.data[:num_points]
        self.data_shape = self.data.shape
        print(f"data shape: {self.data_shape}")
        print("loading finished!")

        if noise is not None:
            for idx in range(self.data_shape[0]):
                if idx % 2 == 0:
                    new_val = self.data[idx, 1] + (
                        np.random.normal(loc=0.0, scale=noise, size=1)
                    )
                    if (new_val <= 1) and (new_val >= -1):
                        self.data[idx, 1] = new_val

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index, :-1]
        label = self.data[index, -1]
        return sample, label


class DomainSampling(torch.nn.Module):
    def __init__(
        self,
        exp_fn: Callable[[Tensor], Tensor],
        n_inputs: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.exp_fn = exp_fn
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.X_POS, self.Y_POS = [i for i in range(n_inputs)]

    def sample(self, requires_grad: bool = False) -> Tensor:
        return rand(
            (self.batch_size, self.n_inputs),
            requires_grad=requires_grad,
            device=self.device,
            dtype=self.dtype,
        )

    def left_boundary(self) -> Tensor:  # u(0,y)=0
        sample = self.sample()
        sample[:, self.X_POS] = 0.0
        return self.exp_fn(sample).pow(2).mean()

    def right_boundary(self) -> Tensor:  # u(L,y)=0
        sample = self.sample()
        sample[:, self.X_POS] = 1.0
        return self.exp_fn(sample).pow(2).mean()

    def top_boundary(self) -> Tensor:  # u(x,H)=0
        sample = self.sample()
        sample[:, self.Y_POS] = 1.0
        return self.exp_fn(sample).pow(2).mean()

    def bottom_boundary(self) -> Tensor:  # u(x,0)=f(x)
        sample = self.sample()
        sample[:, self.Y_POS] = 0.0
        return (self.exp_fn(sample) - sin(np.pi * sample[:, 0])).pow(2).mean()

    def interior(self) -> Tensor:  # uxx+uyy=0
        sample = self.sample(requires_grad=True)
        f = self.exp_fn(sample)
        dfdxy = grad(
            f,
            sample,
            ones_like(f),
            create_graph=True,
        )[0]
        dfdxxdyy = grad(
            dfdxy,
            sample,
            ones_like(dfdxy),
        )[0]

        return (
            (dfdxxdyy[:, self.X_POS] + dfdxxdyy[:, self.Y_POS]).pow(2).mean()
        )


class SumOfSines(Dataset):
    def __init__(self, filepath: str, num_points: int, noise=None):
        self.filepath = filepath
        self.num_points = num_points
        self.noise = noise

        print(f"loading {filepath}...")

        self.data = pd.read_csv(filepath).to_numpy()
        # shuffle and select data
        indices = np.random.choice(
            np.arange(self.data.shape[0]), size=num_points, replace=False
        )
        self.data = self.data[indices]
        self.data_shape = self.data.shape
        print(f"data shape: {self.data_shape}")
        print("loading finished!")

    def get_freq(self, freq):
        bootstrap_filepath = (
            self.filepath.split("boot")[0] + f"boot_{freq}.csv"
        )
        data = pd.read_csv(bootstrap_filepath).to_numpy()
        return data[:, :-1], data[:, -1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index, :-1]
        label = self.data[index, -1]
        return sample, label


def create_dataloaders(
    dataset: Dataset,
    num_train_points: int,
    num_test_points: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    indices = list(range(int(num_train_points + num_test_points)))
    np.random.shuffle(indices)
    train_indices, test_indices = (
        indices[:num_train_points],
        indices[num_train_points : num_train_points + num_test_points],
    )

    train_subsampler = SubsetRandomSampler(train_indices)
    test_subsampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_subsampler,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=num_test_points,
        sampler=test_subsampler,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def setup_dataset(
    dataset_name,
    data_filepath,
    num_train_points,
    num_test_points,
    noise,
    model,
    dim, 
    device,
) -> Union[Dataset, nn.Module]:
    if dataset_name == "student_teacher":
        dataset = StudentTeacherDataset(
            filepath=data_filepath,
            num_points=int(num_train_points + num_test_points),
            noise=noise,
        )
    elif dataset_name == "laplace":
        dataset = DomainSampling(model, dim, num_train_points, device, torch.float32)

    return dataset


def create_sum_of_sine_1d():
    freqs = np.array(
        [1.0, 1.2, 3.0]
    )  # np.array([1, 1.5, 2.2, 3.0, 3.5])  # np.array([1, 3, 5, 10, 12])
    amplitudes = [1 / 3] * 3  # np.array(
    # [1, 0.8, 0.8, 0.7, 0.6]
    # )  # 1 / freqs  # [0.5, 0.3, 0.2, 0.1, 0.05]
    phases = [0, 0, 0, 0, 0]  # [0, np.pi/ 4, np.pi/2, np.pi/8, np.pi/10]
    num_samples = 2000

    t = np.linspace(-4 * np.pi, 4 * np.pi, num_samples)
    signal = np.zeros_like(t)
    fig, ax = plt.subplots(freqs.shape[0])
    # normalizing as we go might mess up vis let us see
    count = 1
    for freq, amp, phase in zip(freqs, amplitudes, phases):
        sine_wave = amp * np.cos(freq * t + phase)
        signal += sine_wave
        # norm_signal = signal / 3
        # signal_max = np.max(np.abs(signal))
        # normalized_signal = signal / signal_max
        print(signal)
        print(t.shape)
        print(count)
        ax[count - 1].plot(t, signal, "-o")
        save_data(
            [t, signal],
            f"datasets/sum_of_sines_1d_depth_3_boot_{count}.csv",
        )
        count += 1

    plt.tight_layout()
    fig.savefig("testing_sine_sum_1d.pdf")
    plt.close()

    y_hat = fft(signal[num_samples // 2 :])
    x_hat = fftfreq(num_samples // 2, d=(t[1] - t[0]))[: num_samples // 4]
    mag = np.abs(y_hat[0 : num_samples // 4])
    plt.plot(x_hat, mag)
    plt.xlim(0, 4)
    plt.savefig("spectrum.pdf")
    plt.close()

    dom_freqs = x_hat[np.where(mag > 100)]
    # save_data(
    #    [t, norm_signal],
    #    "datasets/sum_of_sines_1d_depth_5.csv",
    # )


#create_sum_of_sine_1d()


def find_freq():
    freqs = np.array(
        [1.0, 1.2, 3.0]
    )  # np.array([1, 1.5, 2.2, 3.0, 3.5])  # np.array([1, 3, 5, 10, 12])
    amplitudes = [1 / 3] * 3  # np.array(
    # [1, 0.8, 0.8, 0.7, 0.6]
    # )  # 1 / freqs  # [0.5, 0.3, 0.2, 0.1, 0.05]
    phases = [0, 0, 0, 0, 0]  # [0, np.pi/ 4, np.pi/2, np.pi/8, np.pi/10]
    num_samples = 8000
    T = 4 / num_samples
    t = np.linspace(-1 * num_samples * T / 2, num_samples * T / 2, num_samples)
    signal = np.zeros_like(t)
    fig, ax = plt.subplots(freqs.shape[0])
    # normalizing as we go might mess up vis let us see
    count = 1
    for freq, amp, phase in zip(freqs, amplitudes, phases):
        sine_wave = amp * np.cos(freq * t * 2 * np.pi)
        signal += sine_wave
        # norm_signal = signal / 3
        # signal_max = np.max(np.abs(signal))
        # normalized_signal = signal / signal_max
        print(signal)
        print(t.shape)
        print(count)
        ax[count - 1].plot(t, signal, "-o")
        count += 1

    plt.tight_layout()
    fig.savefig("testing_sine_sum_1d.pdf")
    plt.close()

    y_hat = fft(signal)
    x_hat = fftfreq(num_samples, T)[: num_samples // 2]
    mag = np.abs(y_hat[0 : num_samples // 2])
    plt.plot(x_hat, 2 / num_samples * np.abs(y_hat[0 : num_samples // 2]))
    plt.xlim(0, 4)
    plt.savefig("spectrum.pdf")
    plt.close()

    dom_freqs = x_hat[np.where(mag > 100)]


def create_sum_of_sine_2d():
    # freqs_1 = np.array([0.5, 1.2, 1.5, 2, 4])
    # freqs_2 = np.array([0.1, 0.25, 0.4, 1.5, 2])
    freqs_1 = np.array([1.1, 1.2, 1.4, 2, 2.3])
    freqs_2 = np.array([1.15, 1.25, 2.4, 2.5, 3])

    amplitudes_1 = [5, 4, 3, 2, 1]
    amplitudes_2 = [5, 4, 3, 2, 1]
    num_samples = 1000

    x = np.linspace(0, 1, num_samples)
    y = np.linspace(0, 1, num_samples)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((num_samples, num_samples))

    fig, ax = plt.subplots(5, subplot_kw={"projection": "3d"})

    for k in range(1, freqs_1.shape[0] + 1):
        # final surface
        Z_temp = np.zeros((num_samples, num_samples))
        for i, fx in enumerate(freqs_1[:k]):
            for j, fy in enumerate(freqs_2[:k]):
                Z_temp += (
                    amplitudes_1[i]
                    * np.sin(2 * np.pi * fx * X)
                    * amplitudes_2[j]
                    * np.sin(2 * np.pi * fy * Y)
                )

        signal_max = np.max(np.abs(Z_temp))
        normalized_signal = Z_temp / signal_max

        surf = ax[i].plot_surface(
            X,
            Y,
            normalized_signal,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
    plt.savefig("testing_sine_sum_2d.pdf")

    # final surface
    for i, fx in enumerate(freqs_1):
        for j, fy in enumerate(freqs_2):
            Z += (
                amplitudes_1[i]
                * np.sin(2 * np.pi * fx * X)
                * amplitudes_2[j]
                * np.sin(2 * np.pi * fy * Y)
            )

    signal_max = np.max(np.abs(Z))
    normalized_signal = Z / signal_max

    fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        X,
        Y,
        normalized_signal,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf)
    fig.savefig("sines_3d.pdf")

    save_data(
        [X.ravel(), Y.ravel(), Z.ravel()],
        "datasets/sum_of_sines_2d_depth_5.csv",
    )


def create_sum_of_sine_2d_v2():
    # freqs_1 = np.array([0.5, 1.2, 1.5, 2, 4])
    # freqs_2 = np.array([0.1, 0.25, 0.4, 1.5, 2])
    freqs_1 = np.array([1, 2, 3, 4, 5])  # np.array([0.6, 1.5, 2.1, 4.2, 4.5])
    freqs_2 = np.array([1, 2, 3, 4, 5])  # np.array([0.9, 1.75, 3.4, 5.0])

    amplitudes_1 = 1 / freqs_1  # [1,0.9, 0.8, 0.7, 0.6]#1 / freqs_1
    amplitudes_2 = 1 / freqs_2  # [1,0.9, 0.8, 0.7, 0.6]#1 / freqs_2
    num_samples = 5000

    x = np.linspace(0, 1, num_samples)
    y = np.linspace(0, 1, num_samples)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((num_samples, num_samples))

    fig, ax = plt.subplots(5, subplot_kw={"projection": "3d"})

    for k in range(1, freqs_1.shape[0] + 1):
        # final surface
        Z_temp = np.zeros((num_samples, num_samples))
        for i, fx in enumerate(freqs_1[:k]):
            for j, fy in enumerate(freqs_2[:k]):
                Z_temp += (
                    amplitudes_1[i]
                    * np.sin(2 * np.pi * fx * X)
                    * amplitudes_2[j]
                    * np.sin(2 * np.pi * fy * Y)
                )

        signal_max = np.max(np.abs(Z_temp))
        normalized_signal = Z_temp / signal_max

        surf = ax[i].plot_surface(
            X,
            Y,
            normalized_signal,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
    plt.savefig("testing_sine_sum_2d.pdf")
    plt.close()

    # final surface
    for i, fx in enumerate(freqs_1):
        for j, fy in enumerate(freqs_2):
            Z += (
                amplitudes_1[i]
                * np.sin(2 * np.pi * fx * X)
                * amplitudes_2[j]
                * np.sin(2 * np.pi * fy * Y)
            )

    signal_max = np.max(np.abs(Z))
    normalized_signal = Z / signal_max

    fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        X,
        Y,
        normalized_signal,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf)
    fig.savefig("sines_3d.pdf")
    plt.close()
    save_data(
        [X.ravel(), Y.ravel(), Z.ravel()],
        "datasets/sum_of_sines_2d_depth_5.csv",
    )
    print("CREATED!")


def create_1d_data():
    """
    config 1: (depth 5)
        seed: 42
        low=0, high=0.1
        low=0, high=np.pi/9
        n_qubits=1
        depth=5
        x lims: -2*np.pi, 2*np.pi

    config 2: (depth 21)
        seed: 42
        low=0, high=0.1
        low=0, high=np.pi/10
        n_qubits=1
        depth=21
        x lims: -2*np.pi, 2*np.pi


    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    init_func = partial(np.random.uniform, low=0, high=0.1)
    fm_init = partial(np.random.uniform, low=0, high=np.pi / 2)

    model = trainable_freq_circ(
        n_qubits=1,
        depth=5,
        inputs=["x"],
        init_func=init_func,
        fm_init=fm_init,
    )

    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y = (
        model(torch.tensor(x).reshape(-1, 1).to(device))
        .cpu()
        .detach()
        .numpy()
        .flatten()
    )

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.savefig("spectrum_og.pdf")
    plt.close()
    y_hat = np.fft.fft(y)
    x_hat = np.fft.fftfreq(1000, d=(x[1] - x[0]))
    mag = np.abs(y_hat)
    print(f"nyquist: {2*4*np.pi*np.max(x_hat)}")
    plt.plot(x_hat[: len(x_hat) // 2], mag[: len(mag) // 2])
    plt.savefig("spectrum.pdf")
    plt.close()

    # save_data([x, y], "data/student_teacher_1d_depth_21.csv")


def create_2d_data():
    """
    config 1:
        seed: 42
        low=0, high=np.pi/5 (np.pi/depth)
        low=0, high=np.pi/5
        n_qubits=2
        depth=5
        x lims: -2*np.pi, 2*np.pi
    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    depth = 21
    # 1/ depth
    init_func = partial(np.random.uniform, low=0, high=np.pi / 21)
    fm_init = partial(np.random.uniform, low=0, high=np.pi / 21)

    # std = np.sqrt(1/(4*(depth+2)))
    # std = np.sqrt(1/depth)
    # init_func = partial(np.random.normal, loc=0, scale=std)
    # fm_init = partial(np.random.normal, loc=0, scale=std)

    model = trainable_freq_circ(
        n_qubits=2,
        depth=depth,
        inputs=["x", "y"],
        init_func=init_func,
        fm_init=fm_init,
    )

    x = np.arange(-2 * np.pi, 2 * np.pi, 0.05)
    y = np.arange(-2 * np.pi, 2 * np.pi, 0.05)
    X, Y = np.meshgrid(x, y)
    torch_x = np.ravel(X)
    torch_y = np.ravel(Y)
    torch_xy = np.array([torch_x, torch_y]).T
    torch_xy = torch.tensor(torch_xy, dtype=torch.float32).to(device=device)
    zs = model(torch_xy).cpu().detach().numpy().flatten()
    Z = zs.reshape(X.shape)

    fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    # ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.colorbar(surf)
    fig.savefig("datasets/student_teacher_2d_depth_21.pdf")

    print(X.shape, Y.shape, Z.shape)

    save_data(
        [X.ravel(), Y.ravel(), Z.ravel()],
        "datasets/student_teacher_2d_depth_21.csv",
    )


# create_sum_of_sine_1d()
# create_2d_data()
# create_sum_of_sine_1d()
# create_sum_of_sine_2d_v2()
