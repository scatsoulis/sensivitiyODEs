#!/usr/bin/env python3
"""Neural ODE forecasting for Lorenz-96 system with adjoint method"""

import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from scipy.fft import fft

parser = argparse.ArgumentParser("Lorenz-96 Neural ODE")
parser.add_argument("--method", type=str, choices=["dopri5", "adams"], default="dopri5")
parser.add_argument("--data_size", type=int, default=1000)
parser.add_argument("--batch_time", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--test_freq", type=int, default=20)
parser.add_argument("--viz", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--adjoint", action="store_true")
args = parser.parse_args()

# Plotting parameters
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 24,
        "legend.fontsize": 8,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 2,
        "text.usetex": True,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
        "legend.loc": "upper right",
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
    }
)

from torchdiffeq import odeint_adjoint as odeint

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

# Lorenz-96 parameters
N = 40  # Number of variables
F = 8.0  # Forcing constant
true_y0 = torch.randn(1, N).to(device) * 0.1  # Random initial condition
t = torch.linspace(0.0, 25.0, args.data_size).to(device)


# True Lorenz-96 dynamics
class Lorenz96(nn.Module):
    def __init__(self, F):
        super(Lorenz96, self).__init__()
        self.F = F

    def forward(self, t, y):
        # y shape: [batch_size, 1, N] or [1, N]
        y = y.squeeze(1)  # Shape: [batch_size, N] or [N]
        # Periodic indices
        i = torch.arange(N, device=device)
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        # dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
        dydt = (y[:, ip1] - y[:, im2]) * y[:, im1] - y + self.F
        return dydt.unsqueeze(1)  # Shape: [batch_size, 1, N]


# Generate true trajectory
with torch.no_grad():
    true_model = Lorenz96(F=F).to(device)
    true_y = odeint(true_model, true_y0, t, method="dopri5")  # Shape: [data_size, 1, N]


# Batch sampling
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # [batch_size, 1, N]
    batch_t = t[: args.batch_time]  # [batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # [batch_time, batch_size, 1, N]
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# Visualization
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs("png")
    fig = plt.figure(figsize=(12, 4), facecolor="white")
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_stats = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def compute_spatial_correlation(x, N):
    r = np.arange(N // 2)
    R = np.zeros(N // 2)
    for i in range(N // 2):
        x_shifted = np.roll(x, -i)
        R[i] = np.mean(x * x_shifted)
    return r, R / R[0]


def compute_energy_spectrum(x, N):
    x_hat = fft(x)
    E_k = 0.5 * np.abs(x_hat[: N // 2]) ** 2
    k = np.arange(N // 2)
    return k, E_k


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        # Trajectories
        ax_traj.cla()
        ax_traj.set_title("Trajectories")
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("x_i")
        for i in range(min(3, N)):
            ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, i], label=f"Truth x_{i + 1}")
            ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, i], "--", label=f"Pred x_{i + 1}")
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.legend()

        # Phase portrait
        ax_phase.cla()
        ax_phase.set_title("Phase Portrait")
        ax_phase.set_xlabel("x_1")
        ax_phase.set_ylabel("x_2")
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], "g-")
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], "b--")
        ax_phase.set_xlim(-10, 10)
        ax_phase.set_ylim(-10, 10)

        # Spatial correlation and spectrum
        ax_stats.cla()
        ax_stats.set_title("Spatial Statistics")
        ax_stats.set_xlabel("r / k")
        ax_stats.set_ylabel("R(r) / E(k)")
        true_x = true_y.cpu().numpy()[-1, 0, :]
        pred_x = pred_y.cpu().numpy()[-1, 0, :]
        r, R_true = compute_spatial_correlation(true_x, N)
        _, R_pred = compute_spatial_correlation(pred_x, N)
        k, E_k_true = compute_energy_spectrum(true_x, N)
        _, E_k_pred = compute_energy_spectrum(pred_x, N)
        ax_stats.plot(r, R_true, "g-", label="Truth R(r)")
        ax_stats.plot(r, R_pred, "b--", label="Pred R(r)")
        ax_stats.semilogy(k[1:], E_k_true[1:], "g:", label="Truth E(k)")
        ax_stats.semilogy(k[1:], E_k_pred[1:], "b-.", label="Pred E(k)")
        ax_stats.legend()

        fig.tight_layout()
        plt.savefig(f"png/{itr:03d}.png")
        plt.draw()
        plt.pause(0.001)


# Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, N),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # y shape: [batch_size, 1, N] or [1, N]
        y = y.squeeze(1)  # Shape: [batch_size, N] or [N]
        out = self.net(y)  # Shape: [batch_size, N] or [N]
        return out.unsqueeze(1)  # Shape: [batch_size, 1, N] or [1, N]


# Running average meter
class RunningAverageMeter:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# Training and forecasting
if __name__ == "__main__":
    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    end = time.time()
    ii = 0

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, method=args.method).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method=args.method)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print(f"Iter {itr:04d} | Total Loss {loss.item():.6f}")
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

    # Forecast
    with torch.no_grad():
        t_forecast = torch.linspace(25.0, 50.0, args.data_size).to(device)
        pred_y_forecast = odeint(func, true_y[-1], t_forecast, method=args.method)
        true_y_forecast = odeint(true_model, true_y[-1], t_forecast, method=args.method)

    # Plot forecast
    if args.viz:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        t_all = torch.cat([t, t_forecast]).cpu().numpy()
        true_all = torch.cat([true_y, true_y_forecast]).cpu().numpy()[:, 0, 0]
        pred_all = torch.cat([pred_y, pred_y_forecast]).cpu().numpy()[:, 0, 0]
        ax.plot(t_all, true_all, "g-", label="Truth x_1")
        ax.plot(t_all, pred_all, "b--", label="Forecast x_1")
        ax.axvline(x=25.0, color="k", linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("x_1")
        ax.legend()
        plt.savefig("png/forecast.png")
        plt.show()
