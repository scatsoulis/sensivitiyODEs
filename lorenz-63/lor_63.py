#!/usr/bin/env python3
"""Neural ODE forecasting for Lorenz-63 system with adjoint method and forecasting metrics"""

import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Lorenz-63 Neural ODE")
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

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

# Lorenz-63 parameters
N = 3  # Number of variables
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
true_y0 = torch.tensor([[1.0, 1.0, 1.0]]).to(device)  # Initial condition
t = torch.linspace(0.0, 25.0, args.data_size).to(device)


# True Lorenz-63 dynamics
class Lorenz63(nn.Module):
    def __init__(self, sigma, rho, beta):
        super(Lorenz63, self).__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def forward(self, t, y):
        # y shape: [batch_size, 1, N] or [1, N]
        y = y.squeeze(1)  # Shape: [batch_size, N] or [N]
        x, y_var, z = y[:, 0], y[:, 1], y[:, 2]
        dx_dt = self.sigma * (y_var - x)
        dy_dt = x * (self.rho - z) - y_var
        dz_dt = x * y_var - self.beta * z
        dydt = torch.stack([dx_dt, dy_dt, dz_dt], dim=1)
        return dydt.unsqueeze(1)  # Shape: [batch_size, 1, N]


# Generate true trajectory
with torch.no_grad():
    true_model = Lorenz63(sigma, rho, beta).to(device)
    true_y = odeint(true_model, true_y0, t, method="dopri5")  # Shape: [data_size, 1, N]


# Batch sampling
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # [batch_size, 1, N]
    batch_t = t[: args.batch_time]  # [batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # [batch_time, batch_size, 1, N]
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# Forecasting metrics
def compute_msfe(true_y, pred_y):
    # Mean squared forecast error over state dimensions
    mse = np.mean((true_y - pred_y) ** 2, axis=1)  # Shape: [time]
    return mse


def compute_vpt(true_y, pred_y, t, threshold_factor=0.1):
    # Valid prediction time: time until error exceeds threshold
    true_std = np.std(true_y, axis=0)  # Shape: [N]
    threshold = threshold_factor * np.mean(true_std)  # Scalar
    errors = np.sqrt(np.mean((true_y - pred_y) ** 2, axis=1))  # Shape: [time]
    vpt_idx = np.where(errors > threshold)[0]
    return t[vpt_idx[0]].cpu().numpy() if vpt_idx.size > 0 else t[-1].cpu().numpy()


def compute_lyapunov_exponent(model, y0, t, pert=1e-5):
    y0_pert = y0 + pert
    traj = odeint(model, y0, t).cpu().numpy()[:, 0, :]
    traj_pert = odeint(model, y0_pert, t).cpu().numpy()[:, 0, :]
    dist = np.linalg.norm(traj - traj_pert, axis=1)
    # Compute Lyapunov exponent as ln(final_dist / initial_dist) / time
    return np.log(dist[-1] / dist[0]) / (t[-1].cpu().numpy() - t[0].cpu().numpy())


# Visualization
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs("png")
    fig = plt.figure(figsize=(12, 4), facecolor="white")
    ax_traj = fig.add_subplot(131)
    ax_phase = fig.add_subplot(132, projection="3d")
    ax_stats = fig.add_subplot(133)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr, t):
    if args.viz:
        # Trajectories
        ax_traj.cla()
        ax_traj.set_title("Trajectories")
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("x, y, z")
        labels = ["x", "y", "z"]
        true_traj = true_y.cpu().numpy()[:, 0, :]  # Shape: [time, N]
        pred_traj = pred_y.cpu().numpy()[:, 0, :]  # Shape: [time, N]
        for i in range(N):
            ax_traj.plot(t.cpu().numpy(), true_traj[:, i], label=f"Truth {labels[i]}")
            ax_traj.plot(t.cpu().numpy(), pred_traj[:, i], "--", label=f"Pred {labels[i]}")
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.legend()

        # 3D phase portrait
        ax_phase.cla()
        ax_phase.set_title("Phase Portrait")
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("y")
        ax_phase.set_zlabel("z")
        ax_phase.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], "g-")
        ax_phase.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], "b--")
        ax_phase.set_xlim(-20, 20)
        ax_phase.set_ylim(-30, 30)
        ax_phase.set_zlim(0, 50)

        # Forecasting metrics
        ax_stats.cla()
        ax_stats.set_title("Forecast Metrics")
        ax_stats.set_xlabel("t")
        ax_stats.set_ylabel("MSFE")
        msfe = compute_msfe(true_traj, pred_traj)
        ax_stats.semilogy(t.cpu().numpy(), msfe, "b-", label="MSFE")
        vpt = compute_vpt(true_traj, pred_traj, t)
        ax_stats.axvline(x=vpt, color="r", linestyle="--", label=f"VPT = {vpt:.2f}")
        lyap_true = compute_lyapunov_exponent(true_model, true_y0, t)
        lyap_pred = compute_lyapunov_exponent(odefunc, true_y0, t)
        ax_stats.text(0.05, 0.95, f"lambda_true = {lyap_true:.3f}\n lambda_pred = {lyap_pred:.3f}", transform=ax_stats.transAxes, fontsize=10)
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
            nn.Linear(N, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, N),
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
                visualize(true_y, pred_y, func, ii, t)
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
        ax.plot(t_all, true_all, "g-", label="Truth x")
        ax.plot(t_all, pred_all, "b--", label="Forecast x")
        ax.axvline(x=25.0, color="k", linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.legend()
        plt.savefig("png/forecast.png")
        plt.show()
    import subprocess

    makedirs("data")
    pred_traj = pred_y.cpu().numpy()[:, 0, :]
    true_traj = true_y.cpu().numpy()[:, 0, :]
    for i in range(0, len(t), 10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pred_traj[: i + 1, 0], pred_traj[: i + 1, 1], pred_traj[: i + 1, 2], "b--")
        ax.plot(true_traj[: i + 1, 0], true_traj[: i + 1, 1], true_traj[: i + 1, 2], "g-")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-30, 30)
        ax.set_zlim(0, 50)
        plt.savefig(f"data/{i // 10:04d}.png")
        plt.close()
    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", "data/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "phase_portrait.mp4"]
    subprocess.run(cmd, check=True)

    error = np.linalg.norm(true_traj - pred_traj, axis=1)
    ax_stats.plot(t.cpu().numpy(), error, "k-", label="Error Norm")
    ax_stats.set_ylabel("Error / MSFE")
    ax_stats.legend()

    with torch.no_grad():
        initial_conditions = [true_y[-1] + torch.randn(1, N).to(device) * 0.1 for _ in range(5)]
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        for i, y0 in enumerate(initial_conditions):
            pred_y_ic = odeint(func, y0, t_forecast)
            true_y_ic = odeint(true_model, y0, t_forecast)
            ax.plot(t_forecast.cpu().numpy(), pred_y_ic.cpu().numpy()[:, 0, 0], "b--", alpha=0.5, label=f"Pred IC{i + 1}" if i == 0 else None)
            ax.plot(t_forecast.cpu().numpy(), true_y_ic.cpu().numpy()[:, 0, 0], "g-", alpha=0.5, label=f"True IC{i + 1}" if i == 0 else None)
        ax.axvline(x=25.0, color="k", linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.legend()
        plt.savefig("png/forecast_multiple_ic.png")
