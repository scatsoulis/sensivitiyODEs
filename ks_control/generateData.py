import numpy as np
from tqdm import tqdm
from jax_models import KuramotoSivashinsky
from jax import random

key = random.key(0)


# all_steps = True/False : data at all/final time(s)
def ks_step_data_collector(KS: KuramotoSivashinsky, num_steps: int, u0: np.ndarray, all_steps: bool = True) -> np.ndarray:
    u_full = np.zeros((num_steps + 1, len(u0)))
    u_full[0] = u0
    for i in range(num_steps):
        u_full[i + 1] = KS.step(u_full[i])

    if all_steps:
        return u_full
    return u_full[-1]


def main() -> None:
    path = "."
    samples = 2
    train_period = 2000

    # Define KS domain and system
    dt = 0.25
    Nx = 256
    L = 2 * np.pi * 16
    modes = 16
    lambda_param = 2
    mu = 0
    noise = 0
    KS = KuramotoSivashinsky(dt=dt, Nx=Nx, L=L, modes=modes, lambda_param=lambda_param, mu=mu, noise=noise)  # Kuramoto-Sivashinsky model

    x = np.arange(0, L, L / Nx)
    data = np.zeros((samples, train_period + 1, len(x)))
    for i in tqdm(range(samples)):
        u0 = u_rand(KS.L)  # Generate ICs
        data[i, :, :] = ks_step_data_collector(KS, train_period, u0(x))  # Generate KS data

    outname = f"{path}/data/KSdata_dt{dt}_Nx{Nx}_L{round(L, 2)}_M{modes}_time{train_period}_samples{samples}_lambda{lambda_param}.npy"
    np.save(outname, data)


def u_rand(L: float) -> callable:
    nwaves = 5
    a = np.arange(nwaves) + 1
    b = np.random.rand(nwaves)
    d = np.random.rand(nwaves)

    xi = np.linspace(0, L, 1000)
    fx = xi * 0
    for i in range(nwaves):
        fx = fx + b[i] * np.sin(2 * a[i] * np.pi * (xi - d[i]) / L)

    mina = np.min(fx)
    maxa = np.max(fx)

    maxx = max(np.abs(mina), maxa)

    def f(x: np.ndarray) -> np.ndarray:
        fx = x * 0
        for i in range(nwaves):
            fx = fx + b[i] * np.sin(2 * a[i] * np.pi * (x - d[i]) / L)

        return fx / maxx

    return f


if __name__ == "__main__":
    main()
