import jax.numpy as jnp
from jax import jit
from jax import random
# ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50)

key = random.key(0)


@jit
def kuramoto_sivashinsky_step(u: jnp.ndarray, dt: float, E: jnp.ndarray, E2: jnp.ndarray, Q: jnp.ndarray, f1: jnp.ndarray, f2: jnp.ndarray, f3: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    u_fft = jnp.fft.fft(u)

    Nv = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(u_fft)) ** 2)
    a = E2 * u_fft + Q * Nv

    Na = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(a)) ** 2)
    b = E2 * u_fft + Q * Na

    Nb = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(b)) ** 2)
    c = E2 * a + Q * (2 * Nb - Nv)

    Nc = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(c)) ** 2)

    u_fft_next = E * u_fft + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3  #  + E_forced*mu * jnp.fft.fft(yobs - v)
    u_next = jnp.real(jnp.fft.ifft(u_fft_next))

    # Observation
    # y_obs = u_next + random.multivariate_normal(key, jnp.zeros(len(u_next)), jnp.eye(len(u_next)) * noise * len(u_next))  # Add noise to the next state

    return u_next


# Models as classes
class BaseModel:
    def __init__(self, dt: float = 0.01) -> None:
        self.dt = dt

    def step(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("The step method must be implemented by subclasses.")


class KuramotoSivashinsky(BaseModel):
    def __init__(self, dt: float, Nx: int, L: int, modes: int, lambda_param: float, mu: float, noise: float) -> None:
        super().__init__(dt)
        self.Nx, self.L, self.modes, self.lambda_param = (Nx, L, modes, lambda_param)  # discr. points, domain length, exponential time differencing points (modes)
        self.k, self.E, self.E2, self.Q, self.f1, self.f2, self.f3, self.g = self.precompute_constants()

    def precompute_constants(self) -> tuple:
        k = (2 * jnp.pi / self.L) * jnp.concatenate(
            [
                jnp.arange(0, self.Nx // 2),
                jnp.array([0]),
                jnp.arange(-self.Nx // 2 + 1, 0),
            ]
        )
        Lk = self.lambda_param * k**2 - k**4
        E = jnp.exp(self.dt * Lk)
        E2 = jnp.exp(self.dt * Lk / 2)
        r = jnp.exp(1j * jnp.pi * (jnp.arange(1, self.modes + 1) - 0.5) / self.modes)
        LR = self.dt * jnp.tile(Lk, (self.modes, 1)).T + jnp.tile(r, (self.Nx, 1))
        Q = self.dt * jnp.real(jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=1))

        f1 = self.dt * jnp.real(jnp.mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
        f2 = self.dt * jnp.real(jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR**3, axis=1))
        f3 = self.dt * jnp.real(jnp.mean((-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=1))
        g = -0.5j * k

        return k, E, E2, Q, f1, f2, f3, g

    def step(self, u: jnp.ndarray) -> jnp.ndarray:
        return kuramoto_sivashinsky_step(u, self.dt, self.E, self.E2, self.Q, self.f1, self.f2, self.f3, self.g)
