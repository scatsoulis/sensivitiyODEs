"""
Burger's equation with direct boundary control optimization using JAX (with animation)
"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
# JAX doesn't have solve_banded, we'll implement tridiagonal solver
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse
from functools import partial

# Configure JAX
jax.config.update("jax_enable_x64", True)  # Use double precision

def get_device_info(use_gpu=False):
    """
    Display device information and configure JAX device usage.
    """
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    
    if use_gpu:
        # Try to use GPU if available
        gpus = jax.devices('gpu')
        if gpus:
            print(f"Using GPU: {gpus[0]}")
            return gpus[0]
        else:
            print("GPU requested but not available. Using CPU.")
    
    # Use CPU
    print("Using CPU")
    return jax.devices('cpu')[0]

class BurgersEquationControlJAX:
    """
    JAX implementation of Burger's equation with direct boundary control.
    Controls are applied at the left and right boundaries directly.
    Uses an explicit scheme for the advection term and an implicit scheme for the diffusion term.
    """
    
    def __init__(self, Nx=50, Nt=200, L=1.0, T=1.0, nu=0.1, alpha=0.001):
        """Initialize the Burger's equation controller with direct boundary controls."""
        
        # Store parameters
        self.Nx = Nx
        self.Nt = Nt
        self.L = L
        self.T = T
        self.nu = nu  # Viscosity parameter
        self.alpha = alpha  # Control regularization parameter
        
        # Create discretization
        self.dx = L / (Nx - 1)
        self.dt = T / (Nt - 1)
        self.x = jnp.linspace(0, L, Nx)
        self.t = jnp.linspace(0, T, Nt)
        
        # Check CFL-like conditions
        self.cfl_diffusion = self.nu * self.dt / (self.dx**2)
        print(f"Diffusion CFL-like condition: {self.cfl_diffusion:.6f}")
        print(f"Advection CFL depends on solution. dt={self.dt:.4f}, dx={self.dx:.4f}")
        
        # Set up coefficients for implicit scheme
        self.main_diag_coeff = (1 + 2 * self.nu * self.dt / (self.dx**2))
        self.off_diag_coeff = (-self.nu * self.dt / (self.dx**2))
    
    def q(self, x):
        """Target state: q(x) = 2x for x <= 0.5 and q(x) = 2(1-x) for x > 0.5."""
        return jnp.where(x <= 0.5, 2 * x, 2 * (1 - x))
    
    def u0(self, x):
        """Initial condition - zero everywhere."""
        return jnp.zeros_like(x)
    
    @partial(jit, static_argnums=(0,))
    def solve_tridiagonal(self, main_diag, sub_diag, super_diag, rhs):
        """
        Solve tridiagonal system Ax = b using Thomas algorithm.
        JAX-compatible implementation.
        
        Args:
            main_diag: diagonal elements (length n)
            sub_diag: sub-diagonal elements (length n-1) 
            super_diag: super-diagonal elements (length n-1)
            rhs: right-hand side vector (length n)
        """
        n = len(rhs)
        
        if n == 0:
            return jnp.array([])
        if n == 1:
            return rhs / main_diag
            
        # Forward elimination
        def forward_step(carry, i):
            c_mod, d_mod = carry
            factor = sub_diag[i-1] / c_mod[i-1]
            new_c = main_diag[i] - factor * super_diag[i-1]
            new_d = rhs[i] - factor * d_mod[i-1]
            c_mod = c_mod.at[i].set(new_c)
            d_mod = d_mod.at[i].set(new_d)
            return (c_mod, d_mod), None
        
        # Initialize
        c_mod = jnp.zeros(n)
        c_mod = c_mod.at[0].set(main_diag[0])
        d_mod = jnp.zeros(n)
        d_mod = d_mod.at[0].set(rhs[0])
        
        (c_mod, d_mod), _ = jax.lax.scan(forward_step, (c_mod, d_mod), jnp.arange(1, n))
        
        # Back substitution
        def backward_step(carry, i):
            x = carry
            new_x = (d_mod[i] - super_diag[i] * x[i+1]) / c_mod[i]
            x = x.at[i].set(new_x)
            return x, None
        
        # Initialize solution vector
        x = jnp.zeros(n)
        x = x.at[n-1].set(d_mod[n-1] / c_mod[n-1])
        
        x, _ = jax.lax.scan(backward_step, x, jnp.arange(n-2, -1, -1))
        
        return x
    
    @partial(jit, static_argnums=(0,))
    def solve_burgers_equation(self, boundary_controls):
        """
        Solve Burger's equation u_t + u * u_x = nu * u_xx with specified boundary conditions.
        Uses explicit Euler for advection term and implicit Euler for diffusion term.
        """
        boundary_left = boundary_controls[0]  # Shape (Nt,)
        boundary_right = boundary_controls[1] # Shape (Nt,)
        
        # Initialize solution
        initial_u = self.u0(self.x)
        initial_u = initial_u.at[0].set(boundary_left[0])
        initial_u = initial_u.at[-1].set(boundary_right[0])
        
        M = self.Nx - 2  # Number of interior points
        
        if M <= 0:
            # Only boundary points - create time evolution directly
            def body_fun_boundary_only(i, u_array):
                new_row = jnp.array([boundary_left[i], boundary_right[i]])
                return u_array.at[i].set(new_row)
            
            u_initial = jnp.zeros((self.Nt, self.Nx))
            u_initial = u_initial.at[0].set(initial_u)
            return jax.lax.fori_loop(1, self.Nt, body_fun_boundary_only, u_initial)
        
        # System matrices for interior points
        main_diag = jnp.full(M, self.main_diag_coeff)
        off_diag = jnp.full(M-1, self.off_diag_coeff) if M > 1 else jnp.array([])
        
        def body_fun(i, u_array):
            """Time stepping body function."""
            current_u = u_array[i-1]
            
            # Compute advection term for interior points (explicit)
            u_interior = current_u[1:-1]
            u_left = current_u[:-2]  
            u_right = current_u[2:]  
            advection_term = u_interior * (u_right - u_left) / (2 * self.dx)
            
            # Right-hand side for implicit diffusion solve
            rhs_interior = u_interior - self.dt * advection_term
            
            # Apply boundary conditions to RHS
            rhs_interior = rhs_interior.at[0].add(-self.off_diag_coeff * boundary_left[i])
            if M > 1:
                rhs_interior = rhs_interior.at[-1].add(-self.off_diag_coeff * boundary_right[i])
            
            # Solve tridiagonal system
            if M == 1:
                # Special case: single interior point
                u_next_interior = rhs_interior / main_diag
            else:
                u_next_interior = self.solve_tridiagonal(main_diag, off_diag, off_diag, rhs_interior)
            
            # Construct full solution at next time step
            u_next = jnp.concatenate([
                jnp.array([boundary_left[i]]),
                u_next_interior,
                jnp.array([boundary_right[i]])
            ])
            
            return u_array.at[i].set(u_next)
        
        # Initialize solution array
        u_solution = jnp.zeros((self.Nt, self.Nx))
        u_solution = u_solution.at[0].set(initial_u)
        
        # Time stepping loop
        u_solution = jax.lax.fori_loop(1, self.Nt, body_fun, u_solution)
        
        return u_solution
    
    @partial(jit, static_argnums=(0,))
    def objective(self, boundary_controls):
        """
        Compute objective function for boundary control problem.
        """
        boundary_left = boundary_controls[0]
        boundary_right = boundary_controls[1]
        
        # Solve PDE
        u = self.solve_burgers_equation(boundary_controls)
        
        # Terminal cost (L2 norm of error at final time)
        terminal_cost = 0.5 * jnp.sum((u[-1] - self.q(self.x))**2) * self.dx
        
        # Control cost (regularization)
        control_cost = 0.5 * self.alpha * (jnp.sum(boundary_left**2) + jnp.sum(boundary_right**2)) * self.dt
        
        return terminal_cost + control_cost
    
    def plot_iteration_results(self, iteration, boundary_controls, objective_values, total_iterations=100, fs=30):
        """Plot results for a specific iteration of the optimization."""
        lw = 4
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \usepackage{amssymb}'
        
        boundary_left = boundary_controls[0]
        boundary_right = boundary_controls[1]
        
        u_optimal = self.solve_burgers_equation(boundary_controls)
        
        x_np = np.array(self.x)
        t_np = np.array(self.t)
        u_optimal_np = np.array(u_optimal)
        boundary_left_np = np.array(boundary_left)
        boundary_right_np = np.array(boundary_right)
        q_np = np.array(self.q(self.x))
        u0_np = np.array(self.u0(self.x))
        
        u_min, u_max = -2, 2
        controls_min, controls_max = -3, 3  # Changed from -2, 2 to -3, 3
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if len(objective_values) > 0:
            axes[0, 0].plot(objective_values, linewidth=lw)
            axes[0, 0].set_xlabel('Iteration', fontsize=fs)
            axes[0, 0].set_ylabel(r'$\mathcal{L}(\bm{\theta})$', fontsize=fs)
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_ylim([1e-4, 1.0])  # Changed from [0.001, 1.0] to [1e-4, 1.0]
            axes[0, 0].set_xlim([0, total_iterations-1])
            axes[0, 0].tick_params(axis='both', which='major', labelsize=fs)
        
        axes[0, 1].plot(x_np, q_np, 'r-', label=r'Target $q(x)$', linewidth=lw)
        axes[0, 1].plot(x_np, u_optimal_np[-1], 'b--', label=r'Final state $u(x,t_f)$', linewidth=lw)
        axes[0, 1].set_xlabel(r'$x$', fontsize=fs)
        axes[0, 1].set_ylabel(r'$u(x,t_f)$', fontsize=fs)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='both', which='major', labelsize=fs)
        
        axes[1, 0].plot(t_np, boundary_left_np, 'r-', label=r'$\theta_1 \triangleq u(x=0,t)$', linewidth=lw)
        axes[1, 0].plot(t_np, boundary_right_np, 'b-', label=r'$\theta_2 \triangleq u(x=L,t)$', linewidth=lw)
        axes[1, 0].set_xlabel(r'$t$', fontsize=fs)
        axes[1, 0].set_ylabel(r'$\theta$', fontsize=fs)
        axes[1, 0].set_ylim([controls_min, controls_max])
        axes[1, 0].tick_params(axis='both', which='major', labelsize=fs)
        
        im = axes[1, 1].imshow(
            u_optimal_np, 
            aspect='auto', 
            origin='lower',
            extent=[0, self.L, 0, self.T],
            cmap='plasma',
            vmin=u_min,
            vmax=u_max
        )
        axes[1, 1].set_xlabel(r'$x$', fontsize=fs)
        axes[1, 1].set_ylabel(r'$t$', fontsize=fs)
        cbar = plt.colorbar(im, ax=axes[1, 1], label=r'$u(x,t)$')
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label(r'$u(x,t)$', size=fs)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=fs)
        
        plt.suptitle(f'Burger\'s Equation Optimization (JAX) - Iteration {iteration}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'images/burgers_jax_iteration_{iteration:04d}.png', dpi=300)
        plt.close(fig)
    
    def create_animation(self, objective_values):
        """Create an animation from saved iteration results."""
        try:
            import imageio.v2 as imageio  # Use v2 to avoid deprecation warning
            
            # Check if we have the necessary files
            image_files = sorted([f for f in os.listdir('images') if f.startswith('burgers_jax_iteration_') and f.endswith('.png')])
            
            if not image_files:
                print("No image files found for animation. Skipping animation creation.")
                return
                
            print(f"Creating animation from {len(image_files)} frames...")
            
            # Create gif animation
            with imageio.get_writer('images/burgers_jax_optimization.gif', mode='I', duration=0.2) as writer:
                for filename in image_files:
                    image = imageio.imread(os.path.join('images', filename))
                    writer.append_data(image)
                    
            print("Animation saved to images/burgers_jax_optimization.gif")
            
            # Try to create MP4 with better error handling
            try:
                # Check if ffmpeg backend is available
                try:
                    import imageio_ffmpeg
                    ffmpeg_available = True
                except ImportError:
                    ffmpeg_available = False
                
                if ffmpeg_available:
                    with imageio.get_writer('images/burgers_jax_optimization.mp4', fps=5) as writer:
                        for filename in image_files:
                            image = imageio.imread(os.path.join('images', filename))
                            writer.append_data(image)
                    print("MP4 video saved to images/burgers_jax_optimization.mp4")
                else:
                    print("MP4 creation skipped: ffmpeg backend not available.")
                    print("To enable MP4 creation, install: pip install imageio[ffmpeg]")
                    
            except Exception as e:
                print(f"Could not create MP4 video: {e}")
                print("To enable MP4 creation, try installing: pip install imageio[ffmpeg]")
                
        except ImportError:
            print("imageio module not found. Install with 'pip install imageio' to create animations.")
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    def optimize_control(self, num_iterations=100, learning_rate=0.01, save_animation=True):
        """Optimize the direct boundary controls using optax."""
        
        # Initialize controls
        key = jax.random.PRNGKey(0)
        boundary_controls = jnp.zeros((2, self.Nt))

        # Set up optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(boundary_controls)
        
        # Create gradient function
        grad_fn = jit(grad(self.objective))
        
        objective_values = []
        saved_boundary_controls = []
        
        # Create images directory if saving animation
        if save_animation:
            os.makedirs('images', exist_ok=True)
        
        print("\nStarting optimization...")
        start_time = time.time()
        
        for i in range(num_iterations):
            # Compute objective and gradient
            obj_val = self.objective(boundary_controls)
            grads = grad_fn(boundary_controls)
            
            objective_values.append(float(obj_val))
            
            # Save current boundary controls for animation
            if save_animation:
                saved_boundary_controls.append(boundary_controls)
                # Generate and save plots for this iteration
                self.plot_iteration_results(
                    iteration=i, 
                    boundary_controls=boundary_controls, 
                    objective_values=objective_values,
                    total_iterations=num_iterations
                )
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i}, Objective: {obj_val:.6f}, Elapsed: {elapsed:.2f}s")
            
            if jnp.isnan(obj_val):
                print(f"NaN detected at iteration {i}, stopping optimization")
                break
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            boundary_controls = optax.apply_updates(boundary_controls, updates)
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f} seconds")
        print(f"Final objective value: {self.objective(boundary_controls):.6f}")
        
        # Create animation after optimization
        if save_animation:
            self.create_animation(objective_values)
        
        return boundary_controls, objective_values
    
    def plot_results(self, boundary_controls, objective_values=None, fs=30):
        """Plot key results from the optimization."""
        # Enable LaTeX for text rendering
        lw = 4
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \usepackage{amssymb}'

        boundary_left = boundary_controls[0]
        boundary_right = boundary_controls[1]
        
        # Solve for the optimized state
        u_optimal = self.solve_burgers_equation(boundary_controls)
        
        # Convert to numpy for plotting
        x_np = np.array(self.x)
        t_np = np.array(self.t)
        u_optimal_np = np.array(u_optimal)
        boundary_left_np = np.array(boundary_left)
        boundary_right_np = np.array(boundary_right)
        q_np = np.array(self.q(self.x))
        u0_np = np.array(self.u0(self.x))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Optimization progress
        if objective_values is not None:
            axes[0, 0].plot(objective_values,linewidth=lw)
            axes[0, 0].set_xlabel('Iteration', fontsize=fs)
            axes[0, 0].set_ylabel(r'$\mathcal{L}(\bm{\theta})$', fontsize=fs)
            axes[0, 0].tick_params(axis='both', which='major', labelsize=fs)
        
        # Target vs achieved state
        axes[0, 1].plot(x_np, q_np, 'r-', label=r'Target $q(x)$',linewidth=lw)
        axes[0, 1].plot(x_np, u_optimal_np[-1], 'b--', label=r'Final state $u(x,t_f)$',linewidth=lw)
        axes[0, 1].set_xlabel(r'$x$', fontsize=fs)
        axes[0, 1].set_ylabel(r'$u(x,t_f)$', fontsize=fs)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=fs)
        
        # Boundary controls
        axes[1, 0].plot(t_np, boundary_left_np, 'r-', label=r'$\theta_1 \triangleq u(x=0,t)$',linewidth=lw)
        axes[1, 0].plot(t_np, boundary_right_np, 'b-', label=r'$\theta_2 \triangleq u(x=L,t)$',linewidth=lw)
        axes[1, 0].set_xlabel(r'$t$', fontsize=fs)
        axes[1, 0].set_ylabel(r'$\theta$', fontsize=fs)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=fs)
        
        # State evolution heatmap
        im = axes[1, 1].imshow(
            u_optimal_np, 
            aspect='auto', 
            origin='lower',
            extent=[0, self.L, 0, self.T],
            cmap='plasma'
        )
        axes[1, 1].set_xlabel(r'$x$', fontsize=fs)
        axes[1, 1].set_ylabel(r'$t$', fontsize=fs)
        cbar = plt.colorbar(im, ax=axes[1, 1], label=r'$u(x,t)$')
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label(r'$u(x,t)$', size=fs)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=fs)
        
        plt.tight_layout()
        plt.savefig('burgers_equation_jax_control.png', dpi=300)
        
        # Performance metrics
        l2_error = float(jnp.sqrt(jnp.sum((u_optimal[-1] - self.q(self.x))**2) * self.dx))
        control_energy = float(0.5 * self.alpha * (jnp.sum(boundary_left**2) + 
                                                   jnp.sum(boundary_right**2)) * self.dt)
        
        print("\nPerformance Summary:")
        print(f"L2 error at final time: {l2_error:.6f}")
        print(f"Control energy: {control_energy:.6f}")
        print(f"Number of control parameters: 2 × {self.Nt} = {2 * self.Nt}")
        
        # Time snapshots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(x_np, u_optimal_np[0], 'b-')
        axes[0].set_title('Initial State (t=0)', fontsize=fs)
        axes[0].set_xlabel('x', fontsize=fs)
        axes[0].set_ylabel('u(x,0)', fontsize=fs)
        axes[0].tick_params(axis='both', which='major', labelsize=fs)
        axes[0].grid(True)
        
        mid_idx = self.Nt // 2
        axes[1].plot(x_np, u_optimal_np[mid_idx], 'g-')
        axes[1].set_title(f'Mid State (t={t_np[mid_idx]:.2f})', fontsize=fs)
        axes[1].set_xlabel('x', fontsize=fs)
        axes[1].set_ylabel(f'u(x,{t_np[mid_idx]:.2f})', fontsize=fs)
        axes[1].tick_params(axis='both', which='major', labelsize=fs)
        axes[1].grid(True)
        
        axes[2].plot(x_np, q_np, 'r--', label='Target')
        axes[2].plot(x_np, u_optimal_np[-1], 'b-', label='Final')
        axes[2].set_title(f'Final State (t={self.T})', fontsize=fs)
        axes[2].set_xlabel('x', fontsize=fs)
        axes[2].set_ylabel(f'u(x,{self.T})', fontsize=fs)
        axes[2].legend(fontsize=fs)
        axes[2].tick_params(axis='both', which='major', labelsize=fs)
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('burgers_equation_jax_snapshots.png', dpi=300)
        
        return u_optimal, boundary_left, boundary_right

def main():
    """Main function to run the optimization and display results."""
    parser = argparse.ArgumentParser(description='Burger\'s Equation with Boundary Control (JAX)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for computation if available')
    parser.add_argument('--nx', type=int, default=500, help='Number of spatial points')
    parser.add_argument('--nt', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity parameter')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of optimization iterations')
    parser.add_argument('--learning_rate', type=float, default=0.25, help='Learning rate for optimization')
    parser.add_argument('--no_animation', action='store_true', help='Skip animation generation to speed up')
    args = parser.parse_args()
    
    # Configure device
    device = get_device_info(use_gpu=args.use_gpu)
    
    # Create controller
    controller = BurgersEquationControlJAX(
        Nx=args.nx,           
        Nt=args.nt,          
        L=1.0,           
        T=1.0,           
        nu=args.nu,      
        alpha=0.001     
    )
    
    print(f"\nProblem setup:")
    print(f"Grid: {args.nx} × {args.nt}")
    print(f"Viscosity: {args.nu}")
    print(f"Optimization: {args.iterations} iterations at lr={args.learning_rate}")
    if not args.no_animation:
        print("Animation: Enabled (use --no_animation to disable)")
    else:
        print("Animation: Disabled")
    
    # Run optimization
    boundary_controls, objective_values = controller.optimize_control(
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        save_animation=not args.no_animation
    )
    
    # Plot results
    controller.plot_results(boundary_controls, objective_values)
    
    print("\nPlots saved: burgers_equation_jax_control.png, burgers_equation_jax_snapshots.png")
    if not args.no_animation:
        print("Animation files saved in 'images/' directory")
    
    # Also print device info for verification
    print(f"Computation completed on: {jax.devices()[0]}")

if __name__ == "__main__":
    main()
