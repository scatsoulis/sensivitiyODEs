"""
Burger's equation with direct boundary control optimization using PyTorch (CPU/GPU version)
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import argparse

# Set environment variables for multi-threading when using CPU
cores_to_use = 10
os.environ["OMP_NUM_THREADS"] = str(cores_to_use)
os.environ["MKL_NUM_THREADS"] = str(cores_to_use)

def get_device(use_gpu=False, gpu_id=0):
    """
    Determine the device to use based on user preferences and availability.
    
    Parameters:
    -----------
    use_gpu : bool
        If True, attempt to use GPU; otherwise, use CPU
    gpu_id : int
        The ID of the GPU to use if multiple are available
        
    Returns:
    --------
    device : torch.device
        The device to use for computations
    """
    if use_gpu and torch.cuda.is_available():
        # Check if the requested GPU ID is valid
        if gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
            gpu_id = 0
            
        # Make sure we get properties before setting CUDA_VISIBLE_DEVICES which can change device indexing
        try:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
        except (AssertionError, RuntimeError) as e:
            print(f"Warning: Error getting properties for GPU {gpu_id}: {e}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
            # Try another approach - just report what we can
            gpu_name = f"CUDA Device {gpu_id}"
            gpu_mem = 0
            
        # Now set device and environment variable
        device = torch.device(f'cuda:{gpu_id}')
        # Note: Setting CUDA_VISIBLE_DEVICES may not be needed and could cause issues
        # It's often better to let PyTorch manage device selection
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"Using GPU {gpu_id}: {gpu_name}")
        if gpu_mem > 0:
            print(f"GPU Memory: {gpu_mem:.2f} GB")
        return device
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but not available. Using CPU instead.")
        # Force PyTorch to use CPU only if explicitly requested
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
        # Set PyTorch thread count for CPU
        torch.set_num_threads(cores_to_use)
        
        device = torch.device('cpu')
        print(f"Using CPU with {cores_to_use} threads")
        print(f"PyTorch thread settings: {torch.get_num_threads()} threads")
        return device

# Print PyTorch version info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

class BurgersEquationControl:
    """
    PyTorch implementation of Burger's equation with direct boundary control.
    Controls are applied at the left and right boundaries directly.
    Uses an explicit scheme for the advection term and an implicit scheme for the diffusion term.
    """
    
    def __init__(self, Nx=50, Nt=200, L=1.0, T=1.0, nu=0.1, alpha=0.001, device=None):
        """Initialize the Burger's equation controller with direct boundary controls."""
        self.device = device if device is not None else torch.device('cpu')
        
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
        self.x = torch.linspace(0, L, Nx, device=self.device)
        self.t = torch.linspace(0, T, Nt, device=self.device)
        
        # Check CFL-like conditions
        self.cfl_diffusion = self.nu * self.dt / (self.dx**2)
        print(f"Diffusion CFL-like condition (for implicit scheme, explicit would be <= 0.5): {self.cfl_diffusion:.6f}")
        print(f"Advection CFL (max|u|*dt/dx <= 1) depends on solution. dt={self.dt:.4f}, dx={self.dx:.4f}")
        
        # Set up matrices for implicit scheme for diffusion
        self._setup_matrices()
        
        # Initialize boundary controls as parameters
        self.boundary_controls = torch.zeros((2, Nt), device=self.device, requires_grad=True)
    
    def _setup_matrices(self):
        """Set up scalar coefficients for the implicit solver for the diffusion term."""
        self.main_diag_coeff = (1 + 2 * self.nu * self.dt / (self.dx**2))
        self.off_diag_coeff = (-self.nu * self.dt / (self.dx**2))
    
    def q(self, x):
        """Target state: q(x) = 2x for x <= 0.5 and q(x) = 2(1-x) for x > 0.5."""
        res = torch.zeros_like(x)
        cond1 = x <= 0.5
        cond2 = x > 0.5
        res[cond1] = 2 * x[cond1]
        res[cond2] = 2 * (1 - x[cond2])
        return res
    
    def u0(self, x):
        """Initial condition - zero everywhere."""
        return torch.zeros_like(x)
    
    # We'll use PyTorch's built-in tridiagonal solver instead of implementing our own
    # This method is kept for backwards compatibility but will not be used
    
    def solve_tridiagonal(self, a, b, c, d): # a=main (M), b=sub (M-1), c=super (M-1), d=rhs (M)
        n = a.size(0) # n = M

        a_mod = a.clone() # Cloned, modified in-place, but 'a' itself doesn't require grad from controls.
        
        d_mod_list = [d[0]] 
        for i in range(1, n):
            m = b[i-1] / a_mod[i-1] 
            a_mod[i] = a_mod[i] - m * c[i-1] 
            new_d_i = d[i] - m * d_mod_list[i-1]
            d_mod_list.append(new_d_i)
        d_mod = torch.stack(d_mod_list)

        # Back substitution - build x out-of-place
        x_list = [torch.empty_like(d[0]) for _ in range(n)] 

        if n > 0:
            x_list[n-1] = d_mod[n-1] / a_mod[n-1]
            for i in range(n-2, -1, -1):
                # Ensure c[i] and x_list[i+1] are compatible for multiplication
                # x_list[i+1] is a 0-dim tensor (scalar). c[i] is also a scalar from a 1D tensor.
                term = c[i] * x_list[i+1]
                x_list[i] = (d_mod[i] - term) / a_mod[i]
            return torch.stack(x_list)
        else: 
            return torch.empty(0, device=d.device, dtype=d.dtype)
    
    def solve_burgers_equation(self, boundary_controls=None):
        """
        Solve Burger's equation u_t + u * u_x = nu * u_xx with specified boundary conditions.
        Uses explicit Euler for advection term (central difference) and implicit Euler for diffusion term.
        Builds the solution tensor u out-of-place to avoid autograd issues.
        """
        if boundary_controls is None:
            boundary_controls = self.boundary_controls
            
        boundary_left = boundary_controls[0]  # Shape (Nt,)
        boundary_right = boundary_controls[1] # Shape (Nt,)
        
        initial_u_row = self.u0(self.x).clone()
        initial_u_row[0] = boundary_left[0]       
        initial_u_row[-1] = boundary_right[0]     
        u_rows = [initial_u_row]
        
        M = self.Nx - 2 
        
        if M < 0:
             raise ValueError("Nx must be at least 2 for boundary conditions.")
        
        if M == 0: # Nx=2, only boundary points
            for i_time_step in range(self.Nt - 1):
                # Construct the row for the next time step
                # boundary_left[i_time_step+1] is a scalar tensor
                # boundary_right[i_time_step+1] is a scalar tensor
                # We need a 1D tensor of shape (Nx,) which is (2,) here.
                next_u_row_M0 = torch.stack([boundary_left[i_time_step+1], boundary_right[i_time_step+1]])
                u_rows.append(next_u_row_M0)
            return torch.stack(u_rows)

        # Pre-allocate system matrices
        a_sys = torch.full((M,), self.main_diag_coeff, device=self.device)
        b_sys = torch.full((M - 1,), self.off_diag_coeff, device=self.device)
        c_sys = torch.full((M - 1,), self.off_diag_coeff, device=self.device)
        
        for i_time_step in range(self.Nt - 1):
            current_u_row = u_rows[-1]
            
            # VECTORIZED: Calculate advection term for all interior points at once
            u_interior = current_u_row[1:-1]
            u_left = current_u_row[:-2]
            u_right = current_u_row[2:]
            advection_term = u_interior * (u_right - u_left) / (2 * self.dx)
            rhs_interior = u_interior - self.dt * advection_term
            
            # Adjust boundary conditions
            bc_left_term = self.off_diag_coeff * boundary_left[i_time_step+1]
            rhs_interior[0] = rhs_interior[0] - bc_left_term
            if M > 1:
                bc_right_term = self.off_diag_coeff * boundary_right[i_time_step+1]
                rhs_interior[-1] = rhs_interior[-1] - bc_right_term
            
            # Use your original solver
            u_next_interior = self.solve_tridiagonal(a_sys.clone(), b_sys.clone(), c_sys.clone(), rhs_interior)
            
            # Construct the full next row u_i_plus_1_row out-of-place
            # Create a list of parts and then concatenate
            parts_for_next_row = [
                boundary_left[i_time_step+1].unsqueeze(0), # Left BC (scalar to 1-element tensor)
                u_next_interior,                           # Interior points (1D tensor)
                boundary_right[i_time_step+1].unsqueeze(0) # Right BC (scalar to 1-element tensor)
            ]
            u_i_plus_1_row = torch.cat(parts_for_next_row)
            
            u_rows.append(u_i_plus_1_row)
            
        return torch.stack(u_rows)

    def objective(self, boundary_controls=None):
        """
        Compute objective function for boundary control problem (Burger's equation).
        """
        if boundary_controls is None:
            boundary_controls = self.boundary_controls
            
        boundary_left = boundary_controls[0]
        boundary_right = boundary_controls[1]
        
        u = self.solve_burgers_equation(boundary_controls)
        
        q_target_norm_sq = torch.sum(self.q(self.x)**2) * self.dx
        epsilon = 1e-9
        #terminal_cost = (0.5 * torch.sum((u[-1] - self.q(self.x))**2) * self.dx) / (q_target_norm_sq + epsilon)
        terminal_cost = 0.5 * torch.sum((u[-1] - self.q(self.x))**2) * self.dx
        control_cost = 0.5 * self.alpha * (torch.sum(boundary_left**2) + torch.sum(boundary_right**2)) * self.dt
        
        return terminal_cost + control_cost
    
    def optimize_control(self, num_iterations=100, learning_rate=0.01):
        """Optimize the direct boundary controls using PyTorch optimizers, saving state at each iteration."""
        # Initialize boundary controls with zeros for both boundaries
        
        # Create initial values: both boundaries with zeros
        left_boundary_init = torch.zeros_like(self.t)
        right_boundary_init = torch.zeros_like(self.t)
        
        # Stack together to form boundary_controls with shape (2, Nt)
        self.boundary_controls = torch.stack([left_boundary_init, right_boundary_init]).to(self.device)
        self.boundary_controls.requires_grad_(True)
        
        optimizer = optim.Adam([self.boundary_controls], lr=learning_rate)
        
        objective_values = []
        # Save boundary controls at each iteration
        saved_boundary_controls = []
        
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        print("\nStarting optimization...")
        print("Initial left boundary: 0")
        print("Initial right boundary: 0")
        start_time = time.time()
        
        print(f"PyTorch thread count during optimization: {torch.get_num_threads()}")
        
        # Rest of the method remains unchanged
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            obj_val = self.objective()
            objective_values.append(obj_val.item())
            
            # Save current boundary controls
            saved_boundary_controls.append(self.boundary_controls.detach().clone())
            
            # Generate and save plots for this iteration
            self.plot_iteration_results(
                iteration=i, 
                boundary_controls=saved_boundary_controls[-1], 
                objective_values=objective_values,
                total_iterations=num_iterations
            )
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i}, Objective: {obj_val.item():.6f}, Elapsed: {elapsed:.2f}s")
            
            if torch.isnan(obj_val):
                print(f"NaN detected at iteration {i}, stopping optimization")
                break
            
            obj_val.backward()
            optimizer.step()
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f} seconds")
        print(f"Final objective value: {self.objective().item():.6f}")
        
        # After optimization, create animation
        self.create_animation(saved_boundary_controls, objective_values)
        
        return self.boundary_controls.detach(), objective_values

    def plot_iteration_results(self, iteration, boundary_controls, objective_values, total_iterations=10):
        """Plot results for a specific iteration of the optimization."""
        boundary_left = boundary_controls[0]
        boundary_right = boundary_controls[1]
        
        # Solve for the state with this control
        with torch.no_grad():
            u_optimal = self.solve_burgers_equation(boundary_controls)
        
        # Move tensors to CPU for plotting with numpy
        x_np = self.x.cpu().numpy()
        t_np = self.t.cpu().numpy()
        u_optimal_np = u_optimal.cpu().numpy()
        boundary_left_np = boundary_left.cpu().numpy()
        boundary_right_np = boundary_right.cpu().numpy()
        q_np = self.q(self.x).cpu().numpy()
        u0_np = self.u0(self.x).cpu().numpy()
        
        # Calculate fixed color range for all plots
        # This should only be done once and cached, but we'll do it here for simplicity
        u_min, u_max = -2, 2  # Fixed reasonable range for Burger's equation
        # Fixed y-axis limits for boundary controls
        controls_min, controls_max = -2, 2
        
        # Create the same 2×2 grid as in plot_results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Optimization progress with fixed axes
        if len(objective_values) > 0:
            axes[0, 0].plot(objective_values)
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Objective Value')
            axes[0, 0].set_title('Optimization Progress')
            axes[0, 0].set_yscale('log')
            
            # Set fixed y-axis for log scale (for all iterations)
            # Use a reasonable fixed range based on expected objective values
            axes[0, 0].set_ylim([0.001, 1.0])  # Fixed y-axis range in log scale
            
            # Set fixed x-axis based on total expected iterations
            axes[0, 0].set_xlim([0, total_iterations-1])
            
            axes[0, 0].grid(True)
        
        # Plot 2: Target vs achieved state
        axes[0, 1].plot(x_np, q_np, 'r-', label='Target q(x)')
        axes[0, 1].plot(x_np, u_optimal_np[-1], 'b--', label='Final state u(x,T)')
        axes[0, 1].plot(x_np, u0_np, 'g:', label='Initial state u0(x)')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Target vs Achieved State')
        axes[0, 1].set_ylim([0, 1])  # Fixed y-axis range
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Boundary controls
        axes[1, 0].plot(t_np, boundary_left_np, 'r-', label='Left boundary (x=0)')
        axes[1, 0].plot(t_np, boundary_right_np, 'b-', label='Right boundary (x=L)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Control Value')
        axes[1, 0].set_title('Boundary Controls')
        axes[1, 0].set_ylim([controls_min, controls_max])  # Fixed y-axis range
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: State evolution heatmap
        im = axes[1, 1].imshow(
            u_optimal_np, 
            aspect='auto', 
            origin='lower',
            extent=[0, self.L, 0, self.T],
            cmap='plasma',
            vmin=u_min,  # Fixed color range min
            vmax=u_max   # Fixed color range max
        )
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('t')
        axes[1, 1].set_title('State Evolution')
        plt.colorbar(im, ax=axes[1, 1], label='u(x,t)')
        
        # Add iteration number as figure suptitle
        plt.suptitle(f'Burger\'s Equation Optimization - Iteration {iteration}', fontsize=16)
        
        # Save figure with iteration number in filename
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        plt.savefig(f'images/burgers_iteration_{iteration:04d}.png', dpi=300)
        plt.close(fig)  # Close to avoid memory issues with many plots

    def create_animation(self, saved_boundary_controls, objective_values):
        """Create an animation from saved iteration results."""
        try:
            import imageio
            
            # Check if we have the necessary files
            image_files = sorted([f for f in os.listdir('images') if f.startswith('burgers_iteration_') and f.endswith('.png')])
            
            if not image_files:
                print("No image files found for animation. Skipping animation creation.")
                return
                
            print(f"Creating animation from {len(image_files)} frames...")
            
            # Create gif animation
            with imageio.get_writer('images/burgers_optimization.gif', mode='I', duration=0.2) as writer:
                for filename in image_files:
                    image = imageio.imread(os.path.join('images', filename))
                    writer.append_data(image)
                    
            print("Animation saved to images/burgers_optimization.gif")
            
            # Also create an mp4 if possible
            try:
                with imageio.get_writer('images/burgers_optimization.mp4', fps=5) as writer:
                    for filename in image_files:
                        image = imageio.imread(os.path.join('images', filename))
                        writer.append_data(image)
                print("MP4 video saved to images/burgers_optimization.mp4")
            except Exception as e:
                print(f"Could not create MP4 video: {e}")
                
        except ImportError:
            print("imageio module not found. Install with 'pip install imageio' to create animations.")
        except Exception as e:
            print(f"Error creating animation: {e}")

    def plot_results(self, boundary_controls=None, objective_values=None):
        """Plot final results from the optimization (Burger's equation)."""
        if boundary_controls is None:
            boundary_controls = self.boundary_controls.detach()
            
        boundary_left = boundary_controls[0]
        boundary_right = boundary_controls[1]
        
        # Solve for the optimized state with no gradient tracking
        with torch.no_grad():
            u_optimal = self.solve_burgers_equation(boundary_controls)
        
        # Move tensors to CPU for plotting with numpy
        x_np = self.x.cpu().numpy()
        t_np = self.t.cpu().numpy()
        u_optimal_np = u_optimal.cpu().numpy()
        boundary_left_np = boundary_left.cpu().numpy()
        boundary_right_np = boundary_right.cpu().numpy()
        q_np = self.q(self.x).cpu().numpy()
        u0_np = self.u0(self.x).cpu().numpy()
        
        # Calculate fixed color range for all plots
        u_min, u_max = -2, 2  # Fixed reasonable range for Burger's equation
        # Fixed y-axis limits for boundary controls
        controls_min, controls_max = -2, 2
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if objective_values is not None:
            axes[0, 0].plot(objective_values)
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Objective Value')
            axes[0, 0].set_title('Optimization Progress')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True)
        
        axes[0, 1].plot(x_np, q_np, 'r-', label='Target q(x)')
        axes[0, 1].plot(x_np, u_optimal_np[-1], 'b--', label='Final state u(x,T)')
        axes[0, 1].plot(x_np, u0_np, 'g:', label='Initial state u0(x)')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Target vs Achieved State (Burger\'s)')
        axes[0, 1].set_ylim([u_min, u_max])  # Fixed y-axis range
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(t_np, boundary_left_np, 'r-', label='Left boundary (x=0)')
        axes[1, 0].plot(t_np, boundary_right_np, 'b-', label='Right boundary (x=L)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Control Value')
        axes[1, 0].set_title('Boundary Controls (Burger\'s)')
        axes[1, 0].set_ylim([controls_min, controls_max])  # Fixed y-axis range
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        im = axes[1, 1].imshow(
            u_optimal_np, 
            aspect='auto', 
            origin='lower',
            extent=[0, self.L, 0, self.T],
            cmap='plasma',
            vmin=u_min,  # Fixed color range min
            vmax=u_max   # Fixed color range max
        )
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('t')
        axes[1, 1].set_title('State Evolution (Burger\'s Eq.)')
        plt.colorbar(im, ax=axes[1, 1], label='u(x,t)')
        
        # Additional snapshots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(x_np, u_optimal_np[0], 'b-')
        axes[0].set_title('Initial State (t=0)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('u(x,0)')
        axes[0].set_ylim([u_min, u_max])  # Fixed y-axis range
        axes[0].grid(True)
        
        mid_idx = self.Nt // 2
        axes[1].plot(x_np, u_optimal_np[mid_idx], 'g-')
        axes[1].set_title(f'Mid State (t={t_np[mid_idx]:.2f})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel(f'u(x,{t_np[mid_idx]:.2f})')
        axes[1].set_ylim([u_min, u_max])  # Fixed y-axis range
        axes[1].grid(True)
        
        axes[2].plot(x_np, q_np, 'r--', label='Target')
        axes[2].plot(x_np, u_optimal_np[-1], 'b-', label='Final')
        axes[2].set_title(f'Final State (t={self.T})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel(f'u(x,{self.T})')
        axes[2].set_ylim([u_min, u_max])  # Fixed y-axis range
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('burgers_equation_pytorch_snapshots.png', dpi=300)

def main():
    """Main function to run the optimization and display results for Burger's equation."""
    # Set up argument parser for command line interface
    parser = argparse.ArgumentParser(description='Burger\'s Equation with Boundary Control')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for computation if available')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of GPU to use if multiple are available')
    parser.add_argument('--nx', type=int, default=50, help='Number of spatial points')
    parser.add_argument('--nt', type=int, default=200, help='Number of time steps')
    parser.add_argument('--nu', type=float, default=0.5, help='Viscosity parameter')
    parser.add_argument('--iterations', type=int, default=100, help='Number of optimization iterations')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for optimization')
    args = parser.parse_args()
    
    # Set device based on arguments
    device = get_device(use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    
    controller = BurgersEquationControl(
        Nx=400,           
        Nt=400,          
        L=1.0,          
        T=1.0,           
        nu=0.5,      
        alpha=0.001, # before .0001     
        device=device    
    )
    
    # Run optimization
    boundary_controls, objective_values = controller.optimize_control(
        num_iterations=args.iterations,
        learning_rate=args.learning_rate
    )
    
    # Plot results
    controller.plot_results(boundary_controls, objective_values)

if __name__ == "__main__":
    main()
