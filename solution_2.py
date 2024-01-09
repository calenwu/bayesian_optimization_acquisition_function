"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, Sum, WhiteKernel, Product
import matplotlib.pyplot as plt
## import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        ## initialize hyperparameters lambda, beta, length scale bounds type of kernel for both the f and v gaussian process
        ## initialize the gaussian process with sk learn for f and v 


        self.lambda_hyperparam = 100000.0  # Example value
        self.beta_hyperparam = 3.0  # Example value
        self.beta_v_hyperparam = 100

        self.kernel_v = "rbf"
        self.kernel_f = "rbf"

        noise_level_f = 0.15 ** 2  # Noise variance for kernel_f
        noise_level_v = 0.0001 ** 2  # Noise variance for base_kernel_v

        if self.kernel_f == "matern":
            kernel_f = Matern(length_scale=0.5, nu=2.5, length_scale_bounds="fixed") + WhiteKernel(noise_level=noise_level_f)

        if self.kernel_v == "matern":
            base_kernel_v = Matern(length_scale=10, nu=2.5, length_scale_bounds="fixed") + WhiteKernel(noise_level=noise_level_v)

        if self.kernel_f == "rbf":
            kernel_f = Product(ConstantKernel(0.5), RBF(length_scale=0.5, length_scale_bounds="fixed")) + WhiteKernel(noise_level=noise_level_f)

        if self.kernel_v == "rbf":
            base_kernel_v = Product(ConstantKernel((2.0**0.5)), RBF(length_scale=0.5, length_scale_bounds="fixed")) + WhiteKernel(noise_level=noise_level_v)
        
        
        kernel_v = base_kernel_v


        # Initialize Gaussian Processes
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v)
        


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """

        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        ## return optimize_acquisition_function()
        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        # TODO: Implement the acquisition function you want to optimize.

        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        mu_v_mean0, sigma_v = self.gp_v.predict(x, return_std=True)


        # Adjust the mean of v by adding a constant (4.0 in this case).
        mu_v = mu_v_mean0 + 4.0


        # Calculate the acquisition function.
        # It involves a trade-off between the expected improvement (mu_f),
        # penalization based on the constraint model (mu_v and sigma_v),
        # and exploration term (sigma_f).
        acquisition = mu_f - self.lambda_hyperparam * np.maximum(mu_v-4.0 + self.beta_v_hyperparam*sigma_v, 0) + self.beta_hyperparam * sigma_f 

        return acquisition


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """

        # TODO: Add the observed data {x, f, v} to your model.
        ## add datapoint to gaussian process f and to gaussian process v at x

        # Function to check and update the Gaussian Process
        def update_gp(gp, x_new, y_new):
            if not hasattr(gp, 'X_train_'):
                # Initialize if X_train_ is empty
                gp.fit(np.array([x_new]), np.array([y_new]))
            else:
                # Update and refit if X_train_ is not empty
                X_updated = np.append(gp.X_train_, [[x_new]], axis=0)
                y_updated = np.append(gp.y_train_, [y_new])
                gp.fit(X_updated, y_updated)

        # Update Gaussian Process for f
        update_gp(self.gp_f, x, f)

        # Update Gaussian Process for v re-adjust v values to prior mean 0
        update_gp(self.gp_v, x, v-4.0)
        return

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        ## take gaussian process v and f and do the following: max(f(x)- lambda*max(v(x),4))
        ## return x where the mean function is maximal of that. 
        # Extract training data for f and v
        # Extract training data for f and v
        X_f = self.gp_f.X_train_
        y_f = self.gp_f.y_train_
        y_v = self.gp_v.y_train_

        x0 = X_f[0][0]  # First point in X_train_

        # Define a function to check if a point is in the trivial zone around x0
        def is_in_non_safe_zone(x):
            return abs(x0 - x) <= 0.1

        # Filter out points in the non-safe zones from the training data
        safe_indices_train = (y_v <= SAFETY_THRESHOLD) & (~is_in_non_safe_zone(X_f[:, 0]))
        safe_X_f_train = X_f[safe_indices_train]
        safe_y_f_train = y_f[safe_indices_train]

        # Find the maximum in the training data if safe points exist
        max_value_train = None
        if len(safe_y_f_train) > 0:
            idx_max_train = np.argmax(safe_y_f_train)
            max_value_train = safe_X_f_train[idx_max_train][0]

        # Define the search space based on the DOMAIN
        search_space = DOMAIN[0]  # Assuming DOMAIN = np.array([[0, 10]])

        # Create a grid of points in the search space
        lower_bound, upper_bound = search_space
        X_grid = np.linspace(lower_bound, upper_bound, 5000).reshape(-1, 1)

        # Predict the mean and variance of f and v over the grid
        mu_f, _ = self.gp_f.predict(X_grid, return_std=True)
        mu_v, _ = self.gp_v.predict(X_grid, return_std=True)

        # Identify safe points based on the mean prediction of v in the grid
        safe_indices_grid = (mu_v <= SAFETY_THRESHOLD) & (~is_in_non_safe_zone(X_grid[:, 0]))
        safe_X_grid = X_grid[safe_indices_grid]
        safe_mu_f_grid = mu_f[safe_indices_grid]

        # Find the maximum in the sampled grid if safe points exist
        max_value_grid = None
        if len(safe_mu_f_grid) > 0:
            idx_max_grid = np.argmax(safe_mu_f_grid)
            max_value_grid = safe_X_grid[idx_max_grid][0]

        # Compare the maximum values from evaluated data and sampled grid, and return the highest
        if max_value_train is not None and max_value_grid is not None:
            return max(max_value_train, max_value_grid)
        elif max_value_train is not None:
            return max_value_train
        elif max_value_grid is not None:
            return max_value_grid
        else:
            return None  # No safe points found

    def plot(self, plot_recommendation: bool = True):
            """Plot objective and constraint posterior for debugging (OPTIONAL).

            Parameters
            ----------
            plot_recommendation: bool
                Plots the recommended point if True.
            """
            pass

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
