"""Solution."""
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, ConstantKernel, WhiteKernel

# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.NOISE_STD_F = 0.15  # Standard deviation of noise for f
        self.NOISE_STD_V = 0.0001  # Standard deviation of noise for v

        self.kernel_f = 1.0 * Matern(length_scale=0.5, nu=2.5) + WhiteKernel()
        # self.kernel_f = 0.5 * RBF(length_scale=0.5)
        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=1, n_restarts_optimizer=15)
        # self.kernel_v = 0.5 * RBF(length_scale=10)
        self.kernel_v = DotProduct() + 1 * Matern(length_scale=0.5, nu=2.5) + ConstantKernel(constant_value=4, constant_value_bounds='fixed') + WhiteKernel()
        # self.kernel_v = DotProduct() + np.sqrt(2) * RBF(length_scale=0.5) + ConstantKernel(constant_value=4, constant_value_bounds="fixed")
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=1, n_restarts_optimizer=15)

        # self.kernel_f = 0.5 * RBF(length_scale=0.5) + WhiteKernel()
        # self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=1, n_restarts_optimizer=15)
        # rbf_kernel = np.sqrt(2) * RBF(length_scale=10)
        # # Define the Linear kernel (DotProduct in sklearn)
        # linear_kernel = DotProduct()
        # # Define the Constant Kernel with a value of 4
        # constant_kernel = ConstantKernel(constant_value=4, constant_value_bounds="fixed")
        # # Additive kernel for gp_v
        # self.kernel_v = rbf_kernel + linear_kernel + constant_kernel + WhiteKernel()
        # self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=1, n_restarts_optimizer=15)

        self.x = []  # List to store feature values
        self.y_f = []  # List to store logP values
        self.y_v = []  # List to store SA values

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

        # sample from right and then from left
        # check which one is higher and go in that direction
        # the smaller the y_v and y_variance the bigger the stepsize you can take
        direction = -1 if len(self.y_f) % 2 == 0 else 1
        if len(self.y_v) < 3:
            x_opt = np.clip(
                self.x[0] + 0.018 * ((int(len(self.y_f) - 1) / 2) + 1) * direction,
                *DOMAIN[0]
            )
        elif len(self.y_v) < 9:
            if self.y_v[-2] < 2.8:
                step_size = 0.45
            elif self.y_v[-2] < 3:
                step_size = 0.3
            elif self.y_v[-2] < 3.2:
                step_size = 0.15
            elif self.y_v[-2] < 3.4:
                step_size = 0.05
            elif self.y_v[-2] < 3.6:
                step_size = 0.015
            else:
                step_size = 0.005
            x_opt = np.clip(self.x[-2] + step_size * direction, *DOMAIN[0])
        elif len(self.y_v) < 11:
            # left side is greater
            left_sum = sum(self.y_f[2:9:2])
            right_sum = sum(self.y_f[1:9:2])
            # print(f'1 sum (right): {right_sum}')
            # print(f'2 sum (left): {left_sum}')
            if sum(self.y_f[1:9:2]) < sum(self.y_f[2:9:2]):
                index = len(self.y_v) - 9
                start_index = index * 2
                next_index = start_index + 2 
                x_opt = self.x[start_index] + (self.x[next_index] - self.x[start_index]) / 2
                # print(f'start_index: {start_index}, next_index: {next_index}')
                x_opt = np.clip(x_opt, *DOMAIN[0])
            else:
                index = len(self.y_v) - 9
                start_index = 0 if index == 0 else (index - 1) * 2 + 1
                next_index = 1 if index == 0 else start_index + 2 
                x_opt = self.x[start_index] - (self.x[start_index] - self.x[next_index]) / 2
                # print(f'start_index: {start_index}, next_index: {next_index}')
                x_opt = np.clip(x_opt, *DOMAIN[0])
        else:
            x_opt = self.optimize_acquisition_function()
            sa_value, sa_std = self.gp_v.predict(np.atleast_2d(np.array([x_opt])), return_std=True)
            # print(f'sa_value: {sa_value}, sa_std: {sa_std}')
            max_step = 0.5
            if sa_value > 3.8:
                if sa_std > 0.2:
                    max_step = 0.002
                else:
                    max_step = 0.005
            elif sa_value > 3.6:
                if sa_std > 0.4:
                    max_step = 0.03
                else:
                    max_step = 0.04
            elif sa_value > 3:
                if sa_std > 0.8:
                    max_step = 0.01
                else:
                    max_step = 0.075
            
            min_diff = float('inf')
            closest_x = None
            closest_index = 0
            # Iterate over each array in the list
            for i, array in enumerate(self.x):
                # Compute the absolute difference between the target and the element in the array
                diff = abs(array[0] - x_opt)

                # Update the minimum difference and the closest array if necessary
                if diff < min_diff:
                    min_diff = diff
                    closest_x = array
                    closest_index = i
            if self.y_v[closest_index] > 3.8:
                max_step = 0.005
            elif self.y_v[closest_index] > 3.6:
                max_step = 0.01
            elif self.y_v[closest_index] > 3.4:
                max_step = 0.04
            elif self.y_v[closest_index] > 3.2:
                max_step = 0.08
            elif self.y_v[closest_index] > 3:
                max_step = 0.15
            if abs(x_opt - closest_x) > max_step:
                if x_opt > closest_x:
                    x_opt = closest_x + max_step
                else:
                    x_opt = closest_x - max_step
            # print(f'sa_value: {sa_value}, sa_std: {sa_std}')
        while (x_opt in [x[0] for x in self.x]):
            a, b = random.sample([x[0] for x in self.x], 2)
            x_opt = (a + b) / 2
        return x_opt

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
        if np.isnan(x[0, 0]):
            x = np.array([0])
            x = np.atleast_2d(x)
        mu_f = self.gp_f.predict(x, return_std=False)
        mu_v, std_v = self.gp_v.predict(x, return_std=True)
        temp = mu_v + 0.22 * std_v - 4
        lagrangian_obj =  mu_f - 100 * max(temp, 0)
        # if mu_v >= SAFETY_THRESHOLD: 
        #     lagrangian_obj = mu_f - 500 * mu_v
        # else:
        #     lagrangian_obj = mu_f
        return lagrangian_obj

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
        # noise_f = np.random.normal(0, self.NOISE_STD_F)
        # noise_v = np.random.normal(0, self.NOISE_STD_V)
        if isinstance(x, float):
            x = np.array([x])
        self.x.append(x)  # Assuming x is a scalar, wrap it in a list
        print(f'x: {x}, f: {f}, v: {v}')
        self.y_f.append(f)
        self.y_v.append(v)
        self.gp_f.fit(self.x, self.y_f)
        self.gp_v.fit(self.x, self.y_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        X_array = np.array(self.x)
        Y_logP_array = np.array(self.y_f)
        Y_SA_array = np.array(self.y_v)

        # Filter points that satisfy the SA constraint
        safe_points = X_array[Y_SA_array < SAFETY_THRESHOLD]

        # Find the point with the highest logP among the safe points
        if len(safe_points) > 0:
            safe_logP_values = Y_logP_array[Y_SA_array < SAFETY_THRESHOLD]
            index_best = np.argmax(safe_logP_values)
            solution = safe_points[index_best]
        else:
            # Handle the case where no points satisfy the constraint
            solution = None  # or some default value or behavior
        if solution == self.x[0][0]:
            solution = self.x[0][0] + 0.002
        return solution

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


