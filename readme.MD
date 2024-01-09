# Task Summary: Local Maxima Discovery in Constrained Environment
## Objective
The task involved identifying the local maxima of a distribution, starting from a given initial point with the capability to query an additional 20 points.

## Methodology and Constraints
Modeling Approach: Utilized a Gaussian Process to model the underlying distribution, adhering to specific kernels and parameters outlined in the task.
Constraints Handling: Direct sampling of random points was restricted due to predefined constraints. Violating these constraints resulted in negative rewards.
Optimization Strategy: The primary goal was to develop an effective acquisition function that adeptly balanced exploitation and exploration.
Implemented Solution: Employed the Upper Confidence Bound function, enhanced with Lagrangian relaxation, to navigate the optimization landscape efficiently while adhering to the constraints.