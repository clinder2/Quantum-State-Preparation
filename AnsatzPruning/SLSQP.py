from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scipy.optimize import minimize


def slsqp(
    func: Callable,
    x0: np.ndarray,
    bounds: Optional[list] = None,
    maxiter: int = 100,
    **kwargs,
):
    """
    Sequential Least SQuares Programming optimizer.
    
    Args:
        func: Objective function to minimize.
        x0: Initial guess.
        bounds: List of (min, max) pairs for each variable (optional).
        maxiter: Maximum iterations (default: 100).
        **kwargs: Additional arguments passed to scipy.optimize.minimize:
                  - jac: Gradient function
                  - constraints: List of constraint dictionaries
                  - ftol, tol, eps: Tolerance parameters
                  - disp: Print convergence messages
                  - options: Dictionary of options
    
    Returns:
        scipy.optimize.OptimizeResult object.
    """
    if bounds is not None:
        large = 1e10
        bounds = [
            None if bound is None
            else (
                -large if bound[0] == -np.inf else bound[0],
                large if bound[1] == np.inf else bound[1],
            )
            for bound in bounds
        ]
    
    options = kwargs.pop('options', {})
    options['maxiter'] = maxiter
    
    return minimize(
        fun=func,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        options=options,
        **kwargs,
    )
