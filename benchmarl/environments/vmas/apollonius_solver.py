"""
Apollonius Optimization Solver for Target Defense Game
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple
import warnings

def apo_center(x_d_i: np.ndarray, x_a_j: np.ndarray, nu: float) -> np.ndarray:
    """
    Compute the center of the Apollonius circle.
    
    The circle contains points where |PA|/|PD| = nu.
    When nu > 1 (defender faster), these are points the defender can reach first.
    
    Args:
        x_d_i: Defender position [x, y]
        x_a_j: Attacker position [x, y] 
        nu: Speed ratio (defender_speed / attacker_speed)
        
    Returns:
        Center of Apollonius circle [x, y]
    """
    x_d_i = np.array(x_d_i, dtype=float)
    x_a_j = np.array(x_a_j, dtype=float)
    
    # For points on circle: |PA|/|PD| = nu
    factor = 1.0 / (nu**2 - 1)
    center = factor * np.array([
        nu**2 * x_a_j[0] - x_d_i[0],
        nu**2 * x_a_j[1] - x_d_i[1]
    ])
    return center

def apo_radius(x_d_i: np.ndarray, x_a_j: np.ndarray, nu: float) -> float:
    """
    Compute the radius of the Apollonius circle.
    
    Args:
        x_d_i: Defender position [x, y]
        x_a_j: Attacker position [x, y]
        nu: Speed ratio (defender_speed / attacker_speed)
        
    Returns:
        Radius of Apollonius circle
    """
    x_d_i = np.array(x_d_i, dtype=float)
    x_a_j = np.array(x_a_j, dtype=float)
    
    distance = np.linalg.norm(x_a_j - x_d_i)
    # For defender interception circles (ratio = 1/nu)
    radius = (nu / abs(nu**2 - 1)) * distance
    return radius

def solve_apollonius_optimization(attacker_pos: np.ndarray, 
                                 defender_positions: List[np.ndarray], 
                                 nu: float) -> Dict:
    """
    Solve the Apollonius optimization problem to find the minimum y-coordinate
    the attacker can reach given the defender positions and speed ratio.
    
    Args:
        attacker_pos: Current attacker position [x, y]
        defender_positions: List of defender positions [[x1, y1], [x2, y2], ...]
        nu: Speed ratio (defender_speed / attacker_speed) 
        
    Returns:
        Dictionary containing:
        - success: Whether optimization succeeded
        - min_y_coordinate: Minimum y-coordinate attacker can reach
        - attacker_payoff: Payoff for attacker
        - defender_payoff: Payoff for defenders
        - failure_reason: Reason for failure (if success=False)
    """
    try:
        attacker_pos = np.array(attacker_pos, dtype=float)
        defender_positions = [np.array(pos, dtype=float) for pos in defender_positions]
        
        # Generate Apollonius circles for each defender
        centers = []
        radii = []
        
        for defender_pos in defender_positions:
            center = apo_center(defender_pos, attacker_pos, nu)
            radius = apo_radius(defender_pos, attacker_pos, nu)
            centers.append(center)
            radii.append(radius)
        
        # Scale problem for numerical stability with small radii
        # Find the smallest radius to use as scaling factor
        min_radius = min(radii)
        scale_factor = 1.0 if min_radius > 1e-2 else 1.0 / min_radius
        
        # Set up optimization problem: minimize y subject to being INSIDE all circles
        # With nu > 1, circles represent defender dominance regions
        # The intersection is where all defenders can reach
        # Use scaled variables for better numerical stability
        x_scaled = cp.Variable()
        y_scaled = cp.Variable()
        
        objective = cp.Minimize(y_scaled)
        constraints = []
        
        # Add constraint for each Apollonius circle
        for center, radius in zip(centers, radii):
            # Scale center and radius
            center_scaled = center * scale_factor
            radius_scaled = radius * scale_factor
            
            # Point must be inside the circle (defender can reach)
            constraint = cp.norm(cp.hstack([x_scaled - center_scaled[0], y_scaled - center_scaled[1]]), 2) <= radius_scaled
            constraints.append(constraint)
        
        # Remove boundary constraint to allow negative y values for training purposes
        # constraints.append(y_scaled >= 0)  # Removed constraint
        
        # Choose solver based on availability
        solver = None
        solver_kwargs = {}
        
        if 'CLARABEL' in cp.installed_solvers():
            solver = cp.CLARABEL
            # Tighten tolerances for better accuracy with small radii
            solver_kwargs = {
                'tol_feas': 1e-9,
                'tol_gap_abs': 1e-9,
                'tol_gap_rel': 1e-9,
            }
        elif 'ECOS' in cp.installed_solvers():
            solver = cp.ECOS
            solver_kwargs = {
                'abstol': 1e-9,
                'reltol': 1e-9,
                'feastol': 1e-9,
            }
        elif 'SCS' in cp.installed_solvers():
            solver = cp.SCS
            solver_kwargs = {
                'eps': 1e-9,
            }
        else:
            warnings.warn("No suitable solver found. Using default solver.")
            solver = None
            solver_kwargs = {}
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        if solver:
            problem.solve(solver=solver, **solver_kwargs)
        else:
            problem.solve()
        
        if problem.status == cp.OPTIMAL:
            # Unscale the results
            min_x = float(x_scaled.value) / scale_factor
            min_y = float(y_scaled.value) / scale_factor
            
            # Don't clip min_y - keep raw value for training purposes
            # min_y = max(0.0, min_y)  # Removed clipping
            
            # Compute payoffs
            attacker_payoff = -min_y  # Negative y for attacker
            defender_payoff = min_y   # Positive y for defenders
            
            return {
                'success': True,
                'min_x_coordinate': min_x,  # Include x for future use
                'min_y_coordinate': min_y,
                'attacker_payoff': attacker_payoff,
                'defender_payoff': defender_payoff
            }
        else:
            failure_reason = f"Optimization failed with status: {problem.status}"
            warnings.warn(f"Apollonius solver failed: {failure_reason}")
            return {
                'success': False,
                'failure_reason': failure_reason
            }
            
    except Exception as e:
        failure_reason = f"Exception in Apollonius solver: {str(e)}"
        warnings.warn(failure_reason)
        return {
            'success': False,
            'failure_reason': failure_reason
        }