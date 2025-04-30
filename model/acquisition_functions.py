# Copyright (c) 2025 Insitute of Material Science and Sustainability 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Multi-objective acquisition functions for property-guided active learning."""

import jax
import jax.numpy as jnp
from typing import List, Callable, Tuple, Dict, Any, Optional, Union
import numpy as np

def expected_improvement(mean: jnp.ndarray, 
                        std: jnp.ndarray, 
                        best: float,
                        xi: float = 0.01) -> jnp.ndarray:
    """Compute expected improvement acquisition function.
    
    Args:
        mean: Predicted mean values.
        std: Predicted standard deviation values.
        best: Current best observed value.
        xi: Exploration parameter.
        
    Returns:
        Expected improvement values.
    """
    # Handle case where std is 0
    std = jnp.maximum(std, 1e-6)
    
    # Calculate z-score
    z = (mean - best - xi) / std
    
    # Calculate PDF and CDF of standard normal
    phi = jax.scipy.stats.norm.pdf(z)
    Phi = jax.scipy.stats.norm.cdf(z)
    
    # Calculate expected improvement
    return (mean - best - xi) * Phi + std * phi

def probability_of_improvement(mean: jnp.ndarray,
                              std: jnp.ndarray,
                              best: float,
                              xi: float = 0.01) -> jnp.ndarray:
    """Compute probability of improvement acquisition function.
    
    Args:
        mean: Predicted mean values.
        std: Predicted standard deviation values.
        best: Current best observed value.
        xi: Exploration parameter.
        
    Returns:
        Probability of improvement values.
    """
    # Handle case where std is 0
    std = jnp.maximum(std, 1e-6)
    
    # Calculate z-score
    z = (mean - best - xi) / std
    
    # Calculate CDF of standard normal
    return jax.scipy.stats.norm.cdf(z)

def upper_confidence_bound(mean: jnp.ndarray,
                          std: jnp.ndarray,
                          beta: float = 2.0) -> jnp.ndarray:
    """Compute upper confidence bound acquisition function.
    
    Args:
        mean: Predicted mean values.
        std: Predicted standard deviation values.
        beta: Exploration-exploitation trade-off parameter.
        
    Returns:
        UCB acquisition function values.
    """
    return mean + beta * std

def calculate_hypervolume(points: jnp.ndarray, 
                         reference_point: jnp.ndarray) -> float:
    """Calculate hypervolume indicator for multi-objective optimization.
    
    Args:
        points: Array of non-dominated points [n_points, n_objectives].
        reference_point: Reference point for hypervolume calculation [n_objectives].
        
    Returns:
        Hypervolume value.
    """
    # Sort points by first objective
    sorted_idx = jnp.argsort(points[:, 0])
    sorted_points = points[sorted_idx]
    
    # Initialize hypervolume
    hv = 0.0
    
    # For 2D case, calculate hypervolume by summing rectangles
    if points.shape[1] == 2:
        # Start with worst y value (reference point)
        y_prev = reference_point[1]
        
        for i in range(sorted_points.shape[0]):
            # Current point
            x, y = sorted_points[i]
            
            # Add area of rectangle
            hv += (reference_point[0] - x) * (y_prev - y)
            
            # Update y value
            y_prev = y
    else:
        # For higher dimensions, this is just a placeholder
        # In practice, would use a proper hypervolume calculation algorithm
        hv = jnp.prod(jnp.abs(reference_point - jnp.min(points, axis=0)))
    
    return hv

def expected_hypervolume_improvement(means: jnp.ndarray,
                                   stds: jnp.ndarray,
                                   pareto_front: jnp.ndarray,
                                   reference_point: jnp.ndarray,
                                   n_samples: int = 100,
                                   seed: int = 0) -> jnp.ndarray:
    """Compute expected hypervolume improvement for multi-objective optimization.
    
    Args:
        means: Predicted mean values for each objective [n_points, n_objectives].
        stds: Predicted standard deviation values for each objective [n_points, n_objectives].
        pareto_front: Current Pareto front of non-dominated solutions [n_pareto, n_objectives].
        reference_point: Reference point for hypervolume calculation [n_objectives].
        n_samples: Number of Monte Carlo samples for EHVI estimation.
        seed: Random seed for sampling.
        
    Returns:
        Expected hypervolume improvement values [n_points].
    """
    # Check input dimensions
    n_points, n_objectives = means.shape
    assert stds.shape == means.shape, "Means and stds must have the same shape"
    assert pareto_front.shape[1] == n_objectives, "Pareto front must have same number of objectives"
    assert reference_point.shape == (n_objectives,), "Reference point must have n_objectives elements"
    
    # Calculate baseline hypervolume
    base_hv = calculate_hypervolume(pareto_front, reference_point)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Compute EHVI for each point
    def compute_ehvi(i):
        # Extract mean and std for current point
        point_mean = means[i]
        point_std = stds[i]
        
        # Generate Monte Carlo samples
        key_i = jax.random.fold_in(key, i)
        samples = jnp.zeros((n_samples, n_objectives))
        
        for j in range(n_objectives):
            key_ij = jax.random.fold_in(key_i, j)
            samples = samples.at[:, j].set(
                jax.random.normal(key_ij, (n_samples,)) * point_std[j] + point_mean[j]
            )
        
        # Calculate hypervolume improvement for each sample
        def calculate_hvi(sample):
            # Add sample to Pareto front
            new_front = jnp.vstack([pareto_front, sample[None, :]])
            
            # Calculate new hypervolume
            new_hv = calculate_hypervolume(new_front, reference_point)
            
            # Return improvement
            return jnp.maximum(0.0, new_hv - base_hv)
        
        # Vectorize HVI calculation over samples
        hvi_values = jax.vmap(calculate_hvi)(samples)
        
        # Return expected HVI
        return jnp.mean(hvi_values)
    
    # Vectorize EHVI calculation over points
    return jax.vmap(compute_ehvi)(jnp.arange(n_points))

class PropertyTargetType:
    """Enum for property target types."""
    MAXIMIZE = 'maximize'  # Higher values are better
    MINIMIZE = 'minimize'  # Lower values are better
    TARGET = 'target'     # Closer to target value is better
    RANGE = 'range'       # Within a specified range is better

class MaterialPropertyAcquisition:
    """Multi-objective acquisition functions for materials property optimization."""
    
    def __init__(self,
                property_weights: Dict[str, float],
                property_types: Optional[Dict[str, str]] = None,
                property_targets: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
                stability_threshold: float = 0.0,
                beta: float = 2.0,
                xi: float = 0.01,
                reference_point: Optional[Dict[str, float]] = None):
        """Initialize acquisition function.
        
        Args:
            property_weights: Dictionary mapping property names to weights.
            property_types: Dictionary mapping property names to target types
                (maximize, minimize, target, range).
            property_targets: Dictionary mapping property names to target values or ranges.
            stability_threshold: Threshold for stability (hull distance).
            beta: Exploration parameter for UCB.
            xi: Exploration parameter for EI/PI.
            reference_point: Reference point for hypervolume calculation.
        """
        self.property_weights = property_weights
        self.property_types = property_types or {}
        self.property_targets = property_targets or {}
        self.stability_threshold = stability_threshold
        self.beta = beta
        self.xi = xi
        self.reference_point = reference_point or {}
        
        # Set default property types
        for prop in property_weights:
            if prop not in self.property_types:
                if prop in self.property_targets:
                    if isinstance(self.property_targets[prop], tuple):
                        self.property_types[prop] = PropertyTargetType.RANGE
                    else:
                        self.property_types[prop] = PropertyTargetType.TARGET
                else:
                    # Default to maximize
                    self.property_types[prop] = PropertyTargetType.MAXIMIZE
    
    def single_objective(self, 
                       means: Dict[str, jnp.ndarray],
                       stds: Dict[str, jnp.ndarray],
                       best_values: Optional[Dict[str, float]] = None) -> jnp.ndarray:
        """Calculate single-objective acquisition function values.
        
        Args:
            means: Dictionary mapping property names to predicted means.
            stds: Dictionary mapping property names to predicted standard deviations.
            best_values: Dictionary mapping property names to current best values.
            
        Returns:
            Acquisition function values.
        """
        # Initialize scores
        scores = {}
        
        # Calculate scores for each property
        for prop, weight in self.property_weights.items():
            if prop not in means or prop not in stds:
                continue
            
            prop_mean = means[prop]
            prop_std = stds[prop]
            prop_type = self.property_types.get(prop, PropertyTargetType.MAXIMIZE)
            
            # Adjust sign for minimization
            if prop_type == PropertyTargetType.MINIMIZE:
                prop_mean = -prop_mean
            
            # Calculate acquisition function based on property type
            if prop_type in [PropertyTargetType.MAXIMIZE, PropertyTargetType.MINIMIZE]:
                # Use UCB for maximization/minimization
                scores[prop] = weight * upper_confidence_bound(prop_mean, prop_std, self.beta)
            
            elif prop_type == PropertyTargetType.TARGET:
                # Use distance-based score for target value
                target = self.property_targets[prop]
                distance = -jnp.abs(prop_mean - target)
                uncertainty_bonus = self.beta * prop_std
                scores[prop] = weight * (distance + uncertainty_bonus)
            
            elif prop_type == PropertyTargetType.RANGE:
                # Use range-based score
                lower, upper = self.property_targets[prop]
                
                # Calculate probability of being in range
                p_lower = jax.scipy.stats.norm.cdf((prop_mean - lower) / jnp.maximum(prop_std, 1e-6))
                p_upper = jax.scipy.stats.norm.cdf((upper - prop_mean) / jnp.maximum(prop_std, 1e-6))
                p_in_range = p_lower * p_upper
                
                scores[prop] = weight * p_in_range
        
        # Combine scores
        total_score = jnp.sum(jnp.array([
            score for score in scores.values()
        ]))
        
        return total_score
    
    def multi_objective(self,
                       means: Dict[str, jnp.ndarray],
                       stds: Dict[str, jnp.ndarray],
                       pareto_front: Optional[List[Dict[str, float]]] = None) -> jnp.ndarray:
        """Calculate multi-objective acquisition function values using EHVI.
        
        Args:
            means: Dictionary mapping property names to predicted means.
            stds: Dictionary mapping property names to predicted standard deviations.
            pareto_front: Current Pareto front of non-dominated solutions.
            
        Returns:
            Acquisition function values.
        """
        if not pareto_front:
            # If no Pareto front provided, fall back to single-objective
            return self.single_objective(means, stds)
        
        # Extract properties with weights
        properties = list(self.property_weights.keys())
        properties = [p for p in properties if p in means and p in stds]
        
        if not properties:
            return jnp.zeros((means[list(means.keys())[0]].shape[0],))
        
        # Extract means and stds for properties
        prop_means = jnp.column_stack([means[p] for p in properties])
        prop_stds = jnp.column_stack([stds[p] for p in properties])
        
        # Transform means based on property types
        for i, prop in enumerate(properties):
            prop_type = self.property_types.get(prop, PropertyTargetType.MAXIMIZE)
            
            if prop_type == PropertyTargetType.MINIMIZE:
                # Negate for minimization
                prop_means = prop_means.at[:, i].set(-prop_means[:, i])
            
            elif prop_type == PropertyTargetType.TARGET:
                # Use negative distance to target
                target = self.property_targets[prop]
                prop_means = prop_means.at[:, i].set(-jnp.abs(prop_means[:, i] - target))
            
            elif prop_type == PropertyTargetType.RANGE:
                # Use probability of being in range
                lower, upper = self.property_targets[prop]
                p_lower = jax.scipy.stats.norm.cdf(
                    (prop_means[:, i] - lower) / jnp.maximum(prop_stds[:, i], 1e-6)
                )
                p_upper = jax.scipy.stats.norm.cdf(
                    (upper - prop_means[:, i]) / jnp.maximum(prop_stds[:, i], 1e-6)
                )
                prop_means = prop_means.at[:, i].set(p_lower * p_upper)
                prop_stds = prop_stds.at[:, i].set(0.1)  # Reduced uncertainty for ranges
        
        # Extract Pareto front values
        pareto_values = []
        for point in pareto_front:
            point_values = []
            for i, prop in enumerate(properties):
                value = point.get(prop, 0.0)
                
                # Transform based on property type
                prop_type = self.property_types.get(prop, PropertyTargetType.MAXIMIZE)
                
                if prop_type == PropertyTargetType.MINIMIZE:
                    value = -value
                elif prop_type == PropertyTargetType.TARGET:
                    target = self.property_targets[prop]
                    value = -jnp.abs(value - target)
                elif prop_type == PropertyTargetType.RANGE:
                    lower, upper = self.property_targets[prop]
                    value = 1.0 if lower <= value <= upper else 0.0
                
                point_values.append(value)
            
            pareto_values.append(point_values)
        
        # Convert to array
        if pareto_values:
            pareto_array = jnp.array(pareto_values)
        else:
            # If no Pareto front, create a dummy point with worst possible values
            pareto_array = jnp.zeros((1, len(properties)))
        
        # Define reference point (worse than all observed values)
        if self.reference_point and all(p in self.reference_point for p in properties):
            ref_point = jnp.array([self.reference_point[p] for p in properties])
        else:
            # Default: slightly worse than worst observed value for each property
            ref_point = jnp.min(pareto_array, axis=0) - 0.1
        
        # Calculate EHVI
        return expected_hypervolume_improvement(
            prop_means, prop_stds, pareto_array, ref_point
        )
    
    def __call__(self, 
                means: Dict[str, jnp.ndarray],
                stds: Dict[str, jnp.ndarray],
                pareto_front: Optional[List[Dict[str, float]]] = None,
                best_values: Optional[Dict[str, float]] = None,
                multi_objective: bool = False) -> jnp.ndarray:
        """Calculate acquisition function value for material candidates.
        
        Args:
            means: Dictionary mapping property names to predicted means.
            stds: Dictionary mapping property names to predicted standard deviations.
            pareto_front: Current Pareto front (for multi-objective optimization).
            best_values: Dictionary of current best values (for single-objective).
            multi_objective: Whether to use multi-objective acquisition.
            
        Returns:
            Acquisition function values.
        """
        # Ensure stability constraint is satisfied
        if 'stability' in means:
            stability_mean = means['stability']
            stability_std = stds.get('stability', jnp.zeros_like(stability_mean))
            
            # Probability of being stable
            p_stable = jax.scipy.stats.norm.cdf(
                (self.stability_threshold - stability_mean) / jnp.maximum(stability_std, 1e-6)
            )
            
            # Set very low score for likely unstable materials
            stability_mask = p_stable
        else:
            stability_mask = 1.0
        
        # Calculate property scores
        if multi_objective and len(self.property_weights) > 1:
            property_score = self.multi_objective(means, stds, pareto_front)
        else:
            property_score = self.single_objective(means, stds, best_values)
        
        # Combine scores with stability constraint
        total_score = stability_mask * property_score
        
        return total_score