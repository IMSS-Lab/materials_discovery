"""
Property-guided active learning for materials discovery.

This module implements active learning strategies to efficiently explore
materials space based on predictive uncertainty and targeted property values.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import heapq
from functools import partial

from ..model.bayesian_gnn import BayesianGNN
from ..model.uncertainty import uncertainty_decomposition


class ActiveLearningLoop:
    """Active learning loop for materials discovery."""
    
    def __init__(
        self,
        model: BayesianGNN,
        generator_fn: Callable,
        evaluator_fn: Callable,
        acquisition_fn: Callable,
        initial_dataset: Dict,
        batch_size: int = 32,
        max_iterations: int = 10,
        target_properties: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the active learning loop.
        
        Args:
            model: Bayesian GNN model
            generator_fn: Function to generate candidate materials
            evaluator_fn: Function to evaluate materials (e.g., with DFT)
            acquisition_fn: Function to score and select candidates
            initial_dataset: Initial dataset for model training
            batch_size: Number of materials to evaluate in each iteration
            max_iterations: Maximum number of active learning iterations
            target_properties: Optional dict mapping property names to target ranges
        """
        self.model = model
        self.generator_fn = generator_fn
        self.evaluator_fn = evaluator_fn
        self.acquisition_fn = acquisition_fn
        self.dataset = initial_dataset
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.target_properties = target_properties
        
        self.iteration = 0
        self.history = {
            'iteration': [],
            'model_error': [],
            'acquisition_scores': [],
            'evaluated_candidates': [],
            'discovered_materials': [],
        }
    
    def run(self, rng_key: jnp.ndarray):
        """
        Run the active learning loop.
        
        Args:
            rng_key: JAX random key
        
        Returns:
            Updated dataset and history
        """
        for i in range(self.max_iterations):
            self.iteration = i
            rng_key, subkey = jax.random.split(rng_key)
            
            # Train model on current dataset
            train_state = self._train_model(subkey)
            
            # Generate candidate materials
            rng_key, subkey = jax.random.split(rng_key)
            candidates = self.generator_fn(
                model=self.model, 
                params=train_state.params,
                dataset=self.dataset,
                batch_size=self.batch_size * 10,  # Generate more candidates than we'll evaluate
                rng_key=subkey,
                target_properties=self.target_properties
            )
            
            # Score candidates
            rng_key, subkey = jax.random.split(rng_key)
            scores, uncertainties = self._score_candidates(
                candidates, train_state, subkey
            )
            
            # Select candidates for evaluation
            selected_indices = self._select_candidates(scores, self.batch_size)
            selected_candidates = [candidates[idx] for idx in selected_indices]
            selected_scores = scores[selected_indices]
            
            # Evaluate selected candidates
            evaluation_results = self.evaluator_fn(selected_candidates)
            
            # Update dataset with evaluation results
            self._update_dataset(selected_candidates, evaluation_results)
            
            # Update history
            self._update_history(train_state, selected_candidates, 
                               selected_scores, evaluation_results)
            
            # Early stopping if needed
            if self._should_stop():
                break
        
        return self.dataset, self.history
    
    def _train_model(self, rng_key: jnp.ndarray):
        """
        Train model on current dataset.
        
        Args:
            rng_key: JAX random key
        
        Returns:
            Trained model state
        """
        # Implementation depends on specific training loop,
        # but this would train the Bayesian GNN on the current dataset
        # Using the training procedure from bayesian_gnn.py
        
        # Placeholder - would typically instantiate an optimizer and training state
        train_state = None
        
        return train_state
    
    def _score_candidates(
        self, 
        candidates: List[Dict], 
        train_state: Any,
        rng_key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Score candidates based on acquisition function.
        
        Args:
            candidates: List of candidate materials
            train_state: Trained model state
            rng_key: JAX random key
        
        Returns:
            Tuple of (scores, uncertainties)
        """
        # Prepare candidate inputs
        inputs = self._prepare_inputs(candidates)
        
        # Get predictions and uncertainties
        uncertainties = uncertainty_decomposition(
            model=self.model,
            params=train_state.params,
            inputs=inputs,
            rng_key=rng_key,
            num_models=5,
            num_samples_per_model=10
        )
        
        # Calculate acquisition scores
        scores = self.acquisition_fn(
            means=uncertainties['mean'],
            uncertainties=uncertainties['total_uncertainty'],
            target_properties=self.target_properties
        )
        
        return scores, uncertainties['total_uncertainty']
    
    def _select_candidates(self, scores: jnp.ndarray, count: int) -> List[int]:
        """
        Select candidates with highest acquisition scores.
        
        Args:
            scores: Acquisition scores for each candidate
            count: Number of candidates to select
        
        Returns:
            Indices of selected candidates
        """
        # Convert to numpy for heap operations
        scores_np = np.array(scores)
        
        # Find indices of top scores
        if len(scores_np.shape) > 1:
            # For multi-objective scores, use sum as default strategy
            # (more sophisticated strategies would be implemented in the acquisition function)
            combined_scores = np.sum(scores_np, axis=1)
            top_indices = np.argsort(combined_scores)[-count:]
        else:
            # For single objective
            top_indices = np.argsort(scores_np)[-count:]
        
        return list(top_indices)
    
    def _update_dataset(self, candidates: List[Dict], results: List[Dict]):
        """
        Update dataset with evaluation results.
        
        Args:
            candidates: Evaluated candidates
            results: Evaluation results
        """
        # Extract necessary data from results
        for candidate, result in zip(candidates, results):
            # Add to dataset
            self.dataset.setdefault('graphs', []).append(candidate['graph'])
            self.dataset.setdefault('positions', []).append(candidate['positions'])
            self.dataset.setdefault('boxes', []).append(candidate['box'])
            
            # Add properties from results
            for prop, value in result.items():
                self.dataset.setdefault(prop, []).append(value)
    
    def _update_history(self, train_state: Any, candidates: List[Dict], 
                      scores: jnp.ndarray, results: List[Dict]):
        """
        Update history with results from current iteration.
        
        Args:
            train_state: Current model state
            candidates: Evaluated candidates
            scores: Acquisition scores
            results: Evaluation results
        """
        self.history['iteration'].append(self.iteration)
        
        # Calculate and store model error on validation set (if available)
        # Placeholder - would compute error metrics
        self.history['model_error'].append(0.0)
        
        # Store acquisition scores
        self.history['acquisition_scores'].append(scores)
        
        # Store candidates and results
        self.history['evaluated_candidates'].append(candidates)
        
        # Identify discovered materials (e.g., those with formation energy below threshold)
        # Placeholder - would identify which materials meet stability criteria
        discovered = []
        self.history['discovered_materials'].append(discovered)
    
    def _should_stop(self) -> bool:
        """
        Check if active learning should stop early.
        
        Returns:
            Boolean indicating whether to stop
        """
        # Placeholder - would implement convergence criteria
        # such as diminishing returns in discovery rate
        return False
    
    def _prepare_inputs(self, candidates: List[Dict]) -> Dict:
        """
        Prepare inputs for model prediction.
        
        Args:
            candidates: List of candidate materials
        
        Returns:
            Dictionary of model inputs
        """
        # Extract graph, positions, and box from candidates
        graphs = [c['graph'] for c in candidates]
        positions = [c['positions'] for c in candidates]
        boxes = [c['box'] for c in candidates]
        
        # Batch inputs
        # (Actual implementation would depend on batching strategy)
        inputs = {
            'graph': graphs[0],  # Placeholder
            'positions': positions[0],  # Placeholder
            'box': boxes[0],  # Placeholder
        }
        
        return inputs


# Acquisition functions

def expected_improvement(
    means: jnp.ndarray,
    uncertainties: jnp.ndarray,
    best_value: float,
    minimize: bool = True,
    xi: float = 0.01
) -> jnp.ndarray:
    """
    Expected Improvement acquisition function.
    
    Args:
        means: Predicted mean values
        uncertainties: Prediction uncertainties
        best_value: Current best value
        minimize: Whether the objective should be minimized
        xi: Exploration parameter
        
    Returns:
        Expected improvement scores
    """
    if minimize:
        improvement = best_value - means - xi
    else:
        improvement = means - best_value - xi
    
    # Handle zero uncertainty case
    z = jnp.where(
        uncertainties > 0,
        improvement / uncertainties,
        jnp.where(improvement > 0, jnp.inf, -jnp.inf)
    )
    
    # Calculate expected improvement
    normal = jax.scipy.stats.norm
    ei = uncertainties * (z * normal.cdf(z) + normal.pdf(z))
    
    # Ensure non-negative
    ei = jnp.maximum(ei, 0.0)
    
    return ei


def upper_confidence_bound(
    means: jnp.ndarray,
    uncertainties: jnp.ndarray,
    minimize: bool = True,
    beta: float = 2.0
) -> jnp.ndarray:
    """
    Upper Confidence Bound acquisition function.
    
    Args:
        means: Predicted mean values
        uncertainties: Prediction uncertainties
        minimize: Whether the objective should be minimized
        beta: Exploration-exploitation tradeoff parameter
        
    Returns:
        UCB scores
    """
    if minimize:
        return means - beta * uncertainties
    else:
        return means + beta * uncertainties


def entropy_search(
    means: jnp.ndarray,
    uncertainties: jnp.ndarray,
    model_state: Any,
    dataset: Dict,
    rng_key: jnp.ndarray
) -> jnp.ndarray:
    """
    Entropy Search acquisition function.
    
    Args:
        means: Predicted mean values
        uncertainties: Prediction uncertainties
        model_state: Current model state
        dataset: Current dataset
        rng_key: JAX random key
        
    Returns:
        Entropy search scores
    """
    # Placeholder - would implement entropy search
    # This is more complex as it requires estimating
    # the expected information gain
    
    # Return random scores for now
    return jax.random.uniform(rng_key, means.shape)


def target_property_acquisition(
    means: jnp.ndarray,
    uncertainties: jnp.ndarray,
    target_properties: Dict[str, Tuple[float, float]]
) -> jnp.ndarray:
    """
    Acquisition function targeting specific property ranges.
    
    Args:
        means: Predicted mean values (batch_size, n_properties)
        uncertainties: Prediction uncertainties (batch_size, n_properties)
        target_properties: Dictionary mapping property indices to (min, max) tuples
        
    Returns:
        Acquisition scores (batch_size,)
    """
    batch_size, n_properties = means.shape
    
    # Initialize scores
    scores = jnp.zeros(batch_size)
    
    # For each target property
    for idx, (prop_min, prop_max) in target_properties.items():
        property_means = means[:, idx]
        property_uncertainties = uncertainties[:, idx]
        
        # Calculate probability of being in target range
        prob_in_range = probability_in_range(
            property_means, property_uncertainties, prop_min, prop_max
        )
        
        # Add to scores (weighted by uncertainty to promote exploration)
        scores += prob_in_range * property_uncertainties[:, 0]
    
    return scores


def probability_in_range(
    means: jnp.ndarray,
    uncertainties: jnp.ndarray,
    lower: float,
    upper: float
) -> jnp.ndarray:
    """
    Calculate probability of value falling within target range.
    
    Args:
        means: Predicted mean values
        uncertainties: Prediction uncertainties
        lower: Lower bound of target range
        upper: Upper bound of target range
        
    Returns:
        Probability of being in range for each prediction
    """
    normal = jax.scipy.stats.norm
    
    # Standardize range boundaries
    lower_z = (lower - means) / uncertainties
    upper_z = (upper - means) / uncertainties
    
    # Calculate probability: cdf(upper_z) - cdf(lower_z)
    prob = normal.cdf(upper_z) - normal.cdf(lower_z)
    
    return prob


def expected_hypervolume_improvement(
    means: jnp.ndarray,
    uncertainties: jnp.ndarray,
    pareto_front: jnp.ndarray,
    reference_point: jnp.ndarray,
    maximize: List[bool] = None,
    num_samples: int = 100
) -> jnp.ndarray:
    """
    Expected Hypervolume Improvement for multi-objective optimization.
    
    Args:
        means: Predicted mean values (batch_size, n_objectives)
        uncertainties: Prediction uncertainties (batch_size, n_objectives)
        pareto_front: Current Pareto front (n_pareto, n_objectives)
        reference_point: Reference point for hypervolume calculation
        maximize: List indicating which objectives to maximize
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Expected hypervolume improvement scores
    """
    batch_size, n_objectives = means.shape
    
    if maximize is None:
        maximize = [True] * n_objectives
    
    # Convert to numpy for easier manipulation
    means_np = np.array(means)
    uncertainties_np = np.array(uncertainties)
    pareto_front_np = np.array(pareto_front)
    reference_point_np = np.array(reference_point)
    
    # For objectives to minimize, negate values
    for i, max_flag in enumerate(maximize):
        if not max_flag:
            means_np[:, i] = -means_np[:, i]
            pareto_front_np[:, i] = -pareto_front_np[:, i]
            reference_point_np[i] = -reference_point_np[i]
    
    # Calculate current hypervolume
    current_hv = _compute_hypervolume(pareto_front_np, reference_point_np)
    
    # Initialize scores
    ehvi_scores = np.zeros(batch_size)
    
    # For each candidate
    for i in range(batch_size):
        # Monte Carlo sampling
        samples = np.random.normal(
            loc=means_np[i], 
            scale=uncertainties_np[i], 
            size=(num_samples, n_objectives)
        )
        
        # Calculate hypervolume improvement for each sample
        hv_improvements = np.zeros(num_samples)
        
        for j, sample in enumerate(samples):
            # Create new Pareto front with this sample
            new_front = _update_pareto_front(pareto_front_np, sample)
            
            # Calculate new hypervolume
            new_hv = _compute_hypervolume(new_front, reference_point_np)
            
            # Calculate improvement
            hv_improvements[j] = max(0, new_hv - current_hv)
        
        # Expected improvement is average across samples
        ehvi_scores[i] = np.mean(hv_improvements)
    
    return jnp.array(ehvi_scores)


def _compute_hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume dominated by a Pareto front.
    
    Args:
        pareto_front: Pareto front points (n_points, n_objectives)
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    # This is a simplified implementation - would use specialized
    # libraries like pygmo or pymoo in practice
    
    # For 2D case, we can calculate directly
    if pareto_front.shape[1] == 2:
        # Sort points by first objective
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        
        # Calculate hypervolume
        hv = 0.0
        prev_x = reference_point[0]
        
        for point in sorted_front:
            if point[1] > reference_point[1]:  # Only consider points dominating reference
                hv += (point[0] - prev_x) * (point[1] - reference_point[1])
                prev_x = point[0]
        
        return hv
    
    # For higher dimensions, return placeholder
    # (would implement proper algorithm)
    return 0.0


def _update_pareto_front(pareto_front: np.ndarray, new_point: np.ndarray) -> np.ndarray:
    """
    Update Pareto front with a new point.
    
    Args:
        pareto_front: Current Pareto front
        new_point: New point to consider
        
    Returns:
        Updated Pareto front
    """
    # Check if new point is dominated by any existing point
    for point in pareto_front:
        if np.all(point >= new_point) and np.any(point > new_point):
            # New point is dominated, return unchanged front
            return pareto_front
    
    # Remove points dominated by the new point
    non_dominated = []
    for point in pareto_front:
        if not (np.all(new_point >= point) and np.any(new_point > point)):
            non_dominated.append(point)
    
    # Add new point to Pareto front
    if len(non_dominated) > 0:
        return np.vstack([non_dominated, new_point])
    else:
        return np.array([new_point])