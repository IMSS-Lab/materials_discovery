"""
Uncertainty quantification methods for materials discovery.

This module provides utilities for quantifying uncertainty in
material property predictions, combining both aleatoric uncertainty
(due to inherent randomness) and epistemic uncertainty (due to model limitations).
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

from .util import Array


def predictive_uncertainty(mean: Array, variance: Array) -> Array:
    """
    Compute total predictive uncertainty from mean and variance.
    
    Args:
        mean: Mean predictions (N, D)
        variance: Variance predictions (N, D)
        
    Returns:
        Total predictive uncertainty (N, D)
    """
    return jnp.sqrt(variance)


def epistemic_uncertainty(model_samples: Array) -> Tuple[Array, Array]:
    """
    Compute epistemic uncertainty from multiple model predictions.
    
    Args:
        model_samples: Samples from different models or model parameters (M, N, D)
                      where M is the number of models/samples,
                      N is the number of examples, and D is the output dimension
        
    Returns:
        Tuple of (mean, epistemic_uncertainty)
    """
    mean = jnp.mean(model_samples, axis=0)
    variance = jnp.var(model_samples, axis=0)
    
    return mean, jnp.sqrt(variance)


def aleatoric_uncertainty(model: Callable, params: Dict, 
                         inputs: Dict, rng_key: Array, 
                         num_samples: int = 10) -> Tuple[Array, Array]:
    """
    Compute aleatoric uncertainty via Monte Carlo dropout or noise sampling.
    
    Args:
        model: Model function
        params: Model parameters
        inputs: Model inputs
        rng_key: Random key for sampling
        num_samples: Number of forward passes
        
    Returns:
        Tuple of (mean, aleatoric_uncertainty)
    """
    keys = jax.random.split(rng_key, num_samples)
    
    # Perform multiple forward passes with different random seeds
    samples = []
    for i in range(num_samples):
        outputs = model.apply(
            {'params': params},
            **inputs,
            rng_key=keys[i],
            training=False
        )
        samples.append(outputs)
    
    # Stack samples
    samples = jnp.stack(samples)
    
    # Compute mean and variance
    mean = jnp.mean(samples, axis=0)
    variance = jnp.var(samples, axis=0)
    
    return mean, jnp.sqrt(variance)


def combined_uncertainty(epistemic_var: Array, aleatoric_var: Array) -> Array:
    """
    Combine epistemic and aleatoric uncertainty.
    
    Args:
        epistemic_var: Epistemic variance
        aleatoric_var: Aleatoric variance
        
    Returns:
        Combined standard deviation
    """
    # Total variance is the sum of epistemic and aleatoric variances
    total_var = epistemic_var + aleatoric_var
    
    return jnp.sqrt(total_var)


def uncertainty_decomposition(model: Callable, params: Dict, 
                             inputs: Dict, rng_key: Array,
                             num_models: int = 5, 
                             num_samples_per_model: int = 10) -> Dict[str, Array]:
    """
    Decompose uncertainty into epistemic and aleatoric components.
    
    Args:
        model: Model function
        params: Model parameters (or list of parameters for ensemble)
        inputs: Model inputs
        rng_key: Random key for sampling
        num_models: Number of models in ensemble
        num_samples_per_model: Number of stochastic forward passes per model
        
    Returns:
        Dictionary with mean, epistemic_uncertainty, aleatoric_uncertainty, and total_uncertainty
    """
    keys = jax.random.split(rng_key, num_models)
    
    # Get predictions from multiple models/parameter sets
    model_means = []
    model_vars = []
    
    for i in range(num_models):
        if isinstance(params, list):
            # Ensemble of models with different parameters
            model_params = params[i]
        else:
            # Single model with parameter perturbations
            model_params = perturb_parameters(params, keys[i])
        
        # Get aleatoric uncertainty from this model
        sub_key = jax.random.split(keys[i], 1)[0]
        mean, aleatoric_std = aleatoric_uncertainty(
            model, model_params, inputs, sub_key, num_samples_per_model
        )
        
        model_means.append(mean)
        model_vars.append(aleatoric_std ** 2)
    
    # Stack means and variances
    model_means = jnp.stack(model_means)
    model_vars = jnp.stack(model_vars)
    
    # Compute statistics
    mean = jnp.mean(model_means, axis=0)
    
    # Epistemic uncertainty: variance of means
    epistemic_var = jnp.var(model_means, axis=0)
    
    # Aleatoric uncertainty: mean of variances
    aleatoric_var = jnp.mean(model_vars, axis=0)
    
    # Total uncertainty
    total_var = epistemic_var + aleatoric_var
    
    return {
        'mean': mean,
        'epistemic_uncertainty': jnp.sqrt(epistemic_var),
        'aleatoric_uncertainty': jnp.sqrt(aleatoric_var),
        'total_uncertainty': jnp.sqrt(total_var)
    }


def perturb_parameters(params: Dict, rng_key: Array, 
                      scale: float = 0.01) -> Dict:
    """
    Perturb model parameters to create a new parameter set.
    
    Args:
        params: Original model parameters
        rng_key: Random key for noise generation
        scale: Scale of perturbation relative to parameter magnitude
        
    Returns:
        Perturbed parameters
    """
    # Flatten parameters for easier manipulation
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(rng_key, len(flat_params))
    
    # Perturb each parameter array
    perturbed_flat_params = []
    for p, key in zip(flat_params, keys):
        if p is not None:
            noise = scale * jnp.abs(p) * jax.random.normal(key, p.shape)
            perturbed_flat_params.append(p + noise)
        else:
            perturbed_flat_params.append(None)
    
    # Unflatten back to original structure
    return jax.tree_util.tree_unflatten(tree_def, perturbed_flat_params)


def calibration_error(predictions: Array, uncertainties: Array, 
                     targets: Array, num_bins: int = 10) -> float:
    """
    Compute calibration error to evaluate uncertainty quality.
    
    Args:
        predictions: Mean predictions (N,)
        uncertainties: Uncertainty estimates (N,)
        targets: Ground truth values (N,)
        num_bins: Number of bins for calibration curve
        
    Returns:
        Expected calibration error
    """
    # Compute normalized errors
    normalized_errors = jnp.abs(predictions - targets) / uncertainties
    
    # Create bins
    bin_boundaries = jnp.linspace(0, jnp.max(normalized_errors), num_bins + 1)
    bin_indices = jnp.digitize(normalized_errors, bin_boundaries) - 1
    bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)
    
    # Compute fraction of points in each bin
    bin_counts = jnp.bincount(bin_indices, minlength=num_bins)
    bin_fractions = bin_counts / jnp.sum(bin_counts)
    
    # Compute expected fraction for each bin
    expected_fractions = jnp.array([
        jnp.mean(normalized_errors <= boundary)
        for boundary in bin_boundaries[1:]
    ])
    
    # Compute calibration error (weighted average of absolute differences)
    calibration_error = jnp.sum(
        bin_fractions * jnp.abs(bin_fractions - expected_fractions)
    )
    
    return calibration_error


def compute_confidence_interval(mean: Array, std: Array, 
                              confidence_level: float = 0.95) -> Tuple[Array, Array]:
    """
    Compute confidence interval for predictions assuming Gaussian distribution.
    
    Args:
        mean: Mean predictions
        std: Standard deviation of predictions
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Z-score for the given confidence level
    if confidence_level == 0.68:
        z = 1.0
    elif confidence_level == 0.95:
        z = 1.96
    elif confidence_level == 0.99:
        z = 2.58
    else:
        # Approximate z-score for arbitrary confidence level
        z = jnp.sqrt(2) * jax.scipy.special.erfinv(confidence_level)
    
    lower_bound = mean - z * std
    upper_bound = mean + z * std
    
    return lower_bound, upper_bound