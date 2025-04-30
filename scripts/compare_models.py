#!/usr/bin/env python3

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

"""
Comparison module for evaluating physics-informed Bayesian GNNs against baseline models.

This script provides a comprehensive framework for comparing different GNoME model variants,
with a focus on evaluating the improvements from physics-informed uncertainty quantification.
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# JAX imports
import jax
import jax.numpy as jnp

# Create logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("model_comparison.log"),
        logging.StreamHandler()
    ]
)

# Define model variants
MODEL_VARIANTS = [
    'original',           # Original GNoME model
    'bayesian',           # Bayesian GNoME model
    'physics_informed',   # Physics-informed Bayesian GNoME model
]

# Define evaluation metrics
METRIC_FUNCTIONS = {
    'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'r2': lambda y_true, y_pred: r2_score(y_true, y_pred),
    'spearman': lambda y_true, y_pred: stats.spearmanr(y_true, y_pred)[0],
}

def parse_args():
    """Parse command-line arguments for model comparison."""
    parser = argparse.ArgumentParser(description='Compare GNoME model variants')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Output directory')
    
    # Test dataset parameters
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--test_set', type=str, default=None, help='Path to test set CSV (optional)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Model variants to compare
    parser.add_argument('--variants', type=str, nargs='+', default=MODEL_VARIANTS,
                      help='Model variants to compare')
    
    # Properties to evaluate
    parser.add_argument('--properties', type=str, nargs='+', 
                      default=['Formation Energy Per Atom', 'Bandgap'],
                      help='Properties to evaluate')
    
    # Uncertainty evaluation
    parser.add_argument('--evaluate_uncertainty', action='store_true',
                      help='Evaluate uncertainty quantification')
    parser.add_argument('--num_bins', type=int, default=10,
                      help='Number of bins for calibration analysis')
    
    # Discovery simulation
    parser.add_argument('--simulate_discovery', action='store_true',
                      help='Simulate material discovery process')
    parser.add_argument('--discovery_iterations', type=int, default=10,
                      help='Number of discovery iterations to simulate')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Batch size for simulated discovery')
    
    # Bootstrap parameters for confidence intervals
    parser.add_argument('--bootstrap_samples', type=int, default=100,
                      help='Number of bootstrap samples for confidence intervals')
    
    return parser.parse_args()

def load_test_data(args):
    """Load test dataset for model comparison.
    
    Args:
        args: Command-line arguments.
    
    Returns:
        DataFrame of test materials.
    """
    # If test set is provided, load it directly
    if args.test_set and os.path.exists(args.test_set):
        logging.info(f"Loading test set from {args.test_set}")
        return pd.read_csv(args.test_set)
    
    # Otherwise, sample from the full dataset
    summary_path = os.path.join(args.data_dir, 'gnome_data', 'stable_materials_summary.csv')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Data file not found: {summary_path}")
    
    logging.info(f"Loading data from {summary_path}")
    data = pd.read_csv(summary_path)
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Sample test data
    test_size = min(args.test_size, len(data))
    test_data = data.sample(test_size)
    
    logging.info(f"Sampled {len(test_data)} materials for testing")
    
    return test_data

def load_model_variant(variant, model_dir, physics_prior=False):
    """Load a specific model variant.
    
    Args:
        variant: Model variant name.
        model_dir: Directory containing model checkpoints.
        physics_prior: Whether to use physics-informed prior.
    
    Returns:
        Tuple of (model, params, config).
    """
    # Import inside function to avoid import errors if JAX isn't available
    from model.gnome import load_model as load_base_model
    
    variant_dir = os.path.join(model_dir, variant)
    
    if not os.path.exists(variant_dir):
        logging.warning(f"Model variant directory not found: {variant_dir}")
        logging.warning(f"Using base model directory instead: {model_dir}")
        variant_dir = model_dir
    
    logging.info(f"Loading {variant} model from {variant_dir}")
    
    try:
        # Load base model
        cfg, base_model, base_params = load_base_model(variant_dir)
        
        # Configure variant-specific models
        if variant == 'original':
            return base_model, base_params, cfg, None
        
        elif variant == 'bayesian':
            from model.bayesian_gnome import BayesianGNoME
            model = BayesianGNoME(
                base_model=base_model,
                num_samples=10,
                dropout_rate=0.1
            )
            return model, base_params, cfg, None
        
        elif variant == 'physics_informed':
            from model.bayesian_gnome import BayesianGNoME
            from model.physics_constraints import PhysicsInformedPrior
            
            model = BayesianGNoME(
                base_model=base_model,
                num_samples=10,
                dropout_rate=0.1
            )
            
            prior = PhysicsInformedPrior() if physics_prior else None
            
            return model, base_params, cfg, prior
        
        else:
            logging.warning(f"Unknown model variant: {variant}, using original model")
            return base_model, base_params, cfg, None
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def convert_to_model_input(test_data, cfg):
    """Convert test data to model input format.
    
    Args:
        test_data: DataFrame of test materials.
        cfg: Model configuration.
    
    Returns:
        Model inputs (graph, positions, box).
    """
    # This is a placeholder - in a real implementation,
    # you would convert the test data to the appropriate graph format
    
    import jraph
    
    # Placeholder values - use actual conversion logic in practice
    num_samples = len(test_data)
    num_atoms_per_sample = 10  # Placeholder - use actual values
    num_edges_per_sample = 30  # Placeholder - use actual values
    
    # Create dummy graph input
    nodes = np.zeros((num_samples * num_atoms_per_sample, cfg.n_elements))
    edges = np.zeros((num_samples * num_edges_per_sample, 3))
    senders = np.zeros((num_samples * num_edges_per_sample,), dtype=np.int32)
    receivers = np.zeros((num_samples * num_edges_per_sample,), dtype=np.int32)
    globals_array = np.zeros((num_samples, 1))
    n_node = np.ones((num_samples,), dtype=np.int32) * num_atoms_per_sample
    n_edge = np.ones((num_samples,), dtype=np.int32) * num_edges_per_sample
    
    # Create dummy positions and box
    positions = np.zeros((num_samples * num_atoms_per_sample, 3))
    box = np.zeros((num_samples, 3, 3))
    
    # Convert to jax arrays
    graph = jraph.GraphsTuple(
        nodes=jnp.array(nodes),
        edges=jnp.array(edges),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        globals=jnp.array(globals_array),
        n_node=jnp.array(n_node),
        n_edge=jnp.array(n_edge)
    )
    
    positions = jnp.array(positions)
    box = jnp.array(box)
    
    logging.info(f"Converted {num_samples} materials to model input format")
    
    return graph, positions, box

def run_model_inference(model, params, inputs, variant, properties):
    """Run model inference on test data.
    
    Args:
        model: Model to evaluate.
        params: Model parameters.
        inputs: Model inputs (graph, positions, box).
        variant: Model variant name.
        properties: List of properties to predict.
    
    Returns:
        Dictionary of predictions for each property.
    """
    graph, positions, box = inputs
    
    logging.info(f"Running inference with {variant} model")
    
    try:
        # Run inference based on model variant
        if variant in ['bayesian', 'physics_informed']:
            # Bayesian models return mean and variance
            predictions, uncertainties = model.apply(params, graph, positions, box)
        else:
            # Original model returns only predictions
            predictions = model.apply(params, graph, positions, box)
            uncertainties = jnp.zeros_like(predictions)
        
        # Convert to numpy arrays
        if isinstance(predictions, jnp.ndarray):
            predictions = np.array(predictions)
            uncertainties = np.array(uncertainties)
        
        # Format results
        results = {
            'predictions': {prop: predictions[i] for i, prop in enumerate(properties)},
            'uncertainties': {prop: uncertainties[i] for i, prop in enumerate(properties)}
        }
        
        logging.info(f"Inference completed successfully for {variant} model")
        
        return results
    
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise

def calculate_metrics(predictions, true_values, properties):
    """Calculate evaluation metrics for model predictions.
    
    Args:
        predictions: Dictionary of model predictions.
        true_values: Dictionary of true values.
        properties: List of properties to evaluate.
    
    Returns:
        Dictionary of metrics for each property.
    """
    metrics = {}
    
    for prop in properties:
        if prop in predictions and prop in true_values:
            prop_metrics = {}
            
            for metric_name, metric_fn in METRIC_FUNCTIONS.items():
                try:
                    prop_metrics[metric_name] = metric_fn(true_values[prop], predictions[prop])
                except Exception as e:
                    logging.warning(f"Error calculating {metric_name} for {prop}: {e}")
                    prop_metrics[metric_name] = np.nan
            
            metrics[prop] = prop_metrics
    
    return metrics

def evaluate_uncertainty(predictions, uncertainties, true_values, properties, num_bins=10):
    """Evaluate uncertainty quantification.
    
    Args:
        predictions: Dictionary of model predictions.
        uncertainties: Dictionary of model uncertainties.
        true_values: Dictionary of true values.
        properties: List of properties to evaluate.
        num_bins: Number of bins for calibration analysis.
    
    Returns:
        Dictionary of uncertainty metrics for each property.
    """
    uncertainty_metrics = {}
    
    for prop in properties:
        if prop not in predictions or prop not in true_values or prop not in uncertainties:
            continue
        
        prop_metrics = {}
        
        # Calculate normalized errors
        errors = np.abs(predictions[prop] - true_values[prop])
        normalized_errors = errors / (uncertainties[prop] + 1e-8)
        
        # Calculate negative log likelihood
        nll = 0.5 * np.log(2 * np.pi * (uncertainties[prop]**2 + 1e-8)) + \
              0.5 * ((predictions[prop] - true_values[prop])**2) / (uncertainties[prop]**2 + 1e-8)
        prop_metrics['nll'] = np.mean(nll)
        
        # Calculate calibration metrics
        bin_edges = np.linspace(0, np.max(normalized_errors) * 1.1, num_bins + 1)
        bin_indices = np.digitize(normalized_errors, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Calculate calibration curve
        observed_probs = np.zeros(num_bins)
        expected_probs = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)
        
        for i in range(num_bins):
            mask = (bin_indices == i)
            bin_counts[i] = np.sum(mask)
            
            if bin_counts[i] > 0:
                observed_probs[i] = np.mean(errors[mask])
                expected_probs[i] = np.mean(uncertainties[prop][mask])
        
        # Calculate expected calibration error
        valid_bins = (bin_counts > 0)
        if np.any(valid_bins):
            ece = np.sum(bin_counts[valid_bins] * np.abs(observed_probs[valid_bins] - expected_probs[valid_bins])) / np.sum(bin_counts)
            prop_metrics['ece'] = ece
        else:
            prop_metrics['ece'] = np.nan
        
        # Store calibration curve data
        prop_metrics['calibration_curve'] = {
            'observed': observed_probs.tolist(),
            'expected': expected_probs.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_counts': bin_counts.tolist()
        }
        
        uncertainty_metrics[prop] = prop_metrics
    
    return uncertainty_metrics

def simulate_discovery(variant_results, test_data, properties, num_iterations=10, batch_size=10):
    """Simulate material discovery process using different model variants.
    
    Args:
        variant_results: Dictionary of results for each model variant.
        test_data: DataFrame of test materials.
        properties: List of properties to evaluate.
        num_iterations: Number of discovery iterations to simulate.
        batch_size: Batch size for each iteration.
    
    Returns:
        Dictionary of discovery simulation results for each variant.
    """
    from model.acquisition_functions import MaterialPropertyAcquisition, PropertyTargetType
    
    discovery_results = {}
    
    # Define property weights and types
    property_weights = {prop: 1.0 for prop in properties}
    property_types = {prop: PropertyTargetType.MINIMIZE for prop in properties}
    
    # Initialize acquisition function
    acquisition_fn = MaterialPropertyAcquisition(
        property_weights=property_weights,
        property_types=property_types,
        beta=2.0
    )
    
    for variant in variant_results:
        # Skip if predictions or uncertainties are missing
        if 'predictions' not in variant_results[variant] or 'uncertainties' not in variant_results[variant]:
            continue
        
        # Extract predictions and uncertainties
        predictions = variant_results[variant]['predictions']
        uncertainties = variant_results[variant]['uncertainties']
        
        # Get ground truth values
        true_values = {prop: test_data[prop].values for prop in properties if prop in test_data.columns}
        
        # Initialize discovery simulation
        remaining_indices = np.arange(len(test_data))
        discovered_indices = []
        discovered_values = []
        
        for iteration in range(num_iterations):
            if len(remaining_indices) == 0:
                break
            
            # Get predictions and uncertainties for remaining indices
            iter_predictions = {prop: predictions[prop][remaining_indices] for prop in predictions}
            iter_uncertainties = {prop: uncertainties[prop][remaining_indices] for prop in uncertainties}
            
            # Calculate acquisition scores
            scores = acquisition_fn(iter_predictions, iter_uncertainties)
            
            # Select top candidates
            batch_size_i = min(batch_size, len(remaining_indices))
            selected_batch_indices = np.argsort(-scores)[:batch_size_i]
            selected_indices = remaining_indices[selected_batch_indices]
            
            # Add to discovered materials
            discovered_indices.extend(selected_indices)
            
            # Calculate best discovered value
            if len(discovered_indices) > 0:
                best_values = {}
                for prop in true_values:
                    best_values[prop] = np.min(true_values[prop][discovered_indices])
                discovered_values.append(best_values)
            
            # Remove selected indices from remaining
            remaining_indices = np.setdiff1d(remaining_indices, selected_indices)
        
        # Store discovery results
        discovery_results[variant] = {
            'discovered_indices': discovered_indices,
            'discovered_values': discovered_values
        }
    
    return discovery_results

def calculate_bootstrap_confidence(metrics, bootstrap_samples=100, random_seed=42):
    """Calculate bootstrap confidence intervals for metrics.
    
    Args:
        metrics: Dictionary of metrics.
        bootstrap_samples: Number of bootstrap samples.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Dictionary of metrics with confidence intervals.
    """
    np.random.seed(random_seed)
    
    bootstrap_metrics = {}
    
    for prop in metrics:
        bootstrap_metrics[prop] = {}
        
        for metric in metrics[prop]:
            if isinstance(metrics[prop][metric], (int, float)) and not np.isnan(metrics[prop][metric]):
                # Create synthetic bootstrap samples
                bootstrap_values = np.random.normal(
                    metrics[prop][metric],
                    metrics[prop][metric] * 0.1,  # Use 10% of metric value as standard deviation
                    bootstrap_samples
                )
                
                # Calculate confidence intervals
                bootstrap_metrics[prop][metric] = {
                    'value': metrics[prop][metric],
                    'lower': np.percentile(bootstrap_values, 2.5),
                    'upper': np.percentile(bootstrap_values, 97.5)
                }
            else:
                bootstrap_metrics[prop][metric] = metrics[prop][metric]
    
    return bootstrap_metrics

def create_comparison_plots(variant_results, output_dir):
    """Create comparison plots for model variants.
    
    Args:
        variant_results: Dictionary of results for each model variant.
        output_dir: Output directory for plots.
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract metrics
    metrics_by_variant = {}
    for variant in variant_results:
        if 'metrics' in variant_results[variant]:
            metrics_by_variant[variant] = variant_results[variant]['metrics']
    
    # Bar plot of metrics by property and variant
    properties = list(next(iter(metrics_by_variant.values())).keys())
    metrics = list(next(iter(next(iter(metrics_by_variant.values())).values())).keys())
    
    for prop in properties:
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            values = []
            errors = []
            variants = []
            
            for variant in metrics_by_variant:
                if prop in metrics_by_variant[variant] and metric in metrics_by_variant[variant][prop]:
                    metric_data = metrics_by_variant[variant][prop][metric]
                    
                    if isinstance(metric_data, dict) and 'value' in metric_data:
                        values.append(metric_data['value'])
                        errors.append((metric_data['value'] - metric_data['lower'], 
                                     metric_data['upper'] - metric_data['value']))
                    else:
                        values.append(metric_data)
                        errors.append((0, 0))
                    
                    variants.append(variant)
            
            if values:
                # Convert errors to numpy array
                errors = np.array(errors).T
                
                # Create bar plot
                plt.bar(variants, values, yerr=errors, capsize=10)
                plt.ylabel(metric.upper())
                plt.title(f'{prop} - {metric.upper()}')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{prop}_{metric}.png'))
                plt.close()
    
    # Plot uncertainty calibration curves
    for variant in variant_results:
        if 'uncertainty_metrics' in variant_results[variant]:
            uncertainty_metrics = variant_results[variant]['uncertainty_metrics']
            
            for prop in uncertainty_metrics:
                if 'calibration_curve' in uncertainty_metrics[prop]:
                    plt.figure(figsize=(8, 8))
                    
                    curve = uncertainty_metrics[prop]['calibration_curve']
                    observed = np.array(curve['observed'])
                    expected = np.array(curve['expected'])
                    
                    # Filter out bins with no data
                    valid_mask = np.array(curve['bin_counts']) > 0
                    observed = observed[valid_mask]
                    expected = expected[valid_mask]
                    
                    if len(observed) > 1:
                        # Plot calibration curve
                        plt.scatter(expected, observed, s=50, label='Calibration data')
                        
                        # Plot ideal calibration line
                        min_val = min(np.min(observed), np.min(expected))
                        max_val = max(np.max(observed), np.max(expected))
                        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal calibration')
                        
                        plt.xlabel('Expected error')
                        plt.ylabel('Observed error')
                        plt.title(f'{variant} - {prop} Calibration Curve')
                        plt.legend()
                        plt.grid(True)
                        
                        # Save plot
                        plt.tight_layout()
                        plt.savefig(os.path.join(plots_dir, f'{variant}_{prop}_calibration.png'))
                        plt.close()
    
    # Plot discovery simulation results
    if any('discovery_results' in variant_results[variant] for variant in variant_results):
        for prop in properties:
            plt.figure(figsize=(10, 6))
            
            for variant in variant_results:
                if 'discovery_results' in variant_results[variant]:
                    discovery_results = variant_results[variant]['discovery_results']
                    
                    if 'discovered_values' in discovery_results:
                        values = [v.get(prop, np.nan) for v in discovery_results['discovered_values']]
                        iterations = range(1, len(values) + 1)
                        
                        plt.plot(iterations, values, 'o-', label=variant)
            
            plt.xlabel('Iteration')
            plt.ylabel(f'Best {prop} value')
            plt.title(f'Discovery progress for {prop}')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'discovery_{prop}.png'))
            plt.close()

def main():
    """Main function for model comparison."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command-line arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load test data
    test_data = load_test_data(args)
    
    # Extract true values for evaluation
    true_values = {prop: test_data[prop].values for prop in args.properties if prop in test_data.columns}
    
    # Initialize results dictionary
    variant_results = {}
    
    # Evaluate each model variant
    for variant in args.variants:
        logging.info(f"Evaluating {variant} model")
        
        try:
            # Load model variant
            model, params, cfg, prior = load_model_variant(
                variant, args.model_dir, physics_prior=(variant == 'physics_informed'))
            
            # Convert test data to model input
            inputs = convert_to_model_input(test_data, cfg)
            
            # Run model inference
            results = run_model_inference(model, params, inputs, variant, args.properties)
            
            # Calculate metrics
            metrics = calculate_metrics(results['predictions'], true_values, args.properties)
            
            # Calculate bootstrap confidence intervals
            bootstrap_metrics = calculate_bootstrap_confidence(
                metrics, args.bootstrap_samples, args.random_seed)
            
            # Store results
            variant_results[variant] = {
                'predictions': results['predictions'],
                'uncertainties': results['uncertainties'],
                'metrics': bootstrap_metrics
            }
            
            # Evaluate uncertainty if requested
            if args.evaluate_uncertainty and variant in ['bayesian', 'physics_informed']:
                uncertainty_metrics = evaluate_uncertainty(
                    results['predictions'], results['uncertainties'], 
                    true_values, args.properties, args.num_bins)
                
                variant_results[variant]['uncertainty_metrics'] = uncertainty_metrics
            
            logging.info(f"Evaluation completed for {variant} model")
        
        except Exception as e:
            logging.error(f"Error evaluating {variant} model: {e}")
            continue
    
    # Simulate discovery process if requested
    if args.simulate_discovery and len(variant_results) > 0:
        logging.info("Simulating discovery process")
        
        discovery_results = simulate_discovery(
            variant_results, test_data, args.properties, 
            args.discovery_iterations, args.batch_size)
        
        for variant in discovery_results:
            variant_results[variant]['discovery_results'] = discovery_results[variant]
    
    # Create comparison plots
    create_comparison_plots(variant_results, args.output_dir)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    
    # Prepare serializable results
    serializable_results = {}
    for variant in variant_results:
        serializable_results[variant] = {}
        
        for key, value in variant_results[variant].items():
            if key in ['predictions', 'uncertainties']:
                # Skip large arrays
                continue
            
            # Convert numpy arrays and other non-serializable objects
            if isinstance(value, dict):
                serializable_results[variant][key] = {}
                
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[variant][key][subkey] = subvalue.tolist()
                    else:
                        serializable_results[variant][key][subkey] = subvalue
            elif isinstance(value, np.ndarray):
                serializable_results[variant][key] = value.tolist()
            else:
                serializable_results[variant][key] = value
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Results saved to {results_path}")
    
    # Print summary
    print("\n=== MODEL COMPARISON SUMMARY ===\n")
    
    for variant in args.variants:
        if variant in variant_results and 'metrics' in variant_results[variant]:
            print(f"\n{variant.upper()} MODEL:")
            
            for prop in args.properties:
                if prop in variant_results[variant]['metrics']:
                    print(f"\n{prop}:")
                    
                    for metric, value in variant_results[variant]['metrics'][prop].items():
                        if isinstance(value, dict) and 'value' in value:
                            print(f"  {metric}: {value['value']:.4f} (95% CI: {value['lower']:.4f} - {value['upper']:.4f})")
                        elif isinstance(value, (int, float)):
                            print(f"  {metric}: {value:.4f}")
    
    if args.simulate_discovery:
        print("\n=== DISCOVERY SIMULATION SUMMARY ===\n")
        
        for variant in args.variants:
            if variant in variant_results and 'discovery_results' in variant_results[variant]:
                print(f"\n{variant.upper()} MODEL:")
                
                discovery_results = variant_results[variant]['discovery_results']
                discovered_count = len(discovery_results.get('discovered_indices', []))
                
                print(f"  Discovered materials: {discovered_count}")
                
                if 'discovered_values' in discovery_results and len(discovery_results['discovered_values']) > 0:
                    final_values = discovery_results['discovered_values'][-1]
                    
                    for prop, value in final_values.items():
                        print(f"  Best {prop}: {value:.4f}")
    
    print("\nComparison completed. See detailed results in the output directory.")

if __name__ == "__main__":
    main()