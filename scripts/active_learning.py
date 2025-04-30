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

"""Property-guided active learning for materials discovery."""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Any, Optional
import time
import logging

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("active_learning.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Property-guided active learning for materials discovery')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    # Active learning parameters
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of active learning iterations')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation per iteration')
    parser.add_argument('--candidate_pool_size', type=int, default=1000, 
                        help='Number of candidates to sample per iteration')
    
    # Property optimization parameters
    parser.add_argument('--properties', type=str, nargs='+', default=['Formation Energy Per Atom'],
                      help='Properties to optimize')
    parser.add_argument('--property_weights', type=float, nargs='+', default=[1.0],
                      help='Weights for each property')
    parser.add_argument('--property_types', type=str, nargs='+', default=['minimize'],
                      help='Type for each property (maximize, minimize, target, range)')
    parser.add_argument('--property_targets', type=str, nargs='+', default=None,
                      help='Target values or ranges for each property (if applicable)')
    parser.add_argument('--stability_threshold', type=float, default=0.0,
                      help='Threshold for stability (hull distance)')
    
    # Model and uncertainty parameters
    parser.add_argument('--bayesian', action='store_true', help='Use Bayesian model for uncertainty')
    parser.add_argument('--mc_dropout', action='store_true', help='Use MC dropout for uncertainty')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for uncertainty estimation')
    parser.add_argument('--beta', type=float, default=2.0, help='Exploration parameter for UCB')
    parser.add_argument('--physics_prior', action='store_true', help='Use physics-informed prior')
    
    # Multi-objective optimization
    parser.add_argument('--multi_objective', action='store_true', help='Use multi-objective optimization')
    
    return parser.parse_args()

def load_data(data_dir: str) -> pd.DataFrame:
    """Load GNoME dataset.
    
    Args:
        data_dir: Path to data directory.
        
    Returns:
        DataFrame containing materials data.
    """
    # Load summary CSV
    summary_path = os.path.join(data_dir, 'gnome_data', 'stable_materials_summary.csv')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Data file not found: {summary_path}")
    
    logging.info(f"Loading data from {summary_path}")
    data = pd.read_csv(summary_path)
    logging.info(f"Loaded {len(data)} materials")
    
    return data

def load_model(model_dir: str, args):
    """Load Bayesian GNoME model.
    
    Args:
        model_dir: Path to model directory.
        args: Command-line arguments.
        
    Returns:
        Tuple of (model, params, config).
    """
    # Import inside function to avoid import errors if JAX isn't available
    from model.gnome import load_model as load_base_model
    
    logging.info(f"Loading model from {model_dir}")
    
    # Load base model
    try:
        cfg, base_model, base_params = load_base_model(model_dir)
        logging.info("Base model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading base model: {e}")
        raise
    
    # Create Bayesian model if requested
    if args.bayesian:
        try:
            from model.bayesian_gnome import BayesianGNoME
            model = BayesianGNoME(
                base_model=base_model,
                num_samples=args.num_samples,
                dropout_rate=0.1 if args.mc_dropout else 0.0
            )
            params = base_params  # Would need to convert to Bayesian params in practice
            logging.info("Created Bayesian model")
        except Exception as e:
            logging.error(f"Error creating Bayesian model: {e}")
            raise
    else:
        model = base_model
        params = base_params
        logging.info("Using base model (no uncertainty)")
    
    # Initialize physics-informed prior if requested
    if args.physics_prior:
        try:
            from model.physics_constraints import PhysicsInformedPrior
            physics_prior = PhysicsInformedPrior()
            logging.info("Initialized physics-informed prior")
        except Exception as e:
            logging.error(f"Error initializing physics-informed prior: {e}")
            physics_prior = None
    else:
        physics_prior = None
    
    return model, params, cfg, physics_prior

def prepare_property_targets(args):
    """Prepare property targets from command-line arguments.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Tuple of (property_types, property_targets) dictionaries.
    """
    # Check input lengths
    if len(args.property_weights) != len(args.properties):
        raise ValueError("Number of property weights must match number of properties")
    
    if len(args.property_types) != len(args.properties):
        raise ValueError("Number of property types must match number of properties")
    
    if args.property_targets and len(args.property_targets) != len(args.properties):
        raise ValueError("Number of property targets must match number of properties")
    
    # Create dictionaries
    property_types = {prop: type_str for prop, type_str in zip(args.properties, args.property_types)}
    property_targets = {}
    
    if args.property_targets:
        for prop, target_str, type_str in zip(args.properties, args.property_targets, args.property_types):
            if type_str.lower() == 'range':
                # Parse range as min,max
                try:
                    lower, upper = map(float, target_str.split(','))
                    property_targets[prop] = (lower, upper)
                except Exception as e:
                    logging.error(f"Error parsing range target for {prop}: {e}")
                    raise ValueError(f"Range target for {prop} must be 'min,max'")
            elif type_str.lower() == 'target':
                # Parse single target value
                try:
                    property_targets[prop] = float(target_str)
                except Exception as e:
                    logging.error(f"Error parsing target for {prop}: {e}")
                    raise ValueError(f"Target for {prop} must be a number")
    
    return property_types, property_targets

def generate_candidates(data: pd.DataFrame, 
                      num_candidates: int = 1000,
                      seed: int = None) -> pd.DataFrame:
    """Generate candidate materials for evaluation.
    
    In a real implementation, this would use SAPS and other generation methods.
    For this example, we'll just sample from the existing dataset.
    
    Args:
        data: DataFrame of materials data.
        num_candidates: Number of candidates to generate.
        seed: Random seed.
        
    Returns:
        DataFrame of candidate materials.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample random candidates from data
    # In a real implementation, this would use SAPS and other generation methods
    candidates = data.sample(min(num_candidates, len(data)))
    
    logging.info(f"Generated {len(candidates)} candidate materials")
    
    return candidates

def predict_properties(model, params, candidates, cfg, args):
    """Predict properties for candidate materials.
    
    Args:
        model: Model for prediction.
        params: Model parameters.
        candidates: DataFrame of candidate materials.
        cfg: Model configuration.
        args: Command-line arguments.
        
    Returns:
        Tuple of (means, uncertainties) dictionaries.
    """
    # This is a placeholder - in a real implementation, would convert
    # candidates to graph inputs and run the model
    
    # Simulate predictions
    num_candidates = len(candidates)
    
    # Get property names
    properties = args.properties
    
    # Initialize prediction arrays
    means = {}
    uncertainties = {}
    
    # Simulate predictions for each property
    for prop in properties:
        if prop in candidates.columns:
            # Use actual values plus noise for demonstration
            true_values = candidates[prop].values
            means[prop] = true_values + np.random.normal(0, 0.05, size=num_candidates)
            
            if args.bayesian or args.mc_dropout:
                # Simulate uncertainties higher for extreme values
                uncertainties[prop] = 0.1 + 0.1 * np.abs(true_values - np.mean(true_values)) / np.std(true_values)
            else:
                uncertainties[prop] = np.ones(num_candidates) * 0.1
        else:
            # Simulate random predictions
            means[prop] = np.random.normal(0, 1, size=num_candidates)
            uncertainties[prop] = np.ones(num_candidates) * 0.2
    
    # Add stability prediction if not in properties
    if 'stability' not in means and 'Decomposition Energy Per Atom' in candidates.columns:
        means['stability'] = candidates['Decomposition Energy Per Atom'].values
        uncertainties['stability'] = np.ones(num_candidates) * 0.05
    
    logging.info(f"Generated predictions for {num_candidates} candidates")
    
    return means, uncertainties

def select_candidates(means, uncertainties, args, pareto_front=None):
    """Select candidates using acquisition function.
    
    Args:
        means: Dictionary of predicted means.
        uncertainties: Dictionary of predicted uncertainties.
        args: Command-line arguments.
        pareto_front: Current Pareto front.
        
    Returns:
        Array of indices of selected candidates.
    """
    from model.acquisition_functions import MaterialPropertyAcquisition, PropertyTargetType
    
    # Prepare property types and targets
    property_types, property_targets = prepare_property_targets(args)
    
    # Convert property types to enum values
    property_types_enum = {}
    for prop, type_str in property_types.items():
        if type_str.lower() == 'maximize':
            property_types_enum[prop] = PropertyTargetType.MAXIMIZE
        elif type_str.lower() == 'minimize':
            property_types_enum[prop] = PropertyTargetType.MINIMIZE
        elif type_str.lower() == 'target':
            property_types_enum[prop] = PropertyTargetType.TARGET
        elif type_str.lower() == 'range':
            property_types_enum[prop] = PropertyTargetType.RANGE
    
    # Create property weights dictionary
    property_weights = {prop: weight for prop, weight in zip(args.properties, args.property_weights)}
    
    # Initialize acquisition function
    acquisition_fn = MaterialPropertyAcquisition(
        property_weights=property_weights,
        property_types=property_types_enum,
        property_targets=property_targets,
        stability_threshold=args.stability_threshold,
        beta=args.beta
    )
    
    # Convert means and uncertainties to arrays
    means_arrays = {prop: np.array(values) for prop, values in means.items()}
    uncertainties_arrays = {prop: np.array(values) for prop, values in uncertainties.items()}
    
    # Calculate acquisition scores
    scores = acquisition_fn(
        means_arrays,
        uncertainties_arrays,
        pareto_front=pareto_front,
        multi_objective=args.multi_objective
    )
    
    # Convert to numpy for sorting
    if isinstance(scores, jnp.ndarray):
        scores = np.array(scores)
    
    # Select top candidates
    batch_size = min(args.batch_size, len(scores))
    selected_indices = np.argsort(-scores)[:batch_size]
    
    logging.info(f"Selected {len(selected_indices)} candidates for evaluation")
    
    return selected_indices

def evaluate_candidates(candidates, selected_indices):
    """Evaluate selected candidates with DFT (simulated for this example).
    
    Args:
        candidates: DataFrame of candidate materials.
        selected_indices: Indices of selected candidates.
        
    Returns:
        DataFrame of evaluated candidates.
    """
    # In a real implementation, this would run DFT calculations
    # For this example, we'll just use the true values from the dataset
    
    evaluated = candidates.iloc[selected_indices].copy()
    
    # Add evaluation time
    evaluated['evaluation_time'] = time.time()
    
    logging.info(f"Evaluated {len(evaluated)} candidates")
    
    return evaluated

def update_pareto_front(evaluated, properties, property_types):
    """Update the Pareto front with newly evaluated materials.
    
    Args:
        evaluated: DataFrame of evaluated materials.
        properties: List of property names.
        property_types: Dictionary of property types.
        
    Returns:
        List of dictionaries representing the Pareto front.
    """
    # Extract property values
    points = []
    
    for _, row in evaluated.iterrows():
        point = {prop: row[prop] if prop in row else np.nan for prop in properties}
        point['id'] = row.name  # Use index as ID
        points.append(point)
    
    # Filter out points with missing values
    points = [p for p in points if not any(np.isnan(v) for k, v in p.items() if k != 'id')]
    
    # Determine dominance for each pair of points
    non_dominated = []
    
    for i, point_i in enumerate(points):
        dominated = False
        
        for j, point_j in enumerate(points):
            if i == j:
                continue
            
            # Check if point_j dominates point_i
            dominates = True
            strictly_better = False
            
            for prop in properties:
                if prop not in point_i or prop not in point_j:
                    continue
                
                # Get values
                value_i = point_i[prop]
                value_j = point_j[prop]
                
                # Check dominance based on property type
                prop_type = property_types.get(prop, 'maximize').lower()
                
                if prop_type == 'maximize':
                    if value_j < value_i:
                        dominates = False
                        break
                    if value_j > value_i:
                        strictly_better = True
                elif prop_type == 'minimize':
                    if value_j > value_i:
                        dominates = False
                        break
                    if value_j < value_i:
                        strictly_better = True
                elif prop_type == 'target':
                    # For target, closer is better
                    if abs(value_j) > abs(value_i):
                        dominates = False
                        break
                    if abs(value_j) < abs(value_i):
                        strictly_better = True
                elif prop_type == 'range':
                    # For range, being in range is binary
                    # This is a simplification
                    in_range_i = 0 <= value_i <= 1
                    in_range_j = 0 <= value_j <= 1
                    
                    if in_range_i and not in_range_j:
                        dominates = False
                        break
                    if in_range_j and not in_range_i:
                        strictly_better = True
            
            if dominates and strictly_better:
                dominated = True
                break
        
        if not dominated:
            non_dominated.append(point_i)
    
    logging.info(f"Updated Pareto front with {len(non_dominated)} non-dominated points")
    
    return non_dominated

def save_results(iteration, candidates, selected, evaluated, pareto_front, output_dir):
    """Save results from current iteration.
    
    Args:
        iteration: Current iteration number.
        candidates: DataFrame of candidate materials.
        selected: Indices of selected candidates.
        evaluated: DataFrame of evaluated candidates.
        pareto_front: Current Pareto front.
        output_dir: Output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save iteration data
    iteration_dir = os.path.join(output_dir, f'iteration_{iteration}')
    os.makedirs(iteration_dir, exist_ok=True)
    
    # Save selected candidates
    evaluated.to_csv(os.path.join(iteration_dir, 'evaluated.csv'), index=False)
    
    # Save Pareto front
    with open(os.path.join(iteration_dir, 'pareto_front.json'), 'w') as f:
        json.dump(pareto_front, f, indent=2)
    
    # Save summary
    summary = {
        'iteration': iteration,
        'num_candidates': len(candidates),
        'num_selected': len(selected),
        'num_evaluated': len(evaluated),
        'num_pareto': len(pareto_front)
    }
    
    with open(os.path.join(iteration_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Saved results for iteration {iteration}")

def plot_results(properties, evaluated_all, pareto_front, output_dir):
    """Plot results of active learning.
    
    Args:
        properties: List of property names.
        evaluated_all: List of DataFrames of evaluated materials.
        pareto_front: Current Pareto front.
        output_dir: Output directory.
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Combine all evaluated materials
    evaluated_df = pd.concat(evaluated_all, ignore_index=True)
    
    # Plot progress over iterations
    plt.figure(figsize=(10, 6))
    
    for prop in properties:
        if prop not in evaluated_df.columns:
            continue
        
        # Calculate best value per iteration
        best_values = []
        
        for i in range(len(evaluated_all)):
            df_i = pd.concat(evaluated_all[:i+1], ignore_index=True)
            
            if prop in df_i.columns:
                best_values.append(df_i[prop].min())  # Assuming minimization
            else:
                best_values.append(np.nan)
        
        plt.plot(range(1, len(evaluated_all) + 1), best_values, 'o-', label=prop)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Value')
    plt.title('Progress over Iterations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'progress.png'), dpi=300)
    
    # Plot Pareto front if multi-objective
    if len(properties) >= 2 and len(pareto_front) > 0:
        # Plot first two objectives
        prop1, prop2 = properties[:2]
        
        plt.figure(figsize=(10, 6))
        
        # Plot all evaluated points
        x = [row[prop1] for row in pareto_front if prop1 in row]
        y = [row[prop2] for row in pareto_front if prop2 in row]
        
        if x and y:
            plt.scatter(x, y, marker='o', label='Pareto Front')
        
        plt.xlabel(prop1)
        plt.ylabel(prop2)
        plt.title('Pareto Front')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'pareto_front.png'), dpi=300)
    
    logging.info(f"Saved plots to {plots_dir}")

def main():
    """Main function for active learning."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    data = load_data(args.data_dir)
    
    # Load model
    model, params, cfg, physics_prior = load_model(args.model_dir, args)
    
    # Prepare property types and targets
    property_types, property_targets = prepare_property_targets(args)
    
    # Initialize storage for active learning
    evaluated_all = []
    pareto_front = []
    
    # Active learning loop
    for iteration in range(args.num_iterations):
        logging.info(f"Starting iteration {iteration+1}/{args.num_iterations}")
        
        # Generate candidates
        candidates = generate_candidates(
            data, args.candidate_pool_size, seed=iteration)
        
        # Predict properties
        means, uncertainties = predict_properties(
            model, params, candidates, cfg, args)
        
        # Select candidates
        selected_indices = select_candidates(
            means, uncertainties, args, pareto_front)
        
        # Evaluate selected candidates
        evaluated = evaluate_candidates(candidates, selected_indices)
        evaluated_all.append(evaluated)
        
        # Update Pareto front
        pareto_front = update_pareto_front(
            pd.concat(evaluated_all, ignore_index=True),
            args.properties,
            property_types
        )
        
        # Save results
        save_results(
            iteration + 1,
            candidates,
            selected_indices,
            evaluated,
            pareto_front,
            args.output_dir
        )
        
        logging.info(f"Completed iteration {iteration+1}/{args.num_iterations}")
    
    # Plot results
    plot_results(args.properties, evaluated_all, pareto_front, args.output_dir)
    
    logging.info("Active learning completed successfully")

if __name__ == "__main__":
    main()