#!/usr/bin/env python
"""
Evaluate properties of candidate materials.

This script evaluates the properties of candidate materials using
DFT calculations and compares with model predictions.
"""

import argparse
import os
import time
import json
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from model.bayesian_gnn import BayesianGNN
from pipelines.dft_evaluation import VASPEvaluator, DFTBatchProcessor
from pipelines.uncertainty_propagation import UncertaintyPropagator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate material properties")
    
    # Input options
    parser.add_argument("--candidates_file", type=str, required=True,
                        help="File containing candidate materials")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model")
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Model checkpoint to use")
    
    # Evaluation options
    parser.add_argument("--properties", type=str, nargs="+", 
                        default=["formation_energy", "bandgap"],
                        help="Properties to evaluate")
    parser.add_argument("--max_candidates", type=int, default=100,
                        help="Maximum number of candidates to evaluate")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for evaluation")
    
    # DFT options
    parser.add_argument("--dft_method", type=str, default="vasp",
                        choices=["vasp", "pymatgen"],
                        help="DFT method to use")
    parser.add_argument("--vasp_cmd", type=str, default="vasp_std",
                        help="VASP command")
    parser.add_argument("--potcar_dir", type=str, default=None,
                        help="Directory containing VASP POTCARs")
    parser.add_argument("--dft_settings", type=str, default=None,
                        help="JSON file with DFT settings")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of parallel DFT calculations")
    
    # Uncertainty options
    parser.add_argument("--uncertainty_analysis", action="store_true",
                        help="Perform uncertainty analysis")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for uncertainty analysis")
    parser.add_argument("--calibration_plot", action="store_true",
                        help="Generate calibration plot")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                        help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")
    
    return parser.parse_args()


def load_model(args):
    """
    Load trained model from checkpoint.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (model, params)
    """
    # Load model configuration
    config_file = os.path.join(args.model_dir, "config.json")
    with open(config_file, "r") as f:
        model_config = json.load(f)
    
    # Create model instance
    model = BayesianGNN(
        graph_net_steps=model_config["graph_net_steps"],
        mlp_width=tuple(model_config["mlp_width"]),
        mlp_nonlinearity=model_config["mlp_nonlinearity"],
        embedding_dim=model_config["embedding_dim"],
        prior_scale=model_config.get("prior_scale", 1.0),
        num_samples=model_config.get("num_samples", 10),
        physics_constraints=model_config.get("physics_constraints", [])
    )
    
    # Find checkpoint files
    checkpoint_dir = os.path.join(args.model_dir, args.checkpoint)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("params_")]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Load latest checkpoint
    checkpoint_files.sort()
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    print(f"Loading model from {checkpoint_file}...")
    
    with open(checkpoint_file, "rb") as f:
        params = pickle.load(f)
    
    return model, params


def load_candidates(args):
    """
    Load candidate materials from file.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of candidate materials
    """
    # Load candidates file
    if not os.path.exists(args.candidates_file):
        raise FileNotFoundError(f"Candidates file not found: {args.candidates_file}")
    
    print(f"Loading candidates from {args.candidates_file}...")
    
    # Placeholder implementation - would load actual candidates
    # For now, just create dummy candidates
    candidates = []
    
    for i in range(args.max_candidates):
        candidate = {
            "id": f"candidate_{i}",
            "composition": "DummyComposition",
            "graph": None,  # Placeholder
            "positions": None,  # Placeholder
            "box": None,  # Placeholder
        }
        candidates.append(candidate)
    
    # Limit number of candidates if needed
    candidates = candidates[:args.max_candidates]
    
    print(f"Loaded {len(candidates)} candidates")
    
    return candidates


def load_dft_settings(args):
    """
    Load DFT settings from file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of DFT settings
    """
    # Default DFT settings for VASP
    default_settings = {
        "INCAR": {
            "ALGO": "Normal",
            "EDIFF": 1e-5,
            "ENCUT": 520,
            "IBRION": 2,
            "ICHARG": 1,
            "ISIF": 3,
            "ISMEAR": 1,
            "ISPIN": 2,
            "LORBIT": 11,
            "LREAL": "Auto",
            "LWAVE": False,
            "NELM": 100,
            "NSW": 99,
            "PREC": "Accurate",
            "SIGMA": 0.2,
        },
        "KPOINTS": {
            "type": "gamma",
            "density": 1000,  # k-points per reciprocal atom
        }
    }
    
    # Load settings from file if specified
    if args.dft_settings and os.path.exists(args.dft_settings):
        with open(args.dft_settings, "r") as f:
            custom_settings = json.load(f)
            
            # Merge custom settings with defaults
            for key, value in custom_settings.items():
                if key in default_settings and isinstance(value, dict):
                    # Merge dictionaries
                    default_settings[key].update(value)
                else:
                    # Replace value
                    default_settings[key] = value
    
    return default_settings


def create_dft_evaluator(args, dft_settings):
    """
    Create DFT evaluator.
    
    Args:
        args: Command line arguments
        dft_settings: DFT settings
        
    Returns:
        DFT evaluator
    """
    # Create evaluator based on method
    if args.dft_method == "vasp":
        evaluator = VASPEvaluator(
            vasp_cmd=args.vasp_cmd,
            potcar_dir=args.potcar_dir,
            work_dir="dft_calcs",
            dft_settings=dft_settings,
            compute_properties=args.properties,
            max_workers=args.max_workers,
            timeout=3600,
            clean_workdir=True
        )
    elif args.dft_method == "pymatgen":
        evaluator = PymatgenEvaluator(
            work_dir="dft_calcs",
            dft_settings=dft_settings,
            compute_properties=args.properties,
            max_workers=args.max_workers,
            timeout=3600,
            clean_workdir=True
        )
    else:
        raise ValueError(f"Unknown DFT method: {args.dft_method}")
    
    # Create batch processor
    processor = DFTBatchProcessor(
        evaluator=evaluator,
        max_batch_size=args.batch_size,
        max_concurrent_jobs=args.max_workers,
        results_dir=os.path.join(args.output_dir, "dft_results")
    )
    
    return processor


def predict_properties(model, params, candidates, properties, rng_key):
    """
    Predict properties using the trained model.
    
    Args:
        model: Trained model
        params: Model parameters
        candidates: List of candidate materials
        properties: List of properties to predict
        rng_key: JAX random key
        
    Returns:
        List of candidates with predictions
    """
    # Create uncertainty propagator
    propagator = UncertaintyPropagator(
        model=model,
        params=params,
        num_samples=100,
        rng_seed=42,
        use_monte_carlo=True
    )
    
    # Predict properties with uncertainty quantification
    candidates_with_predictions = propagator.propagate_property_uncertainties(
        candidates=candidates,
        properties=properties
    )
    
    return candidates_with_predictions


def compare_predictions_with_dft(candidates_with_predictions, dft_results):
    """
    Compare model predictions with DFT results.
    
    Args:
        candidates_with_predictions: List of candidates with predictions
        dft_results: List of DFT results
        
    Returns:
        Dictionary of comparison metrics
    """
    # Initialize metrics
    metrics = {
        "mae": {},
        "rmse": {},
        "calibration": {},
    }
    
    # Get properties from first result
    properties = list(dft_results[0].keys())
    properties = [p for p in properties if p != "success" and p != "error_message"]
    
    # Calculate metrics for each property
    for prop in properties:
        # Extract predictions and DFT values
        predictions = []
        uncertainties = []
        dft_values = []
        
        for i, (candidate, result) in enumerate(
            zip(candidates_with_predictions, dft_results)
        ):
            # Skip failed calculations
            if not result.get("success", True):
                continue
            
            # Get prediction and uncertainty
            if prop in candidate:
                pred = candidate[prop]["mean"]
                uncert = candidate[prop]["uncertainty"]
                
                # Get DFT value
                dft_val = result.get(prop)
                
                if dft_val is not None:
                    predictions.append(pred)
                    uncertainties.append(uncert)
                    dft_values.append(dft_val)
        
        # Convert to arrays
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        dft_values = np.array(dft_values)
        
        # Calculate MAE
        mae = np.mean(np.abs(predictions - dft_values))
        metrics["mae"][prop] = float(mae)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - dft_values) ** 2))
        metrics["rmse"][prop] = float(rmse)
        
        # Calculate calibration metrics
        # (how well do the uncertainties reflect the actual errors)
        normalized_errors = np.abs(predictions - dft_values) / uncertainties
        calibration = np.mean(normalized_errors < 1.0)  # Should be close to 0.68 for 1-sigma
        metrics["calibration"][prop] = float(calibration)
    
    return metrics


def generate_calibration_plot(candidates_with_predictions, dft_results, output_dir):
    """
    Generate calibration plot for uncertainty evaluation.
    
    Args:
        candidates_with_predictions: List of candidates with predictions
        dft_results: List of DFT results
        output_dir: Output directory
    """
    # Placeholder implementation - would generate actual plots
    # In a real implementation, would use matplotlib
    
    print("Generating calibration plots...")
    
    # Get properties from first result
    properties = list(dft_results[0].keys())
    properties = [p for p in properties if p != "success" and p != "error_message"]
    
    # Create plot data for each property
    for prop in properties:
        # Extract predictions, uncertainties, and DFT values
        predictions = []
        uncertainties = []
        dft_values = []
        
        for i, (candidate, result) in enumerate(
            zip(candidates_with_predictions, dft_results)
        ):
            # Skip failed calculations
            if not result.get("success", True):
                continue
            
            # Get prediction and uncertainty
            if prop in candidate:
                pred = candidate[prop]["mean"]
                uncert = candidate[prop]["uncertainty"]
                
                # Get DFT value
                dft_val = result.get(prop)
                
                if dft_val is not None:
                    predictions.append(pred)
                    uncertainties.append(uncert)
                    dft_values.append(dft_val)
        
        # Convert to arrays
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        dft_values = np.array(dft_values)
        
        # Calculate normalized errors
        normalized_errors = np.abs(predictions - dft_values) / uncertainties
        
        # Save data for plot
        plot_data = {
            "property": prop,
            "predictions": predictions.tolist(),
            "uncertainties": uncertainties.tolist(),
            "dft_values": dft_values.tolist(),
            "normalized_errors": normalized_errors.tolist(),
        }
        
        # Save plot data
        plot_file = os.path.join(output_dir, f"calibration_{prop}.json")
        with open(plot_file, "w") as f:
            json.dump(plot_data, f, indent=2)


def main():
    """Main function for evaluating material properties."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    if args.exp_name is None:
        args.exp_name = f"eval_{args.dft_method}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(output_dir, "args.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(42)
    
    # Load model
    model, params = load_model(args)
    
    # Load candidates
    candidates = load_candidates(args)
    
    # Load DFT settings
    dft_settings = load_dft_settings(args)
    
    # Create DFT evaluator
    dft_processor = create_dft_evaluator(args, dft_settings)
    
    # Predict properties using the model
    print(f"Predicting properties of {len(candidates)} candidates...")
    
    rng_key, subkey = jax.random.split(rng_key)
    candidates_with_predictions = predict_properties(
        model, params, candidates, args.properties, subkey
    )
    
    # Save predictions
    predictions_file = os.path.join(output_dir, "predictions.json")
    
    print(f"Saving predictions to {predictions_file}...")
    
    # Placeholder - would implement proper serialization
    with open(predictions_file, "w") as f:
        f.write(json.dumps({
            "num_candidates": len(candidates),
            "properties": args.properties,
        }, indent=2))
    
    # Evaluate properties using DFT
    print(f"Evaluating properties using {args.dft_method}...")
    
    dft_results = dft_processor.process_batch(candidates)
    
    # Save DFT results
    dft_results_file = os.path.join(output_dir, "dft_results.json")
    
    print(f"Saving DFT results to {dft_results_file}...")
    
    # Placeholder - would implement proper serialization
    with open(dft_results_file, "w") as f:
        f.write(json.dumps({
            "num_candidates": len(candidates),
            "properties": args.properties,
            "dft_method": args.dft_method,
        }, indent=2))
    
    # Compare predictions with DFT results
    print("Comparing predictions with DFT results...")
    
    metrics = compare_predictions_with_dft(
        candidates_with_predictions, dft_results
    )
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print metrics
    print("\nPrediction metrics:")
    for prop in metrics["mae"]:
        print(f"  {prop}:")
        print(f"    MAE: {metrics['mae'][prop]:.4f}")
        print(f"    RMSE: {metrics['rmse'][prop]:.4f}")
        print(f"    Calibration: {metrics['calibration'][prop]:.4f}")
    
    # Generate calibration plot if requested
    if args.calibration_plot:
        generate_calibration_plot(
            candidates_with_predictions, dft_results, output_dir
        )
    
    # Perform uncertainty analysis if requested
    if args.uncertainty_analysis:
        print("\nPerforming uncertainty analysis...")
        
        # Placeholder - would implement uncertainty analysis
        # and visualization
        
        # Save results
        uncertainty_file = os.path.join(output_dir, "uncertainty_analysis.json")
        with open(uncertainty_file, "w") as f:
            json.dump({
                "num_candidates": len(candidates),
                "properties": args.properties,
                "num_samples": args.num_samples,
            }, indent=2)
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()