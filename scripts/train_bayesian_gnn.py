#!/usr/bin/env python
"""
Train a Bayesian Graph Neural Network for materials discovery.

This script trains a Bayesian GNN on a dataset of materials with
physics-informed priors and uncertainty quantification.
"""

import argparse
import os
import time
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import pickle

from model.bayesian_gnn import BayesianGNN, train_step
from model.physics_constraints import create_physics_prior
from model.util import get_shift_and_scale


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Bayesian GNN model")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing dataset")
    parser.add_argument("--dataset", type=str, default="gnome_data",
                        help="Dataset name")
    parser.add_argument("--property", type=str, default="formation_energy",
                        help="Property to predict")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Test split ratio")
    
    # Model options
    parser.add_argument("--graph_net_steps", type=int, default=3,
                        help="Number of graph network steps")
    parser.add_argument("--mlp_width", type=int, nargs="+", default=[256, 128, 64],
                        help="Width of MLP layers")
    parser.add_argument("--mlp_nonlinearity", type=str, default="swish",
                        help="Nonlinearity function")
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--prior_scale", type=float, default=1.0,
                        help="Scale of prior distribution")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples for uncertainty estimation")
    parser.add_argument("--physics_constraints", type=str, nargs="+", 
                        default=["energy_conservation", "charge_neutrality"],
                        help="Physics constraints to apply")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL divergence weight (beta)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save model every N epochs")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")
    
    return parser.parse_args()


def load_dataset(args):
    """
    Load dataset from disk.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Placeholder - would implement actual data loading
    data_path = os.path.join(args.data_dir, args.dataset)
    
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}...")
    
    # Dummy data structures for illustration
    dummy_data = {
        "graphs": [],
        "positions": [],
        "boxes": [],
        "targets": []
    }
    
    # Split data into train/val/test
    # In a real implementation, would load actual data and split
    
    return dummy_data, dummy_data, dummy_data


def create_train_state(rng_key, model, dummy_batch, learning_rate, weight_decay):
    """
    Create training state.
    
    Args:
        rng_key: JAX random key
        model: Model instance
        dummy_batch: Dummy batch for initialization
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Training state
    """
    # Create optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize model
    variables = model.init(rng_key, **dummy_batch, training=True)
    
    # Create training state
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer
    )


def create_dummy_batch(batch_size=1):
    """
    Create a dummy batch for initialization.
    
    Args:
        batch_size: Batch size
        
    Returns:
        Dummy batch
    """
    # Placeholder - would create proper dummy batch
    import jraph
    
    # Create dummy graph
    n_node = jnp.array([2] * batch_size)
    n_edge = jnp.array([1] * batch_size)
    
    # Create dummy features
    nodes = jnp.zeros((sum(n_node), 10))
    edges = jnp.zeros((sum(n_edge), 3))
    
    # Create dummy sender/receiver indices
    senders = jnp.zeros((sum(n_edge),), dtype=jnp.int32)
    receivers = jnp.ones((sum(n_edge),), dtype=jnp.int32)
    
    # Create dummy global features
    globals_ = jnp.zeros((batch_size, 5))
    
    # Create GraphsTuple
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge
    )
    
    # Create dummy positions and box
    positions = jnp.zeros((sum(n_node), 3))
    box = jnp.eye(3)[None, ...].repeat(batch_size, axis=0)
    
    # Create dummy batch
    dummy_batch = {
        "graph": graph,
        "positions": positions,
        "box": box,
        "rng_key": jax.random.PRNGKey(0),
        "training": True
    }
    
    return dummy_batch


def create_batches(data, batch_size, shuffle=True):
    """
    Create batches from data.
    
    Args:
        data: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        List of batches
    """
    # Get dataset size
    n_samples = len(data["graphs"])
    
    # Create indices
    indices = np.arange(n_samples)
    
    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(indices)
    
    # Create batches
    batches = []
    for i in range(0, n_samples, batch_size):
        # Get batch indices
        batch_indices = indices[i:i+batch_size]
        
        # Create batch
        batch = {}
        for key, value in data.items():
            if key == "graphs":
                # Placeholder - would implement proper graph batching
                batch["graph"] = None
            else:
                batch[key] = [value[j] for j in batch_indices]
        
        batches.append(batch)
    
    return batches


def train_epoch(state, train_batches, rng_key, beta):
    """
    Train for one epoch.
    
    Args:
        state: Training state
        train_batches: Training batches
        rng_key: JAX random key
        beta: KL divergence weight
        
    Returns:
        Updated state and metrics
    """
    # Initialize metrics
    metrics = {
        "loss": [],
        "nll": [],
        "kl": []
    }
    
    # Loop through batches
    for batch in train_batches:
        # Split RNG key
        rng_key, subkey = jax.random.split(rng_key)
        
        # Update state and get metrics
        state, batch_metrics = train_step(state, batch, subkey, beta)
        
        # Update metrics
        for key, value in batch_metrics.items():
            metrics[key].append(value)
    
    # Average metrics
    for key in metrics:
        metrics[key] = float(np.mean(metrics[key]))
    
    return state, metrics, rng_key


def evaluate(model, state, eval_batches, rng_key):
    """
    Evaluate model.
    
    Args:
        model: Model instance
        state: Training state
        eval_batches: Evaluation batches
        rng_key: JAX random key
        
    Returns:
        Evaluation metrics
    """
    # Initialize metrics
    metrics = {
        "mae": [],
        "rmse": [],
        "uncertainty": []
    }
    
    # Loop through batches
    for batch in eval_batches:
        # Split RNG key
        rng_key, subkey = jax.random.split(rng_key)
        
        # Get predictions
        mean, std = model.apply(
            {"params": state.params},
            **batch,
            rng_key=subkey,
            training=False
        )
        
        # Calculate metrics
        targets = batch["targets"]
        mae = jnp.mean(jnp.abs(mean - targets))
        rmse = jnp.sqrt(jnp.mean((mean - targets) ** 2))
        
        # Update metrics
        metrics["mae"].append(float(mae))
        metrics["rmse"].append(float(rmse))
        metrics["uncertainty"].append(float(jnp.mean(std)))
    
    # Average metrics
    for key in metrics:
        metrics[key] = float(np.mean(metrics[key]))
    
    return metrics


def save_model(state, model_config, output_dir, epoch):
    """
    Save model checkpoint.
    
    Args:
        state: Training state
        model_config: Model configuration
        output_dir: Output directory
        epoch: Current epoch
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model parameters
    params_file = os.path.join(output_dir, f"params_{epoch:04d}.pkl")
    with open(params_file, "wb") as f:
        pickle.dump(state.params, f)
    
    # Save model configuration
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Model saved to {output_dir}")


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    if args.exp_name is None:
        args.exp_name = f"bayesian_gnn_{time.strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(output_dir, "args.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed for reproducibility
    rng_key = jax.random.PRNGKey(42)
    
    # Load dataset
    train_data, val_data, test_data = load_dataset(args)
    
    # Create model
    model = BayesianGNN(
        graph_net_steps=args.graph_net_steps,
        mlp_width=tuple(args.mlp_width),
        mlp_nonlinearity=args.mlp_nonlinearity,
        embedding_dim=args.embedding_dim,
        prior_scale=args.prior_scale,
        num_samples=args.num_samples,
        physics_constraints=args.physics_constraints
    )
    
    # Create dummy batch for initialization
    dummy_batch = create_dummy_batch()
    
    # Create training state
    rng_key, init_key = jax.random.split(rng_key)
    state = create_train_state(
        init_key,
        model,
        dummy_batch,
        args.learning_rate,
        args.weight_decay
    )
    
    # Create batches
    train_batches = create_batches(train_data, args.batch_size, shuffle=True)
    val_batches = create_batches(val_data, args.batch_size, shuffle=False)
    test_batches = create_batches(test_data, args.batch_size, shuffle=False)
    
    # Save model configuration
    model_config = {
        "graph_net_steps": args.graph_net_steps,
        "mlp_width": args.mlp_width,
        "mlp_nonlinearity": args.mlp_nonlinearity,
        "embedding_dim": args.embedding_dim,
        "prior_scale": args.prior_scale,
        "num_samples": args.num_samples,
        "physics_constraints": args.physics_constraints,
    }
    
    # Training loop
    best_val_rmse = float("inf")
    metrics_history = []
    
    print(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        # Train for one epoch
        rng_key, subkey = jax.random.split(rng_key)
        state, train_metrics, rng_key = train_epoch(
            state, train_batches, subkey, args.beta
        )
        
        # Evaluate if needed
        if (epoch + 1) % args.eval_every == 0:
            # Evaluate on validation set
            rng_key, subkey = jax.random.split(rng_key)
            val_metrics = evaluate(model, state, val_batches, subkey)
            
            # Combine metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics
            }
            
            # Save metrics
            metrics_history.append(epoch_metrics)
            metrics_file = os.path.join(output_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics_history, f, indent=2)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{args.num_epochs}:")
            print(f"  Train: loss={train_metrics['loss']:.4f}, nll={train_metrics['nll']:.4f}, kl={train_metrics['kl']:.4f}")
            print(f"  Val: mae={val_metrics['mae']:.4f}, rmse={val_metrics['rmse']:.4f}, uncertainty={val_metrics['uncertainty']:.4f}")
            
            # Save model if it's the best so far
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                save_model(state, model_config, os.path.join(output_dir, "best"), epoch + 1)
        
        # Save model checkpoint if needed
        if (epoch + 1) % args.save_every == 0:
            save_model(state, model_config, os.path.join(output_dir, "checkpoints"), epoch + 1)
    
    # Evaluate on test set
    rng_key, subkey = jax.random.split(rng_key)
    test_metrics = evaluate(model, state, test_batches, subkey)
    
    # Print test metrics
    print("\nTest metrics:")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Uncertainty: {test_metrics['uncertainty']:.4f}")
    
    # Save test metrics
    test_metrics_file = os.path.join(output_dir, "test_metrics.json")
    with open(test_metrics_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save final model
    save_model(state, model_config, os.path.join(output_dir, "final"), args.num_epochs)
    
    print(f"\nTraining completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()