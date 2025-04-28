"""
Physics-Informed Constraints for Bayesian Graph Neural Networks.

This module implements physical constraints that can be incorporated
into Bayesian GNN priors or applied as post-processing constraints
to ensure physically meaningful predictions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from .util import Array


def create_physics_prior(constraints: List[str], constraint_weights: Optional[Dict[str, float]] = None):
    """
    Create a physics-informed prior for Bayesian Graph Neural Networks.
    
    Args:
        constraints: List of constraint names to apply
        constraint_weights: Optional dictionary mapping constraint names to weights
        
    Returns:
        A function that computes the log prior probability given predictions
    """
    if constraint_weights is None:
        constraint_weights = {c: 1.0 for c in constraints}
    
    def log_prior(params, predictions, graph, positions, box):
        """Log prior probability incorporating physical constraints."""
        log_p = 0.0
        
        for constraint in constraints:
            if constraint == "energy_conservation":
                log_p += constraint_weights[constraint] * energy_conservation_prior(
                    predictions, graph, positions
                )
            elif constraint == "charge_neutrality":
                log_p += constraint_weights[constraint] * charge_neutrality_prior(
                    predictions, graph
                )
            elif constraint == "symmetry_preservation":
                log_p += constraint_weights[constraint] * symmetry_preservation_prior(
                    predictions, graph, positions, box
                )
            elif constraint == "non_negative_bandgap":
                log_p += constraint_weights[constraint] * non_negative_bandgap_prior(
                    predictions
                )
            elif constraint == "lattice_constants":
                log_p += constraint_weights[constraint] * lattice_constants_prior(
                    predictions, box
                )
        
        return log_p
    
    return log_prior


def apply_physics_constraints(predictions: Array, graph: Any, 
                             positions: Array, constraints: List[str]) -> Array:
    """
    Apply physical constraints to predictions as post-processing.
    
    Args:
        predictions: Model predictions
        graph: Input graph structure
        positions: Atomic positions
        constraints: List of constraint names to apply
        
    Returns:
        Constrained predictions
    """
    constrained_predictions = predictions
    
    for constraint in constraints:
        if constraint == "energy_conservation":
            constrained_predictions = apply_energy_conservation(
                constrained_predictions, graph, positions
            )
        elif constraint == "charge_neutrality":
            constrained_predictions = apply_charge_neutrality(
                constrained_predictions, graph
            )
        elif constraint == "symmetry_preservation":
            # Symmetry constraints typically applied during generation,
            # not as post-processing
            pass
        elif constraint == "non_negative_bandgap":
            constrained_predictions = apply_non_negative_bandgap(
                constrained_predictions
            )
    
    return constrained_predictions


# Prior functions

def energy_conservation_prior(predictions: Array, graph: Any, positions: Array) -> float:
    """
    Prior that encourages energy conservation.
    
    For total energy predictions, this ensures that the sum of per-atom energies
    equals the total energy prediction.
    
    Args:
        predictions: Energy predictions
        graph: Input graph structure
        positions: Atomic positions
        
    Returns:
        Log prior probability
    """
    # If predictions include per-atom and total energies, ensure consistency
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        per_atom_energies = predictions[:, :-1]
        total_energy = predictions[:, -1]
        
        # Sum per-atom energies
        n_node = graph.n_node
        node_graph_idx = jnp.repeat(
            jnp.arange(n_node.shape[0]), n_node, axis=0, 
            total_repeat_length=positions.shape[0]
        )
        expected_total = jax.ops.segment_sum(
            per_atom_energies, node_graph_idx, n_node.shape[0]
        )
        
        # Calculate deviation from conservation
        deviation = jnp.mean((total_energy - expected_total) ** 2)
        
        # Convert to log probability (higher means more consistent)
        return -100.0 * deviation
    
    # If only total energy is predicted, nothing to check
    return 0.0


def charge_neutrality_prior(predictions: Array, graph: Any) -> float:
    """
    Prior that encourages charge neutrality in the crystal.
    
    Args:
        predictions: Predictions that might include charge-related values
        graph: Input graph with node features that include atomic information
        
    Returns:
        Log prior probability
    """
    # Extract oxidation states from graph nodes
    node_features = graph.nodes
    
    # This is a simplified example assuming oxidation states are available
    # In a real implementation, would need to extract actual oxidation states
    # or calculate them from node features
    oxidation_states = node_features[:, 0]  # Placeholder
    
    # Calculate total charge
    total_charge = jnp.sum(oxidation_states)
    
    # Log prior probability (higher for charge-neutral structures)
    return -10.0 * total_charge ** 2


def symmetry_preservation_prior(predictions: Array, graph: Any, 
                               positions: Array, box: Array) -> float:
    """
    Prior that encourages preservation of crystal symmetry.
    
    Args:
        predictions: Model predictions
        graph: Input graph structure
        positions: Atomic positions
        box: Periodic box
        
    Returns:
        Log prior probability
    """
    # This is a complex operation that would require symmetry analysis
    # For now, return a constant value
    return 0.0


def non_negative_bandgap_prior(predictions: Array) -> float:
    """
    Prior that encourages non-negative bandgaps.
    
    Args:
        predictions: Model predictions, including bandgap
        
    Returns:
        Log prior probability
    """
    # Assume bandgap prediction is in a specific index
    bandgap_idx = 0  # Placeholder
    
    if len(predictions.shape) > 1 and predictions.shape[1] > bandgap_idx:
        bandgap = predictions[:, bandgap_idx]
        
        # Create a soft constraint for non-negative bandgap
        negative_penalty = jnp.sum(jnp.maximum(0, -bandgap))
        
        return -100.0 * negative_penalty
    
    return 0.0


def lattice_constants_prior(predictions: Array, box: Array) -> float:
    """
    Prior for reasonable lattice constants based on atomic radii.
    
    Args:
        predictions: Model predictions
        box: Periodic box
        
    Returns:
        Log prior probability
    """
    # Extract lattice constants from box
    lattice_constants = jnp.diagonal(box, axis1=1, axis2=2)
    
    # Reasonable minimum and maximum values for lattice constants
    min_lattice = 2.0  # Angstroms
    max_lattice = 50.0  # Angstroms
    
    # Calculate penalties for unreasonable lattice constants
    below_min_penalty = jnp.sum(jnp.maximum(0, min_lattice - lattice_constants))
    above_max_penalty = jnp.sum(jnp.maximum(0, lattice_constants - max_lattice))
    
    total_penalty = below_min_penalty + above_max_penalty
    
    return -10.0 * total_penalty


# Post-processing constraint functions

def apply_energy_conservation(predictions: Array, graph: Any, positions: Array) -> Array:
    """
    Apply energy conservation constraint by adjusting predictions.
    
    Args:
        predictions: Energy predictions
        graph: Input graph structure
        positions: Atomic positions
        
    Returns:
        Adjusted predictions
    """
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Similar logic to the prior, but actually modifying the predictions
        per_atom_energies = predictions[:, :-1]
        
        # Sum per-atom energies to get total
        n_node = graph.n_node
        node_graph_idx = jnp.repeat(
            jnp.arange(n_node.shape[0]), n_node, axis=0, 
            total_repeat_length=positions.shape[0]
        )
        calculated_total = jax.ops.segment_sum(
            per_atom_energies, node_graph_idx, n_node.shape[0]
        )
        
        # Replace the predicted total with the calculated total
        adjusted_predictions = predictions.at[:, -1].set(calculated_total)
        
        return adjusted_predictions
    
    return predictions


def apply_charge_neutrality(predictions: Array, graph: Any) -> Array:
    """
    Apply charge neutrality constraint by adjusting predictions.
    
    Args:
        predictions: Predictions that might include charge-related values
        graph: Input graph structure
        
    Returns:
        Adjusted predictions
    """
    # Would need to implement logic to adjust charges
    # For now, just return the original predictions
    return predictions


def apply_non_negative_bandgap(predictions: Array) -> Array:
    """
    Apply non-negative bandgap constraint.
    
    Args:
        predictions: Model predictions including bandgap
        
    Returns:
        Adjusted predictions
    """
    # Assume bandgap prediction is in a specific index
    bandgap_idx = 0  # Placeholder
    
    if len(predictions.shape) > 1 and predictions.shape[1] > bandgap_idx:
        # Ensure bandgap is non-negative
        adjusted_predictions = predictions.at[:, bandgap_idx].set(
            jnp.maximum(0, predictions[:, bandgap_idx])
        )
        
        return adjusted_predictions
    
    return predictions


# Utility functions for symmetry operations

def get_symmetry_operations(spacegroup: int) -> List[np.ndarray]:
    """
    Get symmetry operations for a given spacegroup.
    
    Args:
        spacegroup: International space group number (1-230)
        
    Returns:
        List of 4x4 transformation matrices
    """
    # This would use spglib or another library to get symmetry operations
    # For now, return placeholder data
    return [np.eye(4)]  # Identity operation as placeholder


def apply_symmetry_operation(positions: Array, operation: np.ndarray) -> Array:
    """
    Apply symmetry operation to atomic positions.
    
    Args:
        positions: Atomic positions
        operation: 4x4 transformation matrix
        
    Returns:
        Transformed positions
    """
    # Convert to homogeneous coordinates
    homogeneous = jnp.column_stack((positions, jnp.ones(positions.shape[0])))
    
    # Apply transformation
    transformed = jnp.dot(homogeneous, operation.T)
    
    # Convert back to 3D coordinates
    return transformed[:, :3]