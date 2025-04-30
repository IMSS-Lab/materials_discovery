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

"""Physics-informed constraints for material property prediction."""

import jax
import jax.numpy as jnp
from typing import Dict, Callable, List, Tuple, Any, Optional
import e3nn_jax as e3nn
import jraph

def energy_conservation_constraint(predictions: jnp.ndarray, 
                                  atomic_energies: jnp.ndarray,
                                  node_graph_idx: jnp.ndarray,
                                  n_graphs: int) -> jnp.ndarray:
    """Enforces energy conservation by ensuring total energy = sum of atomic contributions.
    
    Args:
        predictions: Predicted total energies per graph [n_graphs].
        atomic_energies: Per-atom energies [n_nodes].
        node_graph_idx: Mapping from nodes to graphs [n_nodes].
        n_graphs: Number of graphs.
        
    Returns:
        Penalty term for violation of energy conservation.
    """
    # Sum atomic energies per graph
    total_energy = jraph.segment_sum(atomic_energies, node_graph_idx, n_graphs)
    
    # Calculate mean squared difference
    return jnp.mean((predictions - total_energy) ** 2)

def charge_neutrality_constraint(atomic_charges: jnp.ndarray,
                                node_graph_idx: jnp.ndarray,
                                n_graphs: int) -> jnp.ndarray:
    """Enforces charge neutrality by ensuring sum of atomic charges = 0.
    
    Args:
        atomic_charges: Per-atom charges [n_nodes].
        node_graph_idx: Mapping from nodes to graphs [n_nodes].
        n_graphs: Number of graphs.
        
    Returns:
        Penalty term for violation of charge neutrality.
    """
    # Sum charges per graph
    total_charge = jraph.segment_sum(atomic_charges, node_graph_idx, n_graphs)
    
    # Calculate mean squared charge
    return jnp.mean(total_charge ** 2)

def crystal_symmetry_constraint(features: jnp.ndarray, 
                               symmetry_ops: List[jnp.ndarray]) -> jnp.ndarray:
    """Enforces crystal symmetry invariance under symmetry operations.
    
    Args:
        features: Node or edge features [n_nodes, feature_dim].
        symmetry_ops: List of symmetry operations [n_ops, 3, 3].
        
    Returns:
        Penalty term for violation of symmetry invariance.
    """
    # Check invariance under each symmetry operation
    penalty = 0.0
    
    for op in symmetry_ops:
        # Apply symmetry operation to features
        # This is a simplified implementation - in practice would depend
        # on the specific feature representation
        
        # Placeholder transformation
        transformed = features  # Would apply the symmetry operation
        
        # Calculate mean squared difference
        penalty += jnp.mean((transformed - features) ** 2)
    
    return penalty / len(symmetry_ops)

def get_oxidation_states(elements: jnp.ndarray, oxidation_state_dict: Dict[int, int]) -> jnp.ndarray:
    """Get oxidation states for elements.
    
    Args:
        elements: Atomic numbers [n_nodes].
        oxidation_state_dict: Dictionary mapping atomic numbers to oxidation states.
        
    Returns:
        Oxidation states [n_nodes].
    """
    # Convert elements to oxidation states
    oxidation_states = jnp.zeros_like(elements, dtype=jnp.float32)
    
    for element, oxidation in oxidation_state_dict.items():
        oxidation_states = jnp.where(elements == element, oxidation, oxidation_states)
    
    return oxidation_states

class PhysicsInformedPrior:
    """Class for constructing physics-informed priors for Bayesian GNNs."""
    
    def __init__(self, 
                energy_weight: float = 1.0,
                charge_weight: float = 1.0,
                symmetry_weight: float = 0.1,
                oxidation_states: Optional[Dict[int, int]] = None):
        """Initialize physics-informed prior.
        
        Args:
            energy_weight: Weight for energy conservation constraint.
            charge_weight: Weight for charge neutrality constraint.
            symmetry_weight: Weight for crystal symmetry constraint.
            oxidation_states: Dictionary mapping atomic numbers to oxidation states.
                If None, uses common oxidation states.
        """
        self.energy_weight = energy_weight
        self.charge_weight = charge_weight
        self.symmetry_weight = symmetry_weight
        
        # Default oxidation states for common elements
        self.oxidation_states = oxidation_states or {
            1: 1,    # H
            3: 1,    # Li
            11: 1,   # Na
            19: 1,   # K
            37: 1,   # Rb
            55: 1,   # Cs
            4: 2,    # Be
            12: 2,   # Mg
            20: 2,   # Ca
            38: 2,   # Sr
            56: 2,   # Ba
            13: 3,   # Al
            26: 2,   # Fe(II)
            27: 2,   # Co(II)
            28: 2,   # Ni(II)
            29: 2,   # Cu(II)
            30: 2,   # Zn
            8: -2,   # O
            9: -1,   # F
            16: -2,  # S
            17: -1,  # Cl
            35: -1,  # Br
            53: -1,  # I
        }
    
    def __call__(self, graph: jraph.GraphsTuple, model_outputs: Dict[str, Any]) -> jnp.ndarray:
        """Calculate physics-informed prior probability (negative log prior).
        
        Args:
            graph: Graph representation.
            model_outputs: Dictionary containing model predictions and intermediate values.
            
        Returns:
            Total physics constraint penalty (negative log prior).
        """
        penalty = 0.0
        n_node = graph.n_node
        n_graph = n_node.shape[0]
        
        # Create node-to-graph mapping for aggregation
        sum_n_node = jnp.sum(n_node)
        node_gr_idx = jnp.repeat(
            jnp.arange(n_graph), n_node, axis=0, total_repeat_length=sum_n_node
        )
        
        # Energy conservation constraint
        if 'energy' in model_outputs and 'atomic_energies' in model_outputs:
            energy_penalty = energy_conservation_constraint(
                model_outputs['energy'], 
                model_outputs['atomic_energies'],
                node_gr_idx,
                n_graph
            )
            penalty += self.energy_weight * energy_penalty
        
        # Charge neutrality constraint
        if 'atomic_charges' in model_outputs:
            charge_penalty = charge_neutrality_constraint(
                model_outputs['atomic_charges'],
                node_gr_idx,
                n_graph
            )
            penalty += self.charge_weight * charge_penalty
        elif graph.nodes is not None:
            # If atomic charges aren't provided but nodes are,
            # we can estimate charges from oxidation states
            if hasattr(graph.nodes, 'shape') and len(graph.nodes.shape) >= 2:
                # Assuming one-hot encoding of elements in graph.nodes
                elements = jnp.argmax(graph.nodes, axis=1)
                oxidation_states = get_oxidation_states(elements, self.oxidation_states)
                
                charge_penalty = charge_neutrality_constraint(
                    oxidation_states,
                    node_gr_idx,
                    n_graph
                )
                penalty += self.charge_weight * charge_penalty
        
        # Crystal symmetry constraint
        if 'node_features' in model_outputs and 'symmetry_ops' in model_outputs:
            symmetry_penalty = crystal_symmetry_constraint(
                model_outputs['node_features'],
                model_outputs['symmetry_ops']
            )
            penalty += self.symmetry_weight * symmetry_penalty
        
        return penalty

def physics_informed_loss(prediction: jnp.ndarray, 
                         target: jnp.ndarray,
                         graph: jraph.GraphsTuple,
                         model_outputs: Dict[str, Any],
                         physics_prior: PhysicsInformedPrior,
                         kl_weight: float = 1.0) -> jnp.ndarray:
    """Physics-informed loss function combining prediction error and physics constraints.
    
    Args:
        prediction: Model prediction.
        target: Target value.
        graph: Graph representation.
        model_outputs: Dictionary of model intermediate outputs.
        physics_prior: Physics-informed prior.
        kl_weight: Weight for KL divergence term.
        
    Returns:
        Total loss combining prediction error and physics constraints.
    """
    # Prediction error (e.g., MSE)
    pred_loss = jnp.mean((prediction - target) ** 2)
    
    # Physics-informed prior (negative log prior)
    prior_loss = physics_prior(graph, model_outputs)
    
    # KL divergence (if available in model_outputs)
    kl_div = model_outputs.get('kl_divergence', 0.0)
    
    # Combine losses
    total_loss = pred_loss + prior_loss + kl_weight * kl_div
    
    return total_loss

def compute_symmetry_operations(lattice: jnp.ndarray, spacegroup: int) -> List[jnp.ndarray]:
    """Compute symmetry operations for a crystal.
    
    Args:
        lattice: Lattice vectors [3, 3].
        spacegroup: Space group number.
        
    Returns:
        List of symmetry operations [n_ops, 3, 3].
    """
    # This is a placeholder - in practice, would use spglib or similar
    # to compute the symmetry operations for the given space group
    
    # Return identity operation as placeholder
    return [jnp.eye(3)]