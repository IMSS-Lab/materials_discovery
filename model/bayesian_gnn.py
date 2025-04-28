"""
Bayesian Graph Neural Networks for Materials Discovery
with Physics-Informed Uncertainty Quantification.

This module implements Bayesian Graph Neural Networks that provide
uncertainty estimates in material property predictions, with
physics-informed priors that incorporate domain knowledge.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from typing import Callable, Dict, List, Optional, Tuple, Union

from .gnn import GraphNetwork
from .util import Array, UnaryFn, PRNGKey
from .physics_constraints import apply_physics_constraints


class BayesianMLP(nn.Module):
    """Bayesian Multi-Layer Perceptron with variational inference."""
    
    features: Tuple[int, ...]
    nonlinearity: str
    prior_scale: float = 1.0
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x, rng_key, training=True):
        features = self.features
        from .util import get_nonlinearity_by_name
        phi = get_nonlinearity_by_name(self.nonlinearity)
        
        # Split the random key for different layers
        keys = jax.random.split(rng_key, len(features))
        
        for i, (h, key) in enumerate(zip(features[:-1], keys)):
            # Define mu and rho parameters for the weight distribution
            weight_mu = self.param(f'weight_mu_{i}', nn.initializers.normal(0.1), (x.shape[-1], h))
            weight_rho = self.param(f'weight_rho_{i}', nn.initializers.normal(0.1), (x.shape[-1], h))
            
            if self.use_bias:
                bias_mu = self.param(f'bias_mu_{i}', nn.initializers.zeros, (h,))
                bias_rho = self.param(f'bias_rho_{i}', nn.initializers.zeros, (h,))
            
            # Apply local reparameterization trick for efficiency
            if training:
                weight_sigma = jnp.log(1 + jnp.exp(weight_rho))
                weight_eps = jax.random.normal(key, weight_mu.shape)
                weight_sample = weight_mu + weight_eps * weight_sigma
                
                if self.use_bias:
                    bias_sigma = jnp.log(1 + jnp.exp(bias_rho))
                    bias_eps = jax.random.normal(key, bias_mu.shape)
                    bias_sample = bias_mu + bias_eps * bias_sigma
                else:
                    bias_sample = None
            else:
                # During inference, just use mean values
                weight_sample = weight_mu
                bias_sample = bias_mu if self.use_bias else None
            
            # Linear transformation with sampled weights
            if bias_sample is not None:
                x = jnp.dot(x, weight_sample) + bias_sample
            else:
                x = jnp.dot(x, weight_sample)
            
            # Apply nonlinearity
            x = phi(x)
        
        # Final layer (no nonlinearity)
        key = keys[-1]
        weight_mu = self.param(f'weight_mu_{len(features)-1}', 
                               nn.initializers.normal(0.1), 
                               (x.shape[-1], features[-1]))
        weight_rho = self.param(f'weight_rho_{len(features)-1}', 
                                nn.initializers.normal(0.1), 
                                (x.shape[-1], features[-1]))
        
        if self.use_bias:
            bias_mu = self.param(f'bias_mu_{len(features)-1}', 
                                nn.initializers.zeros, 
                                (features[-1],))
            bias_rho = self.param(f'bias_rho_{len(features)-1}', 
                                 nn.initializers.zeros, 
                                 (features[-1],))
        
        if training:
            weight_sigma = jnp.log(1 + jnp.exp(weight_rho))
            weight_eps = jax.random.normal(key, weight_mu.shape)
            weight_sample = weight_mu + weight_eps * weight_sigma
            
            if self.use_bias:
                bias_sigma = jnp.log(1 + jnp.exp(bias_rho))
                bias_eps = jax.random.normal(key, bias_mu.shape)
                bias_sample = bias_mu + bias_eps * bias_sigma
            else:
                bias_sample = None
        else:
            weight_sample = weight_mu
            bias_sample = bias_mu if self.use_bias else None
        
        # Final linear transformation
        if bias_sample is not None:
            x = jnp.dot(x, weight_sample) + bias_sample
        else:
            x = jnp.dot(x, weight_sample)
        
        return x
    
    def kl_divergence(self):
        """Compute KL divergence between variational posterior and prior."""
        kl_sum = 0.0
        
        for i in range(len(self.features)):
            # Get variational parameters
            weight_mu = self.variables['params'][f'weight_mu_{i}']
            weight_rho = self.variables['params'][f'weight_rho_{i}']
            weight_sigma = jnp.log(1 + jnp.exp(weight_rho))
            
            # KL divergence between variational posterior N(mu, sigma^2) and prior N(0, prior_scale^2)
            kl_weights = 0.5 * jnp.sum(
                (weight_sigma / self.prior_scale) ** 2 + 
                (weight_mu / self.prior_scale) ** 2 - 
                1 - 
                2 * jnp.log(weight_sigma / self.prior_scale)
            )
            
            kl_sum += kl_weights
            
            if self.use_bias:
                bias_mu = self.variables['params'][f'bias_mu_{i}']
                bias_rho = self.variables['params'][f'bias_rho_{i}']
                bias_sigma = jnp.log(1 + jnp.exp(bias_rho))
                
                kl_bias = 0.5 * jnp.sum(
                    (bias_sigma / self.prior_scale) ** 2 + 
                    (bias_mu / self.prior_scale) ** 2 - 
                    1 - 
                    2 * jnp.log(bias_sigma / self.prior_scale)
                )
                
                kl_sum += kl_bias
        
        return kl_sum


class BayesianGNN(nn.Module):
    """Bayesian Graph Neural Network with physics-informed priors."""
    
    graph_net_steps: int
    mlp_width: Tuple[int]
    mlp_nonlinearity: Union[str, Dict[str, str]]
    embedding_dim: int
    
    prior_scale: float = 1.0
    num_samples: int = 10
    physics_constraints: List[str] = None
    
    @nn.compact
    def __call__(self, graph, positions, box, box_perturbation=None, 
                rng_key=None, training=True, sample_count=None):
        """
        Forward pass through the Bayesian GNN.
        
        Args:
            graph: Input graph structure
            positions: Atomic positions
            box: Periodic box
            box_perturbation: Perturbation to box
            rng_key: JAX random key for stochastic forward pass
            training: Whether in training mode
            sample_count: Number of forward passes to sample (default: self.num_samples)
        
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
            
        sample_count = sample_count or self.num_samples
        
        # Create node embeddings
        node_embed_fn = nn.Dense(self.embedding_dim, name='Node Embedding')
        edge_embed_fn = nn.Dense(self.embedding_dim, name='Edge Embedding')
        global_embed_fn = nn.Dense(self.embedding_dim, name='Global Embedding')
        
        embed = jraph.GraphMapFeatures(
            embed_node_fn=node_embed_fn,
            embed_edge_fn=edge_embed_fn,
            embed_global_fn=global_embed_fn,
        )
        
        # Setup Bayesian MLPs for updates
        def node_update_fn(i):
            return BayesianMLP(
                self.mlp_width, 
                self.mlp_nonlinearity,
                prior_scale=self.prior_scale,
                name=f'Node Update {i}'
            )
        
        def edge_update_fn(i):
            return BayesianMLP(
                self.mlp_width, 
                self.mlp_nonlinearity,
                prior_scale=self.prior_scale,
                name=f'Edge Update {i}'
            )
        
        def global_update_fn(i):
            return BayesianMLP(
                self.mlp_width, 
                self.mlp_nonlinearity,
                prior_scale=self.prior_scale,
                name=f'Global Update {i}'
            )
        
        # Setup readout MLP
        readout_width = self.mlp_width[:-1] + (1,)
        global_readout_fn = BayesianMLP(
            readout_width,
            self.mlp_nonlinearity,
            prior_scale=self.prior_scale,
            name='Readout'
        )
        
        # Embed the graph
        embedded_graph = embed(graph)
        
        if training:
            # Single forward pass during training
            output = self._forward_pass(
                embedded_graph, 
                node_update_fn,
                edge_update_fn,
                global_update_fn,
                global_readout_fn,
                rng_key
            )
            
            # Apply physics constraints if specified
            if self.physics_constraints:
                output = apply_physics_constraints(
                    output, graph, positions, self.physics_constraints
                )
                
            return output, None
        else:
            # Multiple forward passes during inference for uncertainty estimation
            keys = jax.random.split(rng_key, sample_count)
            
            # Run multiple forward passes
            samples = []
            for i in range(sample_count):
                output = self._forward_pass(
                    embedded_graph,
                    node_update_fn,
                    edge_update_fn,
                    global_update_fn,
                    global_readout_fn,
                    keys[i],
                    training=False
                )
                
                # Apply physics constraints if specified
                if self.physics_constraints:
                    output = apply_physics_constraints(
                        output, graph, positions, self.physics_constraints
                    )
                
                samples.append(output)
            
            # Compute mean and standard deviation
            samples = jnp.stack(samples)
            mean_prediction = jnp.mean(samples, axis=0)
            std_prediction = jnp.std(samples, axis=0)
            
            return mean_prediction, std_prediction
    
    def _forward_pass(self, graph, node_update_fn, edge_update_fn, 
                     global_update_fn, global_readout_fn, rng_key, training=True):
        """Perform a single forward pass through the network."""
        # Split RNG key for different update functions
        keys = jax.random.split(rng_key, self.graph_net_steps + 1)
        
        # Process through graph network layers
        output = graph
        for i in range(self.graph_net_steps - 1):
            # Create the graph network for this step
            gn = GraphNetwork(
                update_node_fn=lambda nodes, senders, receivers, globals: 
                    node_update_fn(i)(nodes, keys[i], training),
                update_edge_fn=lambda edges, senders, receivers, globals: 
                    edge_update_fn(i)(edges, keys[i], training),
                update_global_fn=lambda nodes, edges, globals: 
                    global_update_fn(i)(globals, keys[i], training),
                aggregate_edges_for_nodes_fn=jraph.segment_sum,
                aggregate_nodes_for_globals_fn=jraph.segment_mean,
                aggregate_edges_for_globals_fn=jraph.segment_mean,
            )
            
            # Apply the graph network
            output = gn(output)
        
        # Final readout to get predictions
        readout = GraphNetwork(
            update_node_fn=None,
            update_edge_fn=None,
            update_global_fn=lambda nodes, edges, globals: 
                global_readout_fn(globals, keys[-1], training),
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
            aggregate_nodes_for_globals_fn=jraph.segment_mean,
            aggregate_edges_for_globals_fn=jraph.segment_mean,
        )
        
        output = readout(output).globals
        return output
    
    def kl_divergence(self):
        """Compute total KL divergence for variational inference."""
        kl_sum = 0.0
        
        # Sum KL divergence from all Bayesian layers
        for i in range(self.graph_net_steps - 1):
            kl_sum += self.variables['params'][f'Node Update {i}'].kl_divergence()
            kl_sum += self.variables['params'][f'Edge Update {i}'].kl_divergence()
            kl_sum += self.variables['params'][f'Global Update {i}'].kl_divergence()
        
        kl_sum += self.variables['params']['Readout'].kl_divergence()
        
        return kl_sum


def train_step(state, batch, rng_key, beta=1.0):
    """Single training step for ELBO optimization."""
    def loss_fn(params):
        # Forward pass
        model = state.model
        predictions, _ = model.apply(
            {'params': params},
            batch['graph'],
            batch['positions'],
            batch['box'],
            rng_key=rng_key,
            training=True
        )
        
        # Negative log likelihood (assuming Gaussian likelihood)
        targets = batch['target']
        mse_loss = jnp.mean((predictions - targets) ** 2)
        nll = 0.5 * mse_loss  # Proportional to negative log likelihood
        
        # KL divergence
        kl = model.apply({'params': params}, method=model.kl_divergence)
        
        # ELBO = -NLL - beta * KL
        elbo = nll + beta * kl
        
        return elbo, (nll, kl)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (elbo, (nll, kl)), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, {'elbo': elbo, 'nll': nll, 'kl': kl}


class PredictiveDistribution:
    """Utility class for working with predictive distributions."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def confidence_interval(self, alpha=0.95):
        """Compute confidence interval at given alpha level."""
        z = 1.96  # Approximate value for 95% CI
        if alpha != 0.95:
            # TODO: Implement proper quantile for other alpha values
            pass
        
        lower = self.mean - z * self.std
        upper = self.mean + z * self.std
        
        return lower, upper
    
    def entropy(self):
        """Compute entropy of the predictive distribution."""
        # Assuming Gaussian distribution
        return 0.5 * jnp.log(2 * jnp.pi * jnp.e * self.std ** 2)