"""
Uncertainty propagation through material discovery pipelines.

This module provides functions for propagating uncertainties through
the materials discovery pipeline, from property predictions to final
stability assessments.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import functools
from dataclasses import dataclass

from ..model.bayesian_gnn import BayesianGNN
from ..model.uncertainty import uncertainty_decomposition


@dataclass
class PropertyDistribution:
    """Distribution of a material property with uncertainty."""
    
    mean: float
    std: float
    samples: Optional[np.ndarray] = None
    property_name: str = ""
    units: str = ""
    
    def __post_init__(self):
        if self.samples is None and self.std > 0:
            # Generate samples if not provided
            self.samples = np.random.normal(self.mean, self.std, 1000)
    
    def probability_in_range(self, min_val: float, max_val: float) -> float:
        """
        Calculate probability that the property falls within a range.
        
        Args:
            min_val: Minimum value of range
            max_val: Maximum value of range
            
        Returns:
            Probability (0-1) of property falling in range
        """
        if self.samples is not None:
            # Empirical probability from samples
            in_range = np.logical_and(
                self.samples >= min_val,
                self.samples <= max_val
            )
            return np.mean(in_range)
        else:
            # Analytical calculation using normal CDF
            from scipy.stats import norm
            return norm.cdf(max_val, loc=self.mean, scale=self.std) - \
                   norm.cdf(min_val, loc=self.mean, scale=self.std)
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the property.
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.samples is not None:
            # Empirical confidence interval from samples
            alpha = (1 - confidence) / 2
            lower = np.quantile(self.samples, alpha)
            upper = np.quantile(self.samples, 1 - alpha)
            return lower, upper
        else:
            # Analytical calculation using normal quantiles
            from scipy.stats import norm
            z = norm.ppf(1 - (1 - confidence) / 2)
            return self.mean - z * self.std, self.mean + z * self.std


class UncertaintyPropagator:
    """
    Propagate uncertainties through materials discovery pipelines.
    
    This class handles the propagation of uncertainties from property
    predictions to stability assessments and final material rankings.
    """
    
    def __init__(
        self,
        model: BayesianGNN,
        params: Dict,
        num_samples: int = 1000,
        rng_seed: int = 42,
        use_monte_carlo: bool = True
    ):
        """
        Initialize uncertainty propagator.
        
        Args:
            model: Trained BayesianGNN model
            params: Model parameters
            num_samples: Number of samples for Monte Carlo sampling
            rng_seed: Random seed for reproducibility
            use_monte_carlo: Whether to use Monte Carlo sampling
        """
        self.model = model
        self.params = params
        self.num_samples = num_samples
        self.rng_seed = rng_seed
        self.use_monte_carlo = use_monte_carlo
        
        # Initialize random number generator
        self.rng = jax.random.PRNGKey(rng_seed)
    
    def propagate_property_uncertainties(
        self,
        candidates: List[Dict],
        properties: List[str],
        property_functions: Optional[Dict[str, Callable]] = None
    ) -> List[Dict]:
        """
        Propagate uncertainties through property predictions.
        
        Args:
            candidates: List of candidate materials
            properties: List of properties to predict
            property_functions: Optional dict mapping property names to functions
                                that compute derived properties
            
        Returns:
            List of candidates with propagated uncertainties
        """
        # Predict primary properties with uncertainty
        candidates_with_props = self._predict_properties(
            candidates, properties
        )
        
        # Propagate to derived properties if needed
        if property_functions:
            candidates_with_props = self._propagate_to_derived_properties(
                candidates_with_props, property_functions
            )
        
        return candidates_with_props
    
    def propagate_stability_uncertainties(
        self,
        candidates: List[Dict],
        reference_energies: Dict[str, float],
        reference_uncertainties: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Propagate uncertainties to stability assessments.
        
        Args:
            candidates: List of candidate materials with property predictions
            reference_energies: Dictionary mapping compositions to reference energies
            reference_uncertainties: Optional dict of reference energy uncertainties
            
        Returns:
            List of candidates with stability uncertainties
        """
        # Ensure energy predictions are available
        for candidate in candidates:
            if "energy" not in candidate:
                raise ValueError("Energy predictions required for stability assessment")
        
        # Set default reference uncertainties if not provided
        if reference_uncertainties is None:
            reference_uncertainties = {comp: 0.0 for comp in reference_energies}
        
        # Calculate stability with uncertainty for each candidate
        for candidate in candidates:
            # Get composition
            composition = candidate.get("composition", "")
            
            # Get energy distribution
            energy_mean = candidate["energy"]["mean"]
            energy_std = candidate["energy"]["uncertainty"]
            
            # Calculate stability against reference phases
            stability_distribution = self._calculate_stability_distribution(
                composition, energy_mean, energy_std,
                reference_energies, reference_uncertainties
            )
            
            # Store stability distribution in candidate
            candidate["stability"] = {
                "mean": stability_distribution.mean,
                "uncertainty": stability_distribution.std,
                "probability_stable": stability_distribution.probability_in_range(
                    -float("inf"), 0.0
                ),
                "confidence_interval_95": stability_distribution.confidence_interval(0.95),
            }
        
        return candidates
    
    def rank_candidates_with_uncertainty(
        self,
        candidates: List[Dict],
        ranking_property: str = "stability",
        secondary_properties: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        Rank candidates with uncertainty consideration.
        
        Args:
            candidates: List of candidate materials with property predictions
            ranking_property: Primary property for ranking
            secondary_properties: Optional list of secondary properties
            weights: Optional dict mapping properties to weights
            
        Returns:
            Tuple of (ranked_candidates, confidence_scores)
        """
        # Set default weights if not provided
        if weights is None:
            weights = {ranking_property: 1.0}
            if secondary_properties:
                for prop in secondary_properties:
                    weights[prop] = 0.5
        
        # Calculate ranking scores and uncertainties
        candidate_scores = []
        
        for candidate in candidates:
            # Calculate weighted score
            score = 0.0
            score_variance = 0.0
            
            for prop, weight in weights.items():
                if prop in candidate:
                    # Add weighted contribution to score
                    score += weight * candidate[prop]["mean"]
                    
                    # Add weighted contribution to variance
                    # (assuming properties are independent)
                    score_variance += (weight * candidate[prop]["uncertainty"]) ** 2
            
            # Store score and uncertainty
            candidate_scores.append({
                "candidate": candidate,
                "score": score,
                "uncertainty": np.sqrt(score_variance),
            })
        
        # Rank candidates by score
        candidate_scores.sort(key=lambda x: x["score"])
        
        # Calculate ranking confidence
        confidence_scores = self._calculate_ranking_confidence(candidate_scores)
        
        # Extract ranked candidates
        ranked_candidates = [item["candidate"] for item in candidate_scores]
        
        return ranked_candidates, confidence_scores
    
    def _predict_properties(
        self,
        candidates: List[Dict],
        properties: List[str]
    ) -> List[Dict]:
        """
        Predict properties with uncertainty quantification.
        
        Args:
            candidates: List of candidate materials
            properties: List of properties to predict
            
        Returns:
            List of candidates with property predictions and uncertainties
        """
        # Split RNG key
        self.rng, subkey = jax.random.split(self.rng)
        
        # Prepare inputs for prediction
        inputs = self._prepare_inputs(candidates)
        
        # Get predictions with uncertainty decomposition
        predictions = uncertainty_decomposition(
            model=self.model,
            params=self.params,
            inputs=inputs,
            rng_key=subkey,
            num_models=5,
            num_samples_per_model=10
        )
        
        # Create property distributions for each candidate
        for i, candidate in enumerate(candidates):
            for j, prop in enumerate(properties):
                # Extract predictions for this property
                mean = float(predictions["mean"][i, j])
                uncertainty = float(predictions["total_uncertainty"][i, j])
                epistemic = float(predictions["epistemic_uncertainty"][i, j])
                aleatoric = float(predictions["aleatoric_uncertainty"][i, j])
                
                # Create property distribution
                property_dist = PropertyDistribution(
                    mean=mean,
                    std=uncertainty,
                    property_name=prop
                )
                
                # Store in candidate
                candidate[prop] = {
                    "mean": mean,
                    "uncertainty": uncertainty,
                    "epistemic_uncertainty": epistemic,
                    "aleatoric_uncertainty": aleatoric,
                    "distribution": property_dist,
                }
        
        return candidates
    
    def _propagate_to_derived_properties(
        self,
        candidates: List[Dict],
        property_functions: Dict[str, Callable]
    ) -> List[Dict]:
        """
        Propagate uncertainties to derived properties.
        
        Args:
            candidates: List of candidates with primary property predictions
            property_functions: Dict mapping derived property names to functions
            
        Returns:
            List of candidates with derived property predictions
        """
        for candidate in candidates:
            for derived_prop, prop_function in property_functions.items():
                if self.use_monte_carlo:
                    # Monte Carlo propagation
                    derived_dist = self._monte_carlo_propagation(
                        candidate, prop_function, derived_prop
                    )
                else:
                    # First-order approximation
                    derived_dist = self._first_order_propagation(
                        candidate, prop_function, derived_prop
                    )
                
                # Store derived property in candidate
                candidate[derived_prop] = {
                    "mean": derived_dist.mean,
                    "uncertainty": derived_dist.std,
                    "distribution": derived_dist,
                }
        
        return candidates
    
    def _monte_carlo_propagation(
        self,
        candidate: Dict,
        prop_function: Callable,
        derived_prop: str
    ) -> PropertyDistribution:
        """
        Propagate uncertainties using Monte Carlo sampling.
        
        Args:
            candidate: Candidate material with property predictions
            prop_function: Function to compute derived property
            derived_prop: Name of derived property
            
        Returns:
            Property distribution for derived property
        """
        # Split RNG key
        self.rng, subkey = jax.random.split(self.rng)
        
        # Get input properties required by function
        input_properties = prop_function.__code__.co_varnames
        
        # Generate samples for each input property
        property_samples = {}
        for prop in input_properties:
            if prop in candidate and "distribution" in candidate[prop]:
                dist = candidate[prop]["distribution"]
                if dist.samples is not None:
                    # Use existing samples
                    property_samples[prop] = dist.samples
                else:
                    # Generate new samples
                    property_samples[prop] = np.random.normal(
                        dist.mean, dist.std, self.num_samples
                    )
        
        # Apply function to samples
        derived_samples = []
        for i in range(self.num_samples):
            # Get sample for each input property
            sample_inputs = {
                prop: property_samples[prop][i]
                for prop in input_properties
                if prop in property_samples
            }
            
            # Apply function
            try:
                derived_value = prop_function(**sample_inputs)
                derived_samples.append(derived_value)
            except Exception:
                # Skip samples that cause errors
                continue
        
        # Compute statistics
        derived_samples = np.array(derived_samples)
        derived_mean = np.mean(derived_samples)
        derived_std = np.std(derived_samples)
        
        # Create distribution
        return PropertyDistribution(
            mean=derived_mean,
            std=derived_std,
            samples=derived_samples,
            property_name=derived_prop
        )
    
    def _first_order_propagation(
        self,
        candidate: Dict,
        prop_function: Callable,
        derived_prop: str
    ) -> PropertyDistribution:
        """
        Propagate uncertainties using first-order approximation.
        
        Args:
            candidate: Candidate material with property predictions
            prop_function: Function to compute derived property
            derived_prop: Name of derived property
            
        Returns:
            Property distribution for derived property
        """
        # Get input properties required by function
        input_properties = prop_function.__code__.co_varnames
        
        # Get mean values for each input property
        mean_inputs = {
            prop: candidate[prop]["mean"]
            for prop in input_properties
            if prop in candidate
        }
        
        # Evaluate function at mean inputs
        try:
            derived_mean = prop_function(**mean_inputs)
        except Exception:
            # Return zero distribution if function fails
            return PropertyDistribution(
                mean=0.0,
                std=0.0,
                property_name=derived_prop
            )
        
        # Calculate derivatives using finite differences
        derivatives = {}
        eps = 1e-6
        
        for prop in input_properties:
            if prop in candidate:
                # Create perturbed inputs
                perturbed_inputs = mean_inputs.copy()
                perturbed_inputs[prop] += eps
                
                # Evaluate function with perturbed input
                try:
                    perturbed_value = prop_function(**perturbed_inputs)
                    
                    # Calculate derivative
                    derivatives[prop] = (perturbed_value - derived_mean) / eps
                except Exception:
                    # Skip properties that cause errors
                    derivatives[prop] = 0.0
        
        # Calculate variance using error propagation
        derived_variance = 0.0
        
        for prop1 in input_properties:
            if prop1 in candidate and prop1 in derivatives:
                for prop2 in input_properties:
                    if prop2 in candidate and prop2 in derivatives:
                        # Get uncertainties
                        if prop1 == prop2:
                            # Variance
                            cov = candidate[prop1]["uncertainty"] ** 2
                        else:
                            # Covariance (assume independence)
                            cov = 0.0
                        
                        # Add contribution to derived variance
                        derived_variance += derivatives[prop1] * derivatives[prop2] * cov
        
        # Calculate derived uncertainty
        derived_std = np.sqrt(max(0.0, derived_variance))
        
        # Create distribution
        return PropertyDistribution(
            mean=derived_mean,
            std=derived_std,
            property_name=derived_prop
        )
    
    def _calculate_stability_distribution(
        self,
        composition: str,
        energy_mean: float,
        energy_std: float,
        reference_energies: Dict[str, float],
        reference_uncertainties: Dict[str, float]
    ) -> PropertyDistribution:
        """
        Calculate stability distribution against competing phases.
        
        Args:
            composition: Material composition
            energy_mean: Mean formation energy
            energy_std: Formation energy uncertainty
            reference_energies: Dict mapping compositions to reference energies
            reference_uncertainties: Dict mapping compositions to reference uncertainties
            
        Returns:
            Distribution of stability (decomposition energy)
        """
        # Placeholder implementation - would use proper convex hull analysis
        # For now, just use a simple approximation
        
        # Find closest reference composition
        closest_energy = float("inf")
        closest_uncertainty = 0.0
        
        for ref_comp, ref_energy in reference_energies.items():
            # Simple string similarity as placeholder
            # In real implementation, would use composition similarity
            if ref_comp in composition or composition in ref_comp:
                if ref_energy < closest_energy:
                    closest_energy = ref_energy
                    closest_uncertainty = reference_uncertainties.get(ref_comp, 0.0)
        
        # If no similar composition found, use minimum energy
        if closest_energy == float("inf"):
            closest_energy = min(reference_energies.values())
            closest_comp = min(reference_energies, key=reference_energies.get)
            closest_uncertainty = reference_uncertainties.get(closest_comp, 0.0)
        
        # Calculate stability
        stability_mean = energy_mean - closest_energy
        
        # Combine uncertainties (assuming independence)
        stability_std = np.sqrt(energy_std ** 2 + closest_uncertainty ** 2)
        
        # Create distribution
        return PropertyDistribution(
            mean=stability_mean,
            std=stability_std,
            property_name="stability",
            units="eV/atom"
        )
    
    def _calculate_ranking_confidence(
        self,
        candidate_scores: List[Dict]
    ) -> List[float]:
        """
        Calculate confidence in ranking of candidates.
        
        Args:
            candidate_scores: List of candidates with scores and uncertainties
            
        Returns:
            List of confidence scores (0-1) for each ranking position
        """
        n_candidates = len(candidate_scores)
        confidence_scores = []
        
        for i, item in enumerate(candidate_scores):
            # Calculate probability this candidate is better than all others
            confidence = 1.0
            
            for j, other_item in enumerate(candidate_scores):
                if i != j:
                    # Calculate probability item i is better than item j
                    if self.use_monte_carlo and "samples" in item:
                        # Use empirical probability
                        better_count = np.sum(item["samples"] < other_item["samples"])
                        prob_better = better_count / len(item["samples"])
                    else:
                        # Use analytical approximation (normal distribution)
                        mean_diff = other_item["score"] - item["score"]
                        std_diff = np.sqrt(item["uncertainty"] ** 2 + other_item["uncertainty"] ** 2)
                        
                        # Probability i is better than j
                        if std_diff > 0:
                            from scipy.stats import norm
                            prob_better = norm.cdf(mean_diff / std_diff)
                        else:
                            prob_better = 1.0 if mean_diff > 0 else 0.0
                    
                    # Update confidence
                    confidence *= prob_better
            
            confidence_scores.append(confidence)
        
        return confidence_scores
    
    def _prepare_inputs(self, candidates: List[Dict]) -> Dict:
        """
        Prepare inputs for model prediction.
        
        Args:
            candidates: List of candidate materials
            
        Returns:
            Dictionary of model inputs
        """
        # Extract graph, positions, and box from candidates
        # Placeholder implementation - would implement proper batching
        graphs = [c["graph"] for c in candidates]
        positions = [c["positions"] for c in candidates]
        boxes = [c["box"] for c in candidates]
        
        # Placeholder - would implement proper batching
        inputs = {
            "graph": graphs[0],  # Placeholder
            "positions": positions[0],
            "box": boxes[0],
        }
        
        return inputs


# Example property functions for derived properties

def bandgap_temperature_dependence(bandgap: float, temperature: float) -> float:
    """
    Calculate temperature-dependent bandgap.
    
    Args:
        bandgap: Bandgap at 0K (eV)
        temperature: Temperature (K)
        
    Returns:
        Temperature-dependent bandgap (eV)
    """
    # Varshni equation parameters (example values)
    alpha = 5e-4  # eV/K
    beta = 300.0  # K
    
    # Varshni equation
    return bandgap - (alpha * temperature ** 2) / (temperature + beta)


def seebeck_coefficient(bandgap: float, effective_mass: float, temperature: float) -> float:
    """
    Calculate Seebeck coefficient.
    
    Args:
        bandgap: Bandgap (eV)
        effective_mass: Effective mass (m_e)
        temperature: Temperature (K)
        
    Returns:
        Seebeck coefficient (μV/K)
    """
    # Simplified model (placeholder)
    k_B = 8.617e-5  # eV/K (Boltzmann constant)
    
    # Simple approximation
    return 1000.0 * (k_B / 2) * (bandgap / (k_B * temperature)) * np.sqrt(effective_mass)


def thermal_expansion(bulk_modulus: float, gruneisen_parameter: float, heat_capacity: float) -> float:
    """
    Calculate thermal expansion coefficient.
    
    Args:
        bulk_modulus: Bulk modulus (GPa)
        gruneisen_parameter: Gruneisen parameter (dimensionless)
        heat_capacity: Volumetric heat capacity (J/mol/K)
        
    Returns:
        Thermal expansion coefficient (K^-1)
    """
    # Convert bulk modulus to Pa
    bulk_modulus_pa = bulk_modulus * 1e9
    
    # Formula for thermal expansion
    return gruneisen_parameter * heat_capacity / (bulk_modulus_pa * 3)


def phase_stability_temperature(
    formation_energy: float,
    entropy: float,
    temperature: float
) -> float:
    """
    Calculate temperature-dependent phase stability.
    
    Args:
        formation_energy: Formation energy at 0K (eV/atom)
        entropy: Vibrational entropy (eV/atom/K)
        temperature: Temperature (K)
        
    Returns:
        Temperature-dependent formation energy (eV/atom)
    """
    # Free energy = formation_energy - T * entropy
    return formation_energy - temperature * entropy


def ionic_conductivity(
    migration_barrier: float,
    attempt_frequency: float,
    concentration: float,
    temperature: float
) -> float:
    """
    Calculate ionic conductivity.
    
    Args:
        migration_barrier: Migration energy barrier (eV)
        attempt_frequency: Attempt frequency (THz)
        concentration: Mobile ion concentration (mol/cm^3)
        temperature: Temperature (K)
        
    Returns:
        Ionic conductivity (S/cm)
    """
    # Constants
    k_B = 8.617e-5  # eV/K (Boltzmann constant)
    e = 1.602e-19  # C (elementary charge)
    N_A = 6.022e23  # mol^-1 (Avogadro's number)
    
    # Diffusion coefficient (cm^2/s)
    diffusion = attempt_frequency * 1e12 * 1e-16 * np.exp(-migration_barrier / (k_B * temperature))
    
    # Nernst-Einstein relation
    conductivity = (diffusion * concentration * N_A * e ** 2) / (k_B * temperature)
    
    return conductivity