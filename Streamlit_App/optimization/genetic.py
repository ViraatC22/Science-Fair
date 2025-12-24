import numpy as np
from scipy import stats
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network.predictor import ModelManager
from simulation.physics import compute_metrics, LITERATURE_DATA

class GeneticOptimizer:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.param_bounds = {
            'pillar_count': (4, 20),
            'pillar_size_mm': (10.0, 50.0),
            'channel_width_mm': (5.0, 30.0),
            'channel_node_size_mm': (0.0, 15.0),
            'scaffold_stiffness_kPa': (1.0, 50.0),
            'elasticity': (0.0, 1.0),
            'scaffold_density_g_cm3': (1.0, 1.5),
            'initial_mass_g': (0.1, 2.0),
            'dmem_glucose': (0.0, 50.0),
            'dmem_glutamine': (0.0, 50.0),
            'dmem_pyruvate': (0.0, 5.0),
            'ion_na': (100.0, 200.0),
            'ion_k': (1.0, 10.0),
            'ion_cl': (100.0, 200.0),
            'ion_ca': (0.1, 5.0),
            'media_depth_mm': (0.5, 5.0),
            'light_lumens': (0, 1000)
        }
        self.param_names = list(self.param_bounds.keys())
        
    def _encode_params(self, params):
        # Convert dict to array
        return np.array([params.get(k, 0) for k in self.param_names])
    
    def _decode_params(self, individual):
        # Convert array to dict
        params = {}
        for i, k in enumerate(self.param_names):
            val = individual[i]
            if k == 'pillar_count':
                params[k] = int(val)
            else:
                params[k] = float(val)
        return params

    def calculate_p_value_score(self, metrics):
        # We want to be significantly BETTER than literature
        # Example: Growth Rate > Literature Mean
        
        lit_mean = LITERATURE_DATA["Kay_2022_GrowthRate_High"]["mean"]
        lit_std = LITERATURE_DATA["Kay_2022_GrowthRate_High"]["std"]
        
        # Simulated one-sample t-test against literature mean
        # Assuming we have a 'sample' size of 1 for the prediction, we check z-score
        z_score = (metrics['avg_growth_rate'] - lit_mean) / (lit_std + 1e-6)
        
        # p-value from z-score (one-tailed)
        p_val = stats.norm.sf(z_score)
        
        # If p_val < 0.05, it means we are significantly higher
        return p_val, z_score

    def fitness_function(self, individual, model_type):
        params = self._decode_params(individual)
        params['model_type'] = model_type # Pass model type for context
        
        # Use NN to predict metrics (faster than running full simulation)
        # Note: In a real scenario, we'd input the params into the NN. 
        # Here, for demonstration, we might use the compute_metrics logic 
        # IF the NN isn't fully trained, but the requirement is to use the NN.
        # Let's assume the NN is trained on the compute_metrics logic.
        
        # For this implementation, I will use compute_metrics directly as the "Ground Truth" 
        # generator for the GA to optimize, AND we will use the NN to *guide* it 
        # (e.g., using NN gradients or just using NN for fast eval if we had a slow sim).
        # Since our sim is fast, we can use compute_metrics directly for accuracy, 
        # OR use the NN to demonstrate the requirement.
        
        # Requirement: "Neural network should be integrated... and used to make the scaffold"
        # I will use the NN to predict the fitness.
        
        if not self.model_manager.is_trained:
            # Fallback if not trained (should verify training first)
            metrics = compute_metrics(params, model_type)
        else:
            # Predict using NN
            # Input vector must match training data structure
            input_vec = individual.reshape(1, -1)
            # The NN outputs [growth_rate, tortuosity, permeability, ...]
            # We need to map this back to the metrics dict
            pred = self.model_manager.predict(input_vec)[0]
            metrics = {
                'avg_growth_rate': pred[0],
                'mean_tortuosity': pred[1],
                'permeability_kappa_iso': pred[2],
                # ... others
            }
            
        p_val, z_score = self.calculate_p_value_score(metrics)
        
        # Fitness: Maximize Growth, Minimize P-value (i.e., Maximize Z-score)
        # We want Z-score to be high (positive).
        fitness = float(z_score)
        if not np.isfinite(fitness):
            return -float('inf'), 1.0
        
        # Penalties for structural validity
        if params['pillar_size_mm'] < 10 or params['channel_width_mm'] < 5:
            fitness -= 100
            
        return fitness, p_val

    def run_optimization(self, model_type, pop_size=50, generations=20):
        # Initialize population
        population = []
        for _ in range(pop_size):
            ind = []
            for k in self.param_names:
                low, high = self.param_bounds[k]
                ind.append(np.random.uniform(low, high))
            population.append(np.array(ind))
            
        best_individual = None
        best_fitness = -float('inf')
        history = []
        
        for gen in range(generations):
            scores = []
            for ind in population:
                fit, p_val = self.fitness_function(ind, model_type)
                scores.append((fit, ind, p_val))
                
                if fit > best_fitness:
                    best_fitness = fit
                    best_individual = ind
            
            # Sort by fitness
            scores.sort(key=lambda x: x[0], reverse=True)
            history.append(scores[0][0])
            
            # Selection (Top 50%)
            survivors = [s[1] for s in scores[:pop_size//2]]
            
            # Crossover & Mutation
            new_pop = survivors[:]
            while len(new_pop) < pop_size:
                parent1 = survivors[np.random.randint(0, len(survivors))]
                parent2 = survivors[np.random.randint(0, len(survivors))]
                
                # Crossover
                child = (parent1 + parent2) / 2.0
                
                # Mutation
                if np.random.rand() < 0.2:
                    idx = np.random.randint(0, len(child))
                    noise = np.random.normal(0, 1.0)
                    child[idx] += noise
                
                for i, k in enumerate(self.param_names):
                    low, high = self.param_bounds[k]
                    child[i] = np.clip(child[i], low, high)
                    
                new_pop.append(child)
            
            population = new_pop
            
        if best_individual is None:
            best_individual = scores[0][1]
            best_fitness = scores[0][0]
        return self._decode_params(best_individual), best_fitness, history

def train_initial_model(model_manager, n_samples=1000):
    # Generate synthetic data based on simulation logic
    X = []
    y = []
    
    # Define bounds same as optimizer
    bounds = {
        'pillar_count': (4, 20),
        'pillar_size_mm': (10.0, 50.0),
        'channel_width_mm': (5.0, 30.0),
        'channel_node_size_mm': (0.0, 15.0),
        'scaffold_stiffness_kPa': (1.0, 50.0),
        'elasticity': (0.0, 1.0),
        'scaffold_density_g_cm3': (1.0, 1.5),
        'initial_mass_g': (0.1, 2.0),
        'dmem_glucose': (0.0, 50.0),
        'dmem_glutamine': (0.0, 50.0),
        'dmem_pyruvate': (0.0, 5.0),
        'ion_na': (100.0, 200.0),
        'ion_k': (1.0, 10.0),
        'ion_cl': (100.0, 200.0),
        'ion_ca': (0.1, 5.0),
        'media_depth_mm': (0.5, 5.0),
        'light_lumens': (0, 1000)
    }
    keys = list(bounds.keys())
    
    while len(X) < n_samples:
        params = {}
        row = []
        for k in keys:
            val = np.random.uniform(bounds[k][0], bounds[k][1])
            params[k] = val
            row.append(val)
        
        # Assume generic model type for training general physics
        params['model_type'] = "3D Porous (Channel Diffusion)" 
        metrics = compute_metrics(params, "3D Porous (Channel Diffusion)")
        
        # Target: [growth_rate, tortuosity, permeability]
        # We need to standardize what the NN predicts.
        target = [
            metrics['avg_growth_rate'],
            metrics['mean_tortuosity'],
            metrics.get('permeability_kappa_iso', 0),
            metrics.get('mst_ratio', 0),
            metrics.get('fractal_dimension', 0)
        ]
        if np.all(np.isfinite(target)):
            X.append(row)
            y.append(target)
        
    loss = model_manager.train(X, y, epochs=50)
    return loss
