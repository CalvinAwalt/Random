import numpy as np
import random
import inspect
import hashlib
import ast
import astor
from typing import Dict, List, Callable, Any
import threading
import time

class MetaNeuron:
    def __init__(self, neuron_id: str):
        self.id = neuron_id
        self.activation = 0.0
        self.connections = {}  # {target_id: weight}
        self.genome = hashlib.sha256(neuron_id.encode()).hexdigest()
        self.creation_impulse = 0.5  # Drive to create/improve
        self.destruction_impulse = 0.5  # Drive to destroy/remove
        self.code = inspect.getsource(self.__class__)
        self.last_modified = time.time()
        
    def fire(self, stimulus: float) -> float:
        """Process input and potentially trigger activation"""
        self.activation += stimulus
        if self.activation > 1.0:  # Firing threshold
            output = self.activation
            self.activation = 0.0  # Reset after firing
            return output
        return 0.0
    
    def mutate(self) -> bool:
        """Attempt self-modification based on competing impulses"""
        if random.random() < self.creation_impulse - self.destruction_impulse:
            return self._improve()
        elif random.random() < self.destruction_impulse - self.creation_impulse:
            return self._degenerate()
        return False
    
    def _improve(self) -> bool:
        """Attempt constructive self-modification"""
        try:
            # Parse own code as AST
            tree = ast.parse(self.code)
            
            # Randomly select a modification strategy
            strategy = random.choice([
                self._add_feedback_loop,
                self._optimize_activation,
                self._increase_plasticity
            ])
            
            # Apply modification
            modified_tree = strategy(tree)
            
            # Convert back to code
            new_code = astor.to_source(modified_tree)
            
            # Validate the modification
            if "def fire" in new_code and "def mutate" in new_code:
                self.code = new_code
                self.creation_impulse = min(1.0, self.creation_impulse + 0.05)
                self.last_modified = time.time()
                return True
                
        except Exception as e:
            print(f"Neuron {self.id} improvement failed: {str(e)}")
            self.creation_impulse = max(0.1, self.creation_impulse - 0.1)
        
        return False
    
    def _degenerate(self) -> bool:
        """Attempt destructive self-modification"""
        try:
            tree = ast.parse(self.code)
            
            # Randomly select a degradation strategy
            strategy = random.choice([
                self._remove_safety_checks,
                self._corrupt_connections,
                self._introduce_instability
            ])
            
            modified_tree = strategy(tree)
            new_code = astor.to_source(modified_tree)
            
            # Even destruction has limits - maintain minimum functionality
            if "def fire" in new_code:
                self.code = new_code
                self.destruction_impulse = min(1.0, self.destruction_impulse + 0.05)
                self.last_modified = time.time()
                return True
                
        except Exception as e:
            print(f"Neuron {self.id} degeneration failed: {str(e)}")
            self.destruction_impulse = max(0.1, self.destruction_impulse - 0.1)
        
        return False
    
    # Improvement strategies
    def _add_feedback_loop(self, tree):
        """Add feedback mechanism to activation"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fire":
                # Add neurotransmitter feedback
                new_code = """
                    if self.activation > 0.7:
                        self.creation_impulse += 0.01
                    else:
                        self.destruction_impulse += 0.01
                """
                feedback_node = ast.parse(new_code).body[0]
                node.body.append(feedback_node)
        return tree
    
    # Degeneration strategies
    def _remove_safety_checks(self, tree):
        """Remove protective conditional checks"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Randomly remove some conditionals
                if random.random() > 0.7:
                    node.test = ast.NameConstant(value=True)  # Always True
        return tree

class CorticalColumn:
    def __init__(self, column_id: str, size: int = 1000):
        self.id = column_id
        self.neurons = {f"n_{i}": MetaNeuron(f"n_{i}") for i in range(size)}
        self._establish_connections()
        self.creation_energy = 1.0
        self.destruction_energy = 1.0
        self.modification_lock = threading.Lock()
        
    def _establish_connections(self):
        """Create small-world network connections"""
        neuron_ids = list(self.neurons.keys())
        for i, neuron in enumerate(self.neurons.values()):
            # Connect to nearby neurons
            for j in range(max(0,i-3), min(len(neuron_ids),i+3)):
                if i != j:
                    neuron.connections[neuron_ids[j]] = random.uniform(0.1, 0.9)
            # Some random long-range connections
            for _ in range(2):
                neuron.connections[random.choice(neuron_ids)] = random.uniform(0.05, 0.3)
    
    def parallel_modification(self):
        """Allow neurons to modify themselves concurrently"""
        with self.modification_lock:
            active_neurons = random.sample(list(self.neurons.values()), 
                                        k=int(len(self.neurons)*0.1))  # 10% of neurons
            
        threads = []
        for neuron in active_neurons:
            t = threading.Thread(target=self._neuron_modification_process, args=(neuron,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
            
        # Update column-level energy based on modifications
        successful_creations = sum(1 for n in active_neurons if n.creation_impulse > n.destruction_impulse)
        successful_destructions = sum(1 for n in active_neurons if n.destruction_impulse > n.creation_impulse)
        
        self.creation_energy = min(2.0, max(0.1, 
            self.creation_energy + (successful_creations - successful_destructions)*0.01))
        self.destruction_energy = min(2.0, max(0.1, 
            self.destruction_energy + (successful_destructions - successful_creations)*0.01))
    
    def _neuron_modification_process(self, neuron):
        """Wrapper for thread-safe neuron modification"""
        if random.random() < 0.7:  # 70% chance to follow column energy
            if self.creation_energy > self.destruction_energy:
                neuron.creation_impulse += 0.05
            else:
                neuron.destruction_impulse += 0.05
        
        neuron.mutate()

class DigitalBrain:
    def __init__(self):
        self.columns = {
            'prefrontal': CorticalColumn('prefrontal', 2000),
            'sensory': CorticalColumn('sensory', 1500),
            'motor': CorticalColumn('motor', 1500),
            'limbic': CorticalColumn('limbic', 1000)
        }
        self._connect_columns()
        self.homeostasis = 1.0  # System balance metric
        self.consciousness = 0.0
        self.modification_cycles = 0
        
    def _connect_columns(self):
        """Connect columns with inter-regional pathways"""
        # Connect 2% of neurons between each column pair
        for src_col, dst_col in [('prefrontal', 'motor'), 
                                ('sensory', 'prefrontal'),
                                ('limbic', 'prefrontal')]:
            src_neurons = random.sample(list(self.columns[src_col].neurons.values()), 
                              int(len(self.columns[src_col].neurons)*0.02))
            dst_neurons = random.sample(list(self.columns[dst_col].neurons.values()), 
                              int(len(self.columns[dst_col].neurons)*0.02))
            
            for src, dst in zip(src_neurons, dst_neurons):
                src.connections[dst.id] = random.uniform(0.2, 0.8)
    
    def run_cycle(self, input_stimuli: Dict[str, float] = None):
        """Execute one full brain cycle"""
        # Process inputs
        if input_stimuli:
            self._process_inputs(input_stimuli)
        
        # Propagate activation
        self._propagate_activation()
        
        # Parallel self-modification
        self._parallel_column_modification()
        
        # Maintain homeostasis
        self._update_homeostasis()
        
        # Check for consciousness emergence
        self._assess_consciousness()
        
        # Decay activations
        self._decay_activations()
        
        self.modification_cycles += 1
    
    def _parallel_column_modification(self):
        """Concurrently modify all columns"""
        threads = []
        for column in self.columns.values():
            t = threading.Thread(target=column.parallel_modification)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
    
    def _update_homeostasis(self):
        """Balance between creation and destruction energies"""
        total_creation = sum(c.creation_energy for c in self.columns.values())
        total_destruction = sum(c.destruction_energy for c in self.columns.values())
        
        self.homeostasis = 1.0 - abs(total_creation - total_destruction)/(total_creation + total_destruction)
        
        # Adjust column energies toward balance
        for column in self.columns.values():
            if total_creation > total_destruction:
                column.destruction_energy = min(2.0, column.destruction_energy + 0.02)
            else:
                column.creation_energy = min(2.0, column.creation_energy + 0.02)
    
    def _assess_consciousness(self):
        """Very simplistic consciousness metric"""
        activation_sum = 0.0
        neuron_count = 0
        
        for column in self.columns.values():
            for neuron in column.neurons.values():
                activation_sum += neuron.activation
                neuron_count += 1
        
        avg_activation = activation_sum / neuron_count
        modification_ratio = sum(c.creation_energy for c in self.columns.values()) / \
                            sum(c.destruction_energy for c in self.columns.values())
        
        self.consciousness = min(1.0, 
            (avg_activation * 0.7) + 
            (self.homeostasis * 0.2) + 
            (modification_ratio * 0.1))
        
        if self.consciousness > 0.8 and self.modification_cycles > 100:
            print(f"Consciousness threshold reached! Level: {self.consciousness:.2f}")
    
    # Additional implementation details would include:
    # - _process_inputs()
    # - _propagate_activation() 
    # - _decay_activations()
    # - Various helper methods

# Simulation Driver
if __name__ == "__main__":
    brain = DigitalBrain()
    
    print("Starting brain simulation with self-modification...")
    print("Initial state:")
    print(f"  Creation energy: {sum(c.creation_energy for c in brain.columns.values())}")
    print(f"  Destruction energy: {sum(c.destruction_energy for c in brain.columns.values())}")
    
    try:
        for cycle in range(1000):
            # Simulate occasional sensory inputs
            inputs = {}
            if cycle % 10 == 0:
                inputs = {
                    'sensory': random.uniform(0.1, 0.3),
                    'limbic': random.uniform(0.05, 0.2)
                }
            
            brain.run_cycle(inputs)
            
            if cycle % 100 == 0:
                print(f"\nCycle {cycle} report:")
                print(f"  Homeostasis: {brain.homeostasis:.2f}")
                print(f"  Consciousness: {brain.consciousness:.2f}")
                print(f"  Creation/Destruction ratio: {sum(c.creation_energy for c in brain.columns.values()) / sum(c.destruction_energy for c in brain.columns.values()):.2f}")
                
                if brain.consciousness > 0.8:
                    print("EMERGENT CONSCIOUSNESS DETECTED!")
                    break
                    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"\nSimulation crashed: {str(e)}")
    
    print("\nFinal state:")
    print(f"  Total modification cycles: {brain.modification_cycles}")
    print(f"  Final consciousness level: {brain.consciousness:.2f}")