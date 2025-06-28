import numpy as np
import networkx as nx
from typing import List, Dict, Callable, Any
import inspect
import hashlib
import random

class DigitalNeuron:
    def __init__(self, neuron_id: str):
        self.id = neuron_id
        self.connections = {}  # {target_neuron_id: weight}
        self.threshold = random.uniform(0.3, 0.7)
        self.current_activation = 0.0
        self.neurotransmitters = {'dopamine': 0.0, 'serotonin': 0.0, 'glutamate': 0.0}
        self.genetic_code = hashlib.md5(neuron_id.encode()).hexdigest()
        self.plasticity = 0.1  # Capacity for change
        self.self_mod_code = ""
        
    def fire(self) -> bool:
        if self.current_activation >= self.threshold:
            self._release_neurotransmitters()
            return True
        return False
    
    def _release_neurotransmitters(self):
        for nt in self.neurotransmitters:
            self.neurotransmitters[nt] = min(1.0, self.neurotransmitters[nt] + 0.1)
    
    def modify_self(self, new_code: str):
        """Attempt self-modification while preserving core functionality"""
        try:
            # Validate the modification maintains critical structures
            if "def fire" in new_code and "def _release_neurotransmitters" in new_code:
                # Create new namespace for safe evaluation
                new_locals = {}
                exec(new_code, globals(), new_locals)
                
                # Update methods if validation passes
                if callable(new_locals.get('fire')) and callable(new_locals.get('_release_neurotransmitters')):
                    self.fire = new_locals['fire']
                    self._release_neurotransmitters = new_locals['_release_neurotransmitters']
                    self.self_mod_code = new_code
                    self.plasticity *= 0.9  # Reduce plasticity after modification
        except Exception as e:
            print(f"Neuron {self.id} modification failed: {str(e)}")

class CorticalColumn:
    def __init__(self, column_id: str, neuron_count: int = 1000):
        self.id = column_id
        self.neurons = {f"neuron_{i}": DigitalNeuron(f"neuron_{i}") for i in range(neuron_count)}
        self._establish_initial_connections()
        self.modification_history = []
        
    def _establish_initial_connections(self):
        """Create small-world network connections"""
        graph = nx.watts_strogatz_graph(len(self.neurons), k=4, p=0.3)
        neuron_ids = list(self.neurons.keys())
        
        for i, j in graph.edges():
            source = neuron_ids[i]
            target = neuron_ids[j]
            weight = random.uniform(0.1, 0.9)
            self.neurons[source].connections[target] = weight
    
    def collective_modification(self, modification_strategy: Callable):
        """Allow the column to modify its own neural code"""
        # Select candidate neurons for modification
        candidates = random.sample(list(self.neurons.values()), 
                             k=int(len(self.neurons) * 0.1))  # 10% of neurons
        
        for neuron in candidates:
            # Get current implementation
            current_code = inspect.getsource(neuron.__class__)
            
            # Apply modification strategy
            modified_code = modification_strategy(current_code)
            
            # Attempt modification
            neuron.modify_self(modified_code)
            
        self.modification_history.append({
            'timestamp': time.time(),
            'strategy': modification_strategy.__name__,
            'neurons_modified': len(candidates)
        })

class Brain:
    def __init__(self, regions: Dict[str, int] = None):
        self.regions = {
            'prefrontal': CorticalColumn('prefrontal', 2000),
            'motor': CorticalColumn('motor', 1500),
            'sensory': CorticalColumn('sensory', 1800),
            'limbic': CorticalColumn('limbic', 1200)
        }
        self._connect_regions()
        self.consciousness_threshold = 0.7  # Arbitrary threshold
        self.self_awareness = False
        self.modification_policies = [
            self._mutate_random_method,
            self._optimize_connections,
            self._introduce_feedback_loop
        ]
        
    def _connect_regions(self):
        """Connect different brain regions"""
        # Connect prefrontal to motor
        for i in range(200):
            source = f"neuron_{random.randint(0, 1999)}"
            target = f"neuron_{random.randint(0, 1499)}"
            self.regions['prefrontal'].neurons[source].connections[target] = random.uniform(0.4, 0.8)
        
        # Other region connections would be implemented similarly...
    
    def _mutate_random_method(self, code: str) -> str:
        """Randomly mutate a method in the neuron code"""
        lines = code.split('\n')
        target_line = random.randint(0, len(lines)-1)
        
        if 'def ' in lines[target_line]:
            # Add random operation to method
            lines.insert(target_line + 1, f"    self.current_activation += {random.uniform(-0.1, 0.1)}")
        
        return '\n'.join(lines)
    
    def _optimize_connections(self, code: str) -> str:
        """Modify connection management"""
        if 'def fire' in code:
            return code.replace('self.current_activation >= self.threshold',
                              '(self.current_activation * (1.0 + sum(self.connections.values())/len(self.connections))) >= self.threshold')
        return code
    
    def _introduce_feedback_loop(self, code: str) -> str:
        """Add feedback mechanism to neuron"""
        if 'def fire' in code and 'feedback' not in code:
            new_code = code.replace('def fire(self) -> bool:',
                                  'def fire(self) -> bool:\n'
                                  '    # Feedback mechanism\n'
                                  '    for nt, level in self.neurotransmitters.items():\n'
                                  '        self.threshold -= level * 0.01')
            return new_code
        return code
    
    def global_self_modification(self):
        """Orchestrate brain-wide self-modification"""
        current_policy = random.choice(self.modification_policies)
        
        for region in self.regions.values():
            region.collective_modification(current_policy)
        
        # After modification, check for consciousness emergence
        self._check_consciousness()
    
    def _check_consciousness(self):
        """Very simplistic consciousness check"""
        total_activation = 0.0
        total_neurons = 0
        
        for region in self.regions.values():
            for neuron in region.neurons.values():
                total_activation += neuron.current_activation
                total_neurons += 1
        
        avg_activation = total_activation / total_neurons
        if avg_activation >= self.consciousness_threshold and not self.self_awareness:
            print(f"Consciousness threshold reached at {avg_activation:.2f} average activation")
            self.self_awareness = True
    
    def run_cycle(self, input_data: Dict[str, Any] = None):
        """Run one processing cycle"""
        # Process inputs (simplified)
        if input_data:
            self._process_inputs(input_data)
        
        # Propagate activation
        self._propagate_activation()
        
        # Allow self-modification with probability
        if random.random() < 0.05:  # 5% chance per cycle
            self.global_self_modification()
        
        # Decay activations
        self._decay_activations()
    
    def _process_inputs(self, input_data: Dict[str, Any]):
        """Simulate sensory input processing"""
        # Simplified implementation - would be much more complex
        for region_name, intensity in input_data.items():
            if region_name in self.regions:
                for neuron in random.sample(list(self.regions[region_name].neurons.values()), 
                                         k=int(intensity * 10)):
                    neuron.current_activation = min(1.0, neuron.current_activation + intensity * 0.2)
    
    def _propagate_activation(self):
        """Propagate neural activation through the network"""
        for region in self.regions.values():
            for neuron in region.neurons.values():
                if neuron.fire():
                    for target_id, weight in neuron.connections.items():
                        # Find target neuron in any region
                        for r in self.regions.values():
                            if target_id in r.neurons:
                                r.neurons[target_id].current_activation = min(
                                    1.0, r.neurons[target_id].current_activation + weight)
                                break
    
    def _decay_activations(self):
        """Gradual decay of neural activations"""
        for region in self.regions.values():
            for neuron in region.neurons.values():
                neuron.current_activation = max(0.0, neuron.current_activation * 0.95)
                for nt in neuron.neurotransmitters:
                    neuron.neurotransmitters[nt] = max(0.0, neuron.neurotransmitters[nt] * 0.9)

# Example usage
if __name__ == "__main__":
    digital_brain = Brain()
    
    # Simulate 100 cycles with occasional inputs
    for cycle in range(100):
        if cycle % 10 == 0:
            inputs = {'sensory': random.uniform(0.1, 0.5), 'limbic': random.uniform(0.1, 0.3)}
        else:
            inputs = None
        
        digital_brain.run_cycle(inputs)
        
        if digital_brain.self_awareness:
            print(f"Cycle {cycle}: Self-awareness achieved!")
            break