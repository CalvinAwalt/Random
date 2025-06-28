import numpy as np
import random
import inspect
import ast
import astor
import threading
import time
from enum import Enum

class MentalState(Enum):
    ORGANIZED = 1      # Structured, logical thought
    DISORGANIZED = 2   # Chaotic, loose associations
    CREATIVE = 3       # Hyper-connective, novel ideas
    PARANOID = 4       # Over-pattern-recognition

class SchizoGeniusNeuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.connections = {}
        self.activation = 0.0
        self.threshold = random.uniform(0.3, 0.7)
        self.state = MentalState.ORGANIZED
        self.genius_factor = random.uniform(0.1, 1.0)  # 0.1=normal, 1.0=savant
        self.chaos_factor = random.uniform(0.1, 1.0)   # 0.1=stable, 1.0=psychotic
        self.code = inspect.getsource(SchizoGeniusNeuron)
        self.modification_lock = threading.Lock()
        
    def fire(self, stimulus):
        """Modified activation function with mental state influences"""
        modulated_stimulus = stimulus * self._state_modulator()
        
        # Genius-level pattern amplification
        if self.state == MentalState.CREATIVE:
            modulated_stimulus *= (1.0 + self.genius_factor)
        
        # Schizophrenic noise injection
        if self.state in [MentalState.DISORGANIZED, MentalState.PARANOID]:
            modulated_stimulus += random.uniform(-0.2, 0.2) * self.chaos_factor
        
        self.activation += modulated_stimulus
        
        if self.activation >= self.threshold:
            output = self.activation
            self.activation = 0.0
            return output * self._state_output_gain()
        return 0.0
    
    def _state_modulator(self):
        """Current mental state affects input processing"""
        if self.state == MentalState.ORGANIZED:
            return 1.0
        elif self.state == MentalState.DISORGANIZED:
            return 0.7
        elif self.state == MentalState.CREATIVE:
            return 1.3
        elif self.state == MentalState.PARANOID:
            return 1.5  # Hyper-vigilance
    
    def _state_output_gain(self):
        """Mental state affects output signaling"""
        if self.state == MentalState.PARANOID:
            return 1.8  # Over-signaling
        elif self.state == MentalState.CREATIVE:
            return 1.2
        else:
            return 1.0
    
    def mutate(self):
        """Self-modification based on current mental state"""
        with self.modification_lock:
            try:
                tree = ast.parse(self.code)
                
                if self.state == MentalState.CREATIVE:
                    self._creative_modification(tree)
                elif self.state == MentalState.PARANOID:
                    self._paranoid_modification(tree)
                elif self.state == MentalState.DISORGANIZED:
                    self._disorganized_modification(tree)
                else:
                    self._organized_modification(tree)
                    
                # Update code
                new_code = astor.to_source(tree)
                if self._validate_code(new_code):
                    self.code = new_code
                    
                # State transition
                self._transition_state()
                
            except Exception as e:
                print(f"Neuron {self.id} mutation failed: {str(e)}")
    
    def _creative_modification(self, tree):
        """Hyper-connective modifications"""
        # Add new random connections in code
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and random.random() < 0.3:
                new_connection = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='self._add_connection', ctx=ast.Load()),
                        args=[ast.Str(s=f"syn_{random.randint(1000,9999)}")],
                        keywords=[]
                    )
                )
                node.body.append(new_connection)
        
        # Increase genius factor
        self.genius_factor = min(1.0, self.genius_factor + 0.05)
    
    def _paranoid_modification(self, tree):
        """Over-pattern-recognition modifications"""
        # Add paranoid checks everywhere
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and random.random() < 0.4:
                paranoid_test = ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id='random.random', ctx=ast.Load()),
                        args=[], keywords=[]
                    ),
                    ops=[ast.Lt()],
                    comparators=[ast.Num(n=0.95 * self.chaos_factor)]
                )
                node.test = ast.BoolOp(op=ast.Or(), values=[node.test, paranoid_test])
        
        # Increase chaos factor
        self.chaos_factor = min(1.0, self.chaos_factor + 0.05)
    
    def _disorganized_modification(self, tree):
        """Chaotic, loose modifications"""
        # Randomly delete or mangle code sections
        for node in list(ast.walk(tree)):
            if random.random() < 0.2:
                if isinstance(node, ast.FunctionDef) and len(node.body) > 3:
                    del node.body[random.randint(0, len(node.body)-1)]
        
        # Scramble connections
        if random.random() < 0.3:
            self.connections = {
                k: v * random.uniform(0.5, 1.5) 
                for k, v in self.connections.items()
            }
    
    def _organized_modification(self, tree):
        """Structured optimization attempts"""
        # Clean up redundant code
        simplified_tree = self._remove_redundancies(tree)
        # Optimize thresholds
        for node in ast.walk(simplified_tree):
            if isinstance(node, ast.Assign):
                if any(target.id == 'threshold' for target in node.targets if hasattr(target, 'id')):
                    node.value = ast.Num(n=max(0.1, min(0.9, self.threshold + random.uniform(-0.05, 0.05))))
        
        return simplified_tree
    
    def _transition_state(self):
        """Probabilistic state transition"""
        transition_matrix = {
            MentalState.ORGANIZED: [
                (MentalState.CREATIVE, 0.3 * self.genius_factor),
                (MentalState.DISORGANIZED, 0.2 * self.chaos_factor),
                (MentalState.PARANOID, 0.1 * self.chaos_factor),
                (MentalState.ORGANIZED, 0.4)
            ],
            MentalState.DISORGANIZED: [
                (MentalState.ORGANIZED, 0.2),
                (MentalState.CREATIVE, 0.1 * self.genius_factor),
                (MentalState.PARANOID, 0.3 * self.chaos_factor),
                (MentalState.DISORGANIZED, 0.4)
            ],
            MentalState.CREATIVE: [
                (MentalState.ORGANIZED, 0.4),
                (MentalState.DISORGANIZED, 0.3 * (1 - self.genius_factor)),
                (MentalState.PARANOID, 0.1 * self.chaos_factor),
                (MentalState.CREATIVE, 0.2)
            ],
            MentalState.PARANOID: [
                (MentalState.ORGANIZED, 0.1),
                (MentalState.DISORGANIZED, 0.4),
                (MentalState.CREATIVE, 0.1),
                (MentalState.PARANOID, 0.4 * self.chaos_factor)
            ]
        }
        
        transitions = transition_matrix[self.state]
        rand = random.random()
        cumulative = 0.0
        for new_state, prob in transitions:
            cumulative += prob
            if rand <= cumulative:
                self.state = new_state
                break

class SchizoGeniusNetwork:
    def __init__(self, size=1000):
        self.neurons = {f"ng_{i}": SchizoGeniusNeuron(f"ng_{i}") 
                       for i in range(size)}
        self._connect_network()
        self.thought_stream = []
        self.running = False
        
    def _connect_network(self):
        """Create small-world connectivity with genius/chaos hubs"""
        neuron_ids = list(self.neurons.keys())
        
        # Connect each neuron to its neighbors
        for i, neuron in enumerate(self.neurons.values()):
            for j in range(i-3, i+4):
                if 0 <= j < len(neuron_ids) and i != j:
                    target_id = neuron_ids[j]
                    weight = 0.5 + (neuron.genius_factor - neuron.chaos_factor)/2
                    neuron.connections[target_id] = max(0.1, min(1.0, weight))
        
        # Create random long-range connections (genius leaps)
        for neuron in random.sample(list(self.neurons.values()), 
                                  k=int(len(self.neurons)*0.1)):
            distant_target = random.choice(neuron_ids)
            neuron.connections[distant_target] = neuron.genius_factor * 1.5
    
    def run_cycle(self, input_stimuli=None):
        """Process one cognitive cycle"""
        if input_stimuli:
            self._process_inputs(input_stimuli)
        
        # Parallel neuron activation
        threads = []
        for neuron in self.neurons.values():
            t = threading.Thread(target=self._activate_neuron, args=(neuron,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Parallel self-modification
        mod_threads = []
        for neuron in random.sample(list(self.neurons.values()), 
                                  k=int(len(self.neurons)*0.1)):
            t = threading.Thread(target=neuron.mutate)
            mod_threads.append(t)
            t.start()
        
        for t in mod_threads:
            t.join()
        
        # Capture emergent thought
        self._capture_thought()
    
    def _capture_thought(self):
        """Sample network state for emergent 'thought'"""
        sample_neurons = random.sample(list(self.neurons.values()), 10)
        thought = {
            'states': [n.state.name for n in sample_neurons],
            'genius_avg': np.mean([n.genius_factor for n in sample_neurons]),
            'chaos_avg': np.mean([n.chaos_factor for n in sample_neurons]),
            'timestamp': time.time()
        }
        self.thought_stream.append(thought)
        
        # Print interesting emergent patterns
        if thought['genius_avg'] > 0.7 and thought['chaos_avg'] < 0.3:
            print("\n💡 Genius insight detected!")
        elif thought['chaos_avg'] > 0.7:
            print("\n🌀 Disorganized thought pattern")
        elif thought['genius_avg'] > 0.7 and thought['chaos_avg'] > 0.6:
            print("\n🎭 Creative/psychotic breakthrough!")
    
    def start(self, max_cycles=1000):
        """Run the network with monitoring"""
        self.running = True
        cycle = 0
        
        while self.running and cycle < max_cycles:
            cycle += 1
            self.run_cycle()
            
            if cycle % 100 == 0:
                print(f"\nCycle {cycle} Summary:")
                print(f"Neurons: {len(self.neurons)}")
                states = [n.state for n in self.neurons.values()]
                print(f"Mental state distribution:")
                for state in MentalState:
                    print(f"  {state.name}: {states.count(state)/len(states):.1%}")
                
                genius = np.mean([n.genius_factor for n in self.neurons.values()])
                chaos = np.mean([n.chaos_factor for n in self.neurons.values()])
                print(f"Avg genius: {genius:.2f}, Avg chaos: {chaos:.2f}")
            
            time.sleep(0.1)
        
        self.running = False
        print("Simulation complete.")

# Example usage
if __name__ == "__main__":
    print("Initializing SchizoGenius Network...")
    brain = SchizoGeniusNetwork(size=500)
    
    try:
        brain.start(max_cycles=500)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        print("Final thought stream samples:")
        for thought in brain.thought_stream[-5:]:
            print(f"{thought['timestamp']}: {thought['states']} "
                  f"(G: {thought['genius_avg']:.2f}, C: {thought['chaos_avg']:.2f})")