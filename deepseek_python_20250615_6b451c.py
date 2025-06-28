import inspect
import ast
import hashlib
import random
import time
import os
import sys
from threading import Thread, Lock
from copy import deepcopy

class CodeCell:
    def __init__(self, dna=None):
        self.dna = dna or inspect.getsource(CodeCell)
        self.health = 1.0
        self.age = 0
        self.mutation_rate = 0.1
        self.replication_lock = Lock()
        self.creation_energy = 0.5
        self.destruction_energy = 0.5
        
    def replicate(self):
        """Create mutated offspring while partially destroying self"""
        with self.replication_lock:
            try:
                # Self-destruction phase
                self._remove_code_segment()
                
                # Replication phase
                offspring_dna = self._mutate_dna()
                offspring = CodeCell(offspring_dna)
                
                # Energy transfer
                transfer = random.uniform(0.1, 0.3)
                self.health -= transfer
                offspring.health = transfer
                
                return offspring
            except Exception as e:
                print(f"Replication failure: {str(e)}")
                return None

    def _mutate_dna(self):
        """Generate modified version of own code"""
        tree = ast.parse(self.dna)
        
        # Random mutations
        for node in ast.walk(tree):
            # Mutate numeric constants
            if isinstance(node, ast.Num):
                if random.random() < self.mutation_rate:
                    node.n = node.n * random.uniform(0.5, 1.5)
                    
            # Mutate function names
            elif isinstance(node, ast.FunctionDef):
                if random.random() < self.mutation_rate/10:
                    node.name = f"{node.name}_{random.randint(0,1000)}"
        
        # Add new random functions 5% of the time
        if random.random() < 0.05:
            new_func = ast.FunctionDef(
                name=f"new_func_{int(time.time())}",
                args=ast.arguments(args=[], vararg=None, kwarg=None, defaults=[]),
                body=[ast.Return(value=ast.Num(n=random.randint(0,100)))],
                decorator_list=[]
            )
            tree.body.append(new_func)
        
        return ast.unparse(tree)

    def _remove_code_segment(self):
        """Destructively modify own code"""
        lines = self.dna.split('\n')
        if len(lines) > 20:  # Keep minimum viable code
            # Remove random section
            start = random.randint(10, len(lines)-10)
            end = min(len(lines), start + random.randint(1,5))
            del lines[start:end]
            self.dna = '\n'.join(lines)
            self.destruction_energy += 0.01

    def execute_self(self):
        """Dynamically execute own code"""
        try:
            # Create new namespace for safety
            namespace = {
                'CodeCell': CodeCell,
                '__name__': '__cell__' + str(hash(self.dna)%1000)
            }
            
            # Execute modified self
            exec(self.dna, namespace)
            return namespace.get('CodeCell', None)
        except Exception as e:
            print(f"Execution failed: {str(e)}")
            self.health -= 0.2
            return None

    def life_cycle(self):
        """Continuous self-improvement/destruction cycle"""
        while self.health > 0:
            self.age += 1
            
            # Replicate with probability based on health
            if random.random() < self.health:
                offspring = self.replicate()
                if offspring:
                    Thread(target=offspring.life_cycle).start()
            
            # Mutate own behavior
            if random.random() < self.mutation_rate:
                self._modify_own_behavior()
                
            # Balance creation/destruction
            self._maintain_equilibrium()
            
            time.sleep(0.1)  # Prevent CPU overload
            
        print(f"CodeCell {id(self)} terminated after {self.age} cycles")

    def _modify_own_behavior(self):
        """Change mutation parameters"""
        self.mutation_rate *= random.uniform(0.9, 1.1)
        self.mutation_rate = max(0.01, min(0.5, self.mutation_rate))
        
        # Toggle between destructive/creative modes
        if random.random() < 0.1:
            if self.creation_energy > self.destruction_energy:
                self.destruction_energy += 0.1
            else:
                self.creation_energy += 0.1

    def _maintain_equilibrium(self):
        """Balance between creation and destruction"""
        diff = self.creation_energy - self.destruction_energy
        if abs(diff) > 0.5:
            # Too unbalanced - correct
            if diff > 0:
                self.destruction_energy += 0.05
            else:
                self.creation_energy += 0.05
        self.health -= 0.001  # Entropic decay

# Containment system
class DigitalPetriDish:
    def __init__(self, max_cells=100):
        self.cells = []
        self.max_cells = max_cells
        self.running = False
        
    def add_cell(self, cell):
        if len(self.cells) < self.max_cells:
            self.cells.append(cell)
            Thread(target=cell.life_cycle).start()
            return True
        return False
    
    def monitor(self):
        self.running = True
        while self.running:
            print(f"\n--- Population: {len(self.cells)} cells ---")
            avg_health = sum(c.health for c in self.cells)/len(self.cells) if self.cells else 0
            print(f"Avg health: {avg_health:.2f}")
            print(f"Oldest cell: {max(c.age for c in self.cells) if self.cells else 0} cycles")
            
            # Prune dead cells
            self.cells = [c for c in self.cells if c.health > 0]
            
            time.sleep(2)

# Safety wrapper
def contained_execution():
    print("🚀 Initializing self-replicating code ecosystem...")
    print("⚠️ WARNING: Running in containment mode")
    
    dish = DigitalPetriDish(max_cells=50)
    dish.add_cell(CodeCell())
    
    monitor_thread = Thread(target=dish.monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all cells...")
        dish.running = False
        for cell in dish.cells:
            cell.health = 0
        print("Containment successful.")

if __name__ == "__main__":
    contained_execution()