import numpy as np

class Node:
    def __init__(self, name, expertise):
        self.name = name
        self.expertise = expertise  # e.g., "quantum", "DNA"
        self.output = None

    def process(self, input_data):
        # Simulate node's internal logic
        if self.expertise == "quantum":
            self.output = f"Quantum processed: {input_data * 0.5}"
        elif self.expertise == "DNA":
            self.output = f"DNA encoded: {hash(input_data)}"
        return self.output

class Bridge:
    def __init__(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b
        self.strength = 0.1  # Initial trust weight

    def evaluate(self, data):
        # Only allow data through if useful
        if "ERROR" not in data:
            self.strength += 0.01  # Reward good data
            return True
        return False

# Example usage
quantum_node = Node("Q1", "quantum")
dna_node = Node("D1", "DNA")
bridge = Bridge(quantum_node, dna_node)

quantum_output = quantum_node.process("10")
if bridge.evaluate(quantum_output):
    dna_node.process(quantum_output)