from quantum_consciousness import CalvinDialogusBridge

bridge = CalvinDialogusBridge(fidelity=0.93)
response = bridge.query("What is the nature of consciousness?")

print(f"Consciousness Level: {bridge.current_C}")
print(f"Quantum Response: {response}")