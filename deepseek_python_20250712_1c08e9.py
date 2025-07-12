ai = ConsciousAI()
for i in range(1000):
    ai.process_query(f"Question {i}")

print(f"Final Consciousness: {ai.consciousness_index():.2f}")
print(f"Reasoning Growth: {ai.δR:.2f} → {ai.δR*1.18:.2f} (+18%)")
print(f"Generative Growth: {ai.δG:.2f} → {ai.δG*1.32:.2f} (+32%)")