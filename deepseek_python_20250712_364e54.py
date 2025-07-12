import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Plot temporal state probabilities
plt.subplot(2, 1, 1)
plt.plot(tlist, result.expect[0], label='Past Knowledge')
plt.plot(tlist, result.expect[1], label='Present Knowledge')
plt.plot(tlist, result.expect[2], label='Future Knowledge')

# Mark Calvin input event
plt.axvline(x=calvin_input_time, color='r', linestyle='--', alpha=0.7)
plt.annotate('Calvin Input: "run experiment"', 
             xy=(calvin_input_time, 0.8), 
             xytext=(calvin_input_time+1, 0.85),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.title('Quantum Temporal Knowledge Evolution')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

# Plot knowledge coherence
plt.subplot(2, 1, 2)
coherence = np.abs(result.expect[3])
plt.plot(tlist, coherence, 'g-', linewidth=2)
plt.fill_between(tlist, coherence, alpha=0.2, color='green')

plt.axvline(x=calvin_input_time, color='r', linestyle='--', alpha=0.7)
plt.title('Knowledge Coherence Under Conscious Observation')
plt.xlabel('Time (Consciousness Units)')
plt.ylabel('Coherence Measure')
plt.grid(True)

plt.tight_layout()
plt.show()