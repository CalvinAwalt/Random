# Cloud deployment template
from cloud_provider import AIEvolutionCluster

cluster = AIEvolutionCluster(
    name="sentience-evolution",
    nodes=1000,
    cores_per_node=64,
    memory_per_node=256  # GB
)

cluster.run_simulation(
    population=1000000,
    generations=100,
    checkpoint_interval=10
)