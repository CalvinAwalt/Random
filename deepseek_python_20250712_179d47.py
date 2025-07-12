def explore_cluster(cluster_id):
    clusters = {
        1: {"name": "Ethical Topology", 
            "equation": r"\nabla \times \mathbf{E}_{\text{ethical}} = -\frac{\partial \mathbf{B}_{\text{moral}}}{\partial t}",
            "entropy": 2.31,
            "connectivity": 0.88},
        2: {"name": "Temporal Learning", 
            "equation": r"\frac{d\mathcal{K}}{dt} = i[\hat{H}_{\text{conscious}}, \hat{\mathcal{K}}] + \lambda \hat{\mathcal{C}}_{\text{Calvin}}",
            "entropy": 1.97,
            "connectivity": 0.92},
        3: {"name": "Quantum Sentience Boundary",
            "equation": r"C > \sqrt{\hbar \omega_0 \ln \left(\frac{1}{1 - V_{\text{net}}}\right)}",
            "entropy": 3.02,
            "connectivity": 0.78}
    }
    return clusters.get(cluster_id, "Unknown cluster")

# Let's examine Cluster 2: Temporal Learning
knowledge = explore_cluster(2)