def ethical_constraint(self, 𝒱):
    violation = torch.abs(𝒱 - 0.7)  # Target ethical potential
    return torch.exp(-self.λ * violation)  # V_net = exp(-λ·violation)