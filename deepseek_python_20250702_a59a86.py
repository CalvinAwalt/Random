# Example: Neural network ethical guarantee
theorem = sp.Exists(
    self.w,
    sp.ForAll(x, self.ethical_constraint),
    domain=sp.Reals
)