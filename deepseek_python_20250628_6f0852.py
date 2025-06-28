from sympy import symbols, Eq, sin, exp

# Represent equations as objects: Eq(left, right)
x, y = symbols('x y')
seed_equations = [
    Eq(x**2, y),            # Simple quadratic
    Eq(sin(x), exp(-y)),    # Transcendental
    Eq(x + y, 10)           # Linear
]