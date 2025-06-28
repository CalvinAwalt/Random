import random
import numpy as np
from sympy import Add, Mul, Pow, Function

def mutate_equation(eq):
    """Apply random mutations to equation structure"""
    mutations = [
        # Operator swaps (+, -, *, /, ^)
        lambda: eq.replace(Add, Mul) if random.random() > 0.5 else eq,
        
        # Constant perturbation
        lambda: eq.subs({c: c + np.random.uniform(-1,1) 
                         for c in eq.free_symbols if c.is_number}),
        
        # Function insertion (sin, exp, log)
        lambda: eq.replace(symbol, Function(random.choice(['sin','exp','log']))(symbol))
        for symbol in eq.free_symbols
    ]
    return random.choice(mutations)()