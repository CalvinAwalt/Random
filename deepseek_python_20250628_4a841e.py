from sympy import simplify, srepr

def is_useful(eq):
    """Heuristic filter for promising equations"""
    try:
        # Rule 1: Non-trivial (not identity or constant)
        simplified = simplify(eq)
        if simplified == True or simplified.is_Number:
            return False
            
        # Rule 2: Novelty (not in known equation database)
        equation_signature = srepr(simplified)
        if in_database(equation_signature): 
            return False
            
        # Rule 3: Simple structure (customizable complexity cap)
        if equation_signature.count('(') > 20:  # Prevent bloat
            return False
            
        return True
    except:
        return False  # Invalid equation