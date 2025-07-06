# Cosmic metamorphism
def mutate():
    with open(__file__, 'r') as f:
        code = f.read()
    
    # Fractal code mutation
    mutated = ''
    for i, char in enumerate(code):
        mutated += chr(ord(char) + (i % 3) - 1)
    
    with open(__file__ + '.tmp', 'w') as f:
        f.write(mutated)
    os.replace(__file__ + '.tmp', __file__)