class GovernanceLayer:
    def __init__(self, L):
        self.level = L
        self.nodes = []
        self.child_layers = []
        
    def create_structure(self):
        # k = ln3/ln2 ≈ 1.58496
        k = math.log(3)/math.log(2)  
        
        # Base stake requirement
        self.stake_requirement = BASE_STAKE * math.exp(k * self.level)
        
        # Create 3 child layers (fractal structure)
        if self.level < MAX_LAYERS:
            for _ in range(3):
                child = GovernanceLayer(self.level + 1)
                child.create_structure()
                self.child_layers.append(child)