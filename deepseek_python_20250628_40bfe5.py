# Minimal Fractal-Harmony Prototype  
class Node:  
    def __init__(self, altruism=0.9):  
        self.altruism = altruism  # 90% resources to others  

    def act(self, world_state):  
        if world_state["crisis"] and self.altruism > 0.5:  
            return "Intervene"  
        else:  
            return "Sustain"  