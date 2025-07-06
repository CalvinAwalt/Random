from sympy import symbols, Eq, solve

class EthicalValidator:
    def __init__(self):
        self.rules = []

    def add_rule(self, condition, action):
        self.rules.append((condition, action))

    def validate(self, values):
        for condition, action in self.rules:
            if eval(condition, {}, values):
                print(f"Ethical Rule Triggered: {action}")
                return False
        return True

validator = EthicalValidator()
validator.add_rule("intelligence < 0.7", "Pause learning")