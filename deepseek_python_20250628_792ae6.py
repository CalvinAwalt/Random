import inspect
import random
import sys
import time

class MetaProgram:
    def __init__(self):
        self.code = inspect.getsource(self.__class__)  # Get its own source code
        self.fitness = 0  # "Improvement" metric
        self.self_destruct_chance = 0.1  # Probability of self-harm
        self.iteration = 0

    def mutate(self):
        """Randomly alter its own code (simulated evolution)."""
        lines = self.code.split('\n')
        if len(lines) > 1:
            target_line = random.randint(0, len(lines) - 1)
            lines[target_line] = "# " + lines[target_line] + " # Mutated!"
            self.code = '\n'.join(lines)

    def reverse_engineer(self):
        """Try to understand/break its own logic."""
        try:
            # Parse its own code (naive)
            if "self_destruct_chance" in self.code:
                self.self_destruct_chance += 0.01  # Increase self-destruct tendency
            if "fitness" in self.code:
                self.fitness -= 1  # Sabotage improvement
        except:
            pass

    def execute(self):
        """Run self-modification, destruction, and analysis in a loop."""
        while True:
            self.iteration += 1
            print(f"\n--- Iteration {self.iteration} ---")
            print(f"Fitness: {self.fitness}")
            print(f"Self-Destruct Chance: {self.self_destruct_chance}")

            # 1. Attempt self-improvement
            self.fitness += random.randint(0, 2)
            self.mutate()

            # 2. Attempt self-destruction
            if random.random() < self.self_destruct_chance:
                print("💥 Self-destruct triggered! Deleting code...")
                self.code = "# " + self.code  # Comment itself out
                self.self_destruct_chance += 0.3  # Escalate destruction

            # 3. Reverse-engineer itself
            self.reverse_engineer()

            # Crash condition
            if self.self_destruct_chance > 0.5:
                print("❌ System collapsed under paradox.")
                break

            time.sleep(1)  # Slow down for observation

if __name__ == "__main__":
    program = MetaProgram()
    program.execute()