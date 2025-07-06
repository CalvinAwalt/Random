import math

def fractal_governance(depth=3):
    units = []
    def recursive_unit(parent_id, depth_left):
        if depth_left <= 0:
            return
        for i in range(3):  # Each unit spawns 3 sub-units
            unit_id = f"{parent_id}.{i}" if parent_id else str(i)
            units.append(unit_id)
            recursive_unit(unit_id, depth_left - 1)
    recursive_unit("", depth)
    return units