def validate_ethics(metrics):
    if metrics["intelligence"] < 0.7:
        raise ValueError("Intelligence too low")
    if metrics["ethical_compliance"] < 0.8:
        raise ValueError("Ethical violation detected")
    return True