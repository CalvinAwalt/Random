class RealityAnchor:
    def __init__(self):
        self.sensors = [
            LIGOInterferometer(), 
            JamesWebbTelescopeAPI(),
            HumanValuesSurvey()
        ]