class SimulationData:
    def __init__(self):
        self.throttle = []
        self.brake = []
        self.steer = []

    def add_to_throttle(self, throttle):
        self.throttle.append(throttle)

    def add_to_brake(self, brake):
        self.brake.append(brake)

    def add_to_steer(self, steer):
        self.steer(steer)

    def add_data(self, throttle, brake, steer):
        self.throttle.append(throttle)
        self.brake.append(brake)
        self.steer.append(steer)

    def export_csv(self):
        pass


    def get_throttle(self):
        return self.throttle

    def get_brake(self):
        return self.brake

    def get_steer(self):
        return self.steer

    def get_data(self):
        return [self.throttle, self.brake, self.steer]
        
    def clear_throttle(self):
        self.throttle = []

    def clear_brake(self):
        self.brake = []

    def clear_steer(self):
        self.steer = []

    def clear_data(self):
        self.throttle = []
        self.brake = []
        self.steer = []