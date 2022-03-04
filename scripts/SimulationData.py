import glob
import os

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

    def export_csv(self, img_path):
        # Format: img_name, throttle, break, steer

        csv_file = open(os.path.join(img_path, 'MainSimulation_out.csv'), "w") 
        filenames = []

        # get image names
        os.chdir(img_path)
        for imagePath in glob.glob("*.jpg"):
            filenames.append(os.path.basename(imagePath))

        # write first line in output .csv file
        line = 'img_name, throttle, break, steer\n'

        csv_file.write(line)

        # write the rest of recorded data to output .csv file
        for i in range(len(filenames)):
            line = filenames[i] + ',' + str(self.throttle[i]) + ',' + str(self.brake[i]) + ',' + str(self.steer[i]) + '\n'
            csv_file.write(line)

        csv_file.close()

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