'''
    SimScenarioRunner.py - class is responsible for running predefined scenarios. Spawning of ego vechile, actor vehicles and 
                           all crutial components for scenario to take place
'''

import carla
import random
import numpy as np
from scripts.DisplayManager import DisplayManager
from scripts.RGBCamera import RGBCamera
from scripts.PIDLongitudinalController import PIDLongitudinalController

class SimScenarioRunner:
    def __init__(self, client, displayManager, spawnMatrix, egoSpawnId):
        
        self.scenarioId = 1 # by default scenarioId is 0

        self.client = client
        self.world = self.client.get_world()
        self.vehicle = None
        self.vehicle_list = []
        self.cam = None
        self.pidLongitudinalController = None

        self.display_manager = displayManager

        # format:
        # [[scen1x, scen1y], [scen2x, scen2y], ...]
        self.spawnMatrix = spawnMatrix # used for predefining location of actor vehicles trough all scenarios 
        
        self.egoSpawnId = egoSpawnId # list of indexes of spawn points of ego vehicle for each scenario 

        self.initScenario(self.scenarioId) # initialize first scenario

    def getPIDLongitudinalController(self):
        return self.pidLongitudinalController

    def getRGBCamera(self):
        return self.cam

    def getEgoVehicle(self):
        return self.vehicle

    def getVehicleList(self):
        return self.vehicle_list

    def destroyVehicle(self):
        self.vehicle.destroy()

    def getCurrentScenarioId(self):
        return self.scenarioId

    def initScenario(self, scenarioId):

        # save scenarioId
        self.scenarioId = scenarioId

        # if any, destroy vehicles from previous scenario
        if(len(self.vehicle_list)):
            for v in self.vehicle_list:
                v.destroy()

        # Instanciating the vehicle to which we attached the sensors
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle.bmw.*'))
        
        # Spawn actor vehicles in the world 
        self.vehicle_list = self.__spawnScenarioVehicles__(self.spawnMatrix[scenarioId][0], self.spawnMatrix[scenarioId][1])

        # Spawn ego vehicle in the world 
        spawn_points = self.world.get_map().get_spawn_points()
        #for i in range(len(spawn_points)):
        #    print("SP[" + str(i) + "]: " + str(spawn_points[i]))
        self.vehicle = self.world.spawn_actor(bp, spawn_points[self.egoSpawnId[scenarioId]])        
        self.vehicle_list.append(self.vehicle)

        # RGBCamera can be used to spawn RGB Camera as needed
        # and assign each to a grid position 
        self.cam = RGBCamera(self.world, self.display_manager, carla.Transform(carla.Location(x=0, z=1.7), carla.Rotation(yaw=+00)), 
                      self.vehicle, {}, display_pos=[0, 0], saveResolution=7)

        # create PIDLongitudinalController for ego vehicle 'vehicle'
        self.pidLongitudinalController = PIDLongitudinalController(self.vehicle)

    def __spawnScenarioVehicles__(self, vehPosX, vehPosY):
        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*.*')   # don't specify the type of vehicle
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]  # only choose car with 4 wheels

        vehicle_list = []  # keep the spawned vehicle in vehicle_list, because we need to link them with traffic_manager

        spawn_points = self.world.get_map().get_spawn_points()

        for i in range(len(vehPosX)):  # generate the free vehicle
            transform = spawn_points[self.egoSpawnId[self.scenarioId]] 
            location = carla.Location(transform.location.x + vehPosX[i], transform.location.y + vehPosY[i], transform.location.z) 
            point = carla.Transform(location, transform.rotation) 
            vehicle_bp = np.random.choice(vehicle_bps)
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, point)
                vehicle_list.append(vehicle)
            except:
                print('failed')  # if failed, print the hints.
                pass

        # Add vehicles to TM
        if (len(vehicle_list)):
            tm = self.client.get_trafficmanager()  # create a TM object
            tm.global_percentage_speed_difference(45.0)  # set the global speed limitation
            tm_port = tm.get_port()  # get the port of tm. we need add vehicle to tm by this port
            for v in vehicle_list:  # set every vehicle's mode
                v.set_autopilot(True, tm_port)  # you can get those functions detail in carla document
                tm.auto_lane_change(v,False)    # disable auto lane change
                tm.distance_to_leading_vehicle(v, 0.5)  # leave safety distance to leading vehicle 
                tm.vehicle_percentage_speed_difference(v, -20) # drive 20 percent faster than current speed limit

        return vehicle_list