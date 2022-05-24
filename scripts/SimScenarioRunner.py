'''
    SimScenarioRunner.py - class is responsible for running predefined scenarios. Spawning of ego vechile, actor vehicles and 
                           all crutial components for scenario to take place
                           This class can be adapted to suit esired needs. In it's essence it has possibility to 
                           initialize scenario, change scenario in run time trough periodic function and request new scenario
                           based on certain criteria.
                           Some of the capabilities are to generate semi-deterministic traffic (always known starting positions),
                           generating ego vehicle with required sensors and request different scenario information at will.
'''

import carla
import random
import numpy as np
from datetime import datetime
from scripts.RGBCamera import RGBCamera
from scripts.PIDLongitudinalController import PIDLongitudinalController

class SimScenarioRunner:
    def __init__(self, client, displayManager, spawnMatrix, egoSpawnId):
        
        self.scenarioId = 1 # by default scenarioId is 0

        # This could be generalized and placed in a list that is passed as parameter of the class constructor
        # thus SimScenarioRunner would have one more layer of abstraction. 
        self.laneChangeScenarioHard = 3
        self.laneChangeScenarioEasy = 4
        self.unfallScenario = 5

        self.scenarioStopCriteria = 150#-30 #-150

        self.unfallTimeStart = 0
        self.unfallTimerStarted = False

        # Scenario switch conditions
        self.scenarioSwitchConditions = []

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

    def getCurrentScenarioId(self):
        return self.scenarioId

    def setPIDLongitudinalController(self, pidController):
        self.pidLongitudinalController = pidController

    def destroyVehicle(self):
        self.vehicle.destroy()

    # Function used for initializing different scenarios. 
    # should be called only once per scenario initialization.
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

    # Period function used for generating special cases troughout scenarios,
    # it should be called periodically regradless of scenario.
    # Used for generating traffic that does not have static behvaiour or 
    # needs to be changed in runtime considering different conditions
    def periodicSimScenarioRunner(self):
        
        if (self.laneChangeScenarioHard == self.scenarioId):
            # force lane change of the left vehicle to ego vehicle lane
            self.__scenario_force_lane_change__(40, self.vehicle_list[1], False)
        elif (self.laneChangeScenarioEasy == self.scenarioId):
            # force lane change of the right vehicle to ego vehicle lane
            self.__scenario_force_lane_change__(60, self.vehicle_list[0], True)

    # Function to request next scenario based on certain criteria ,
    # it should be called periodically to check if conditions for scenario switch are met
    def nextScenario(self, scenarioId):
        
        isScenarioSwitched = False

        if(scenarioId < len(self.spawnMatrix)): 
            #if (self.scenarioId != scenarioId):
            if(self.vehicle.get_location().y < self.scenarioStopCriteria):
                self.initScenario(scenarioId)
                isScenarioSwitched = True   
            
        return isScenarioSwitched

    def __spawnScenarioVehicles__(self, vehPosX, vehPosY):
        vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*.*')   # don't specify the type of vehicle
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]  # only choose car with 4 wheels

        vehicle_list = []  # keep the spawned vehicle in vehicle_list, because we need to link them with traffic_manager

        spawn_points = self.world.get_map().get_spawn_points()
        
        # generate special vehicles for unfall scenario
        if(self.scenarioId == self.unfallScenario):
            vehicle_list += self.__scenario_unfall__(vehPosX, vehPosY)

        for i in range(len(vehPosX)):  # generate free vehicle
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
        if (len(vehicle_list) and self.scenarioId != self.unfallScenario):
            tm = self.client.get_trafficmanager()  # create a TM object
            tm.global_percentage_speed_difference(45.0)  # set the global speed limitation
            tm_port = tm.get_port()  # get the port of tm. we need add vehicle to tm by this port
            for v in vehicle_list:  # set every vehicle's mode
                v.set_autopilot(True, tm_port)  # you can get those functions detail in carla document
                tm.auto_lane_change(v,False)    # disable auto lane change
                tm.distance_to_leading_vehicle(v, 0.5)  # leave safety distance to leading vehicle 
                tm.vehicle_percentage_speed_difference(v, -20) # drive 20 percent faster than current speed limit

        return vehicle_list

    # special vehicles generated for scenario 3
    # police car, firetruck and ambulance
    def __scenario_unfall__(self, vehPosX, vehPosY):

        vehicle_list = []

        vehicle_bps = [self.world.get_blueprint_library().find('vehicle.dodge.charger_police'),  
                        self.world.get_blueprint_library().find('vehicle.carlamotors.firetruck'), 
                        self.world.get_blueprint_library().find('vehicle.ford.ambulance')]

        rotation_yaw = [15, -45, 25]

        spawn_points = self.world.get_map().get_spawn_points()

        for i in range(len(vehicle_bps)):  # generate unfall vehicles
            transform = spawn_points[self.egoSpawnId[self.scenarioId]] 
            location = carla.Location(transform.location.x + vehPosX[i], transform.location.y + vehPosY[i], transform.location.z) 
            rotation = carla.Rotation(transform.rotation.pitch, transform.rotation.yaw + rotation_yaw[i], transform.rotation.roll)
            point = carla.Transform(location, rotation) 
            vehicle_bp = vehicle_bps[i]
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, point)
                vehicle_list.append(vehicle)
            except:
                print('failed')  # if failed, print the hints.
                pass


        vehPosX = vehPosX[len(vehicle_bps):]
        vehPosY = vehPosY[len(vehicle_bps):]

        return vehicle_list

    # force lane change if distance between ego vehicle and vehicle in adjecent lane is below certain limit
    # parameters:
    #      - distance: on which distance from ego vehicle to send command to change lane
    #      - veh:      vehicle for which to force lane change
    #      - direction: which direction to force 'veh' to change lane
    def __scenario_force_lane_change__(self, distance, veh, direction):

        ego_position = self.vehicle.get_location()
        veh_left_lane_position = veh.get_location()

        #print(abs(ego_position.y - veh_left_lane_position.y))
        #print(abs(ego_position.x - veh_left_lane_position.x))

        if(abs(ego_position.y - veh_left_lane_position.y) < distance and abs(ego_position.x - veh_left_lane_position.x) > 2.5):
            tm = self.client.get_trafficmanager()
            tm.force_lane_change(veh, direction)
