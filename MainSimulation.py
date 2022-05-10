"""
    Main simulation script used in ADAS-ML-DistanceKeeping software.
    Script renders one RGBCamera in the pygame window, uses 
    YOLOv3 for localization and classification of current traffic and 
    CNN for controling the ego vehicle.
"""

import glob
import os
import sys

import cv2
import tensorflow as tf
from PIL import Image, ImageOps

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import numpy as np
import math
import datetime

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

""" Import all essential custom classes 
"""
from scripts.CustomTimer import CustomTimer
from scripts.DisplayManager import DisplayManager
from scripts.RGBCamera import RGBCamera
from scripts.SimulationData import SimulationData
from cnn.cnn import create_cnn_model
from util.PIDLongitudinalController import PIDLongitudinalController
from util.SimpleMovingAverage import SimpleMovingAverage

#################################################################################################################
datasetSavePath = 'camera_sensors_output/center_town01_4' # Path where images and .csv file will be saved in training mode
#################################################################################################################

#################################################################################################################
# CNN parameters
model_name = 'CNN_distanceKeeping.h5'
#model_path = 'cnn/model_out/model_out_center_it4_b4_200_200_lr_0005/'
model_path = 'cnn/model_out/model_out_center_it6_b4_200_200_dset_16568_final/'
in_width = 200      # width of the input in the CNN model
in_heigth = 200     # heigth of the input in the CNN model
in_channels = 1     # number of input channels to the CNN model 
output_no = 5       # number of outputs of the CNN model
#################################################################################################################

#################################################################################################################
# constants
gSafeToAccThreshold = 0.4
gNotSafeToAccThreshold = 0.6
gBrakeKMHStep = 2.5
gKeepDistanceSpeedSubtract = 5
gSMABufferLen = 5
#################################################################################################################

#################################################################################################################
# ego vehicle parameters for town06
ego_spawn_point = 2 # ego vehicle initial spawn point
ego_location_spawn = carla.Location(x=586.856873, y=-17.063015, z=0.300000) #ego vehicle initial spawn location
ego_transform_spawn = carla.Rotation(pitch=0.000000, yaw=-180.035, roll=0.000000) # ego vehicle initial spawn rotation
ego_speed_limit = 70 # ego vehicle speed limit in km/h
ego_keep_distance_speed = 0 # store vehicle speed when neural network predicts that the vehicle should keep distance
ego_keep_distance_saved = False # store boolean whether ego_keep_distance_speed is saved or not
#################################################################################################################

#################################################################################################################
# global variables
gRecord_data = False # boolean variable to start or stop recording simulation data for training purposes 
gData_collected = False # boolean variable that marks data was collected in training mode

nativeImgWidth = 1280 # native image width for RGBCamera sensor
nativeImgHeight = 720 # native image height for RGBCamera sensor

targetImgWidth = 600   # image width on which CNN was trained
targetImgHeight = 370    # image heigth on which CNN was trained
#################################################################################################################

def spawn_vehicles_around_ego_vehicles(client, world, ego_vehicle, radius, spawn_points, numbers_of_vehicles):
    # parameters:
    # ego_vehicle :: your target vehicle
    # radius :: the distance limitation between ego-vehicle and other free-vehicles
    # spawn_points  :: the available spawn points in current map
    # numbers_of_vehicles :: the number of free-vehicles around ego-vehicle that you need
    np.random.shuffle(spawn_points)  # shuffle  all the spawn points
    ego_location = ego_vehicle.get_location()
    accessible_points = []
    for spawn_point in spawn_points:
        dis = math.sqrt((ego_location.x-spawn_point.location.x)**2 + (ego_location.y-spawn_point.location.y)**2)
        # it also can include z-coordinate,but it is unnecessary
        if dis < radius:
            #print(dis)
            accessible_points.append(spawn_point)

    vehicle_bps = world.get_blueprint_library().filter('vehicle.*.*')   # don't specify the type of vehicle
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]  # only choose car with 4 wheels

    vehicle_list = []  # keep the spawned vehicle in vehicle_list, because we need to link them with traffic_manager
    if len(accessible_points) < numbers_of_vehicles:
        # if your radius is relatively small,the satisfied points may be insufficient
        numbers_of_vehicles = len(accessible_points)

    for i in range(numbers_of_vehicles):  # generate the free vehicle
        point = accessible_points[i]
        vehicle_bp = np.random.choice(vehicle_bps)
        try:
            vehicle = world.spawn_actor(vehicle_bp, point)
            vehicle_list.append(vehicle)
        except:
            print('failed')  # if failed, print the hints.
            pass
        
    # you also can add those free vehicle into trafficemanager,and set them to autopilot.
    # Only need to get rid of comments for below code. Otherwise, the those vehicle will be static
    tm = client.get_trafficmanager()  # create a TM object
    tm.global_percentage_speed_difference(50.0)  # set the global speed limitation
    tm_port = tm.get_port()  # get the port of tm. we need add vehicle to tm by this port
    for v in vehicle_list:  # set every vehicle's mode
        v.set_autopilot(True, tm_port)  # you can get those functions detail in carla document
        tm.ignore_lights_percentage(v, 0)
        tm.distance_to_leading_vehicle(v, 0.5)
        tm.vehicle_percentage_speed_difference(v, -20)

def ego_vehicle_control(vehicle, cnn_predictions, pidLongitudinalController):
    # variable to store current speed when neural network predicts that 
    # ego vehicle should keep distance to the leading vehicle
    global ego_keep_distance_speed
    global ego_keep_distance_saved

    # extract cnn predictions
    lSafeToAcc = cnn_predictions[4]

    velocity_kmh = pidLongitudinalController.get_speed(vehicle)

    # check lSafeToAcc parameter
    # if neural network predicts that vehicle should accelerate, acceleration should be possible if within road limit
    # if neural netowrk predicts that vehicle should keep distance, the speed should be stay the same as current speed
    # if neural network predicts that vehicle should brake, the speed should decrease to the point where neural network 
    #       predicts that it should keep distance or possibly accelerate
    if(lSafeToAcc <= gSafeToAccThreshold): 
        # it is safe to accelerate, accelerate within road limit
        pid_control = pidLongitudinalController.run_step(ego_speed_limit)

        # if ego_keep_distance_speed is saved in previous iterations and state changed to state when 
        # neural network predicted that it is safe to accelerate, change boolean ego_keep_distance_saved to
        # False
        if (ego_keep_distance_saved == True):
            ego_keep_distance_saved = False

        print("Accelerate " + str(pid_control))
    elif (lSafeToAcc > gSafeToAccThreshold and lSafeToAcc < gNotSafeToAccThreshold): 
        # keep distance from leading vehicle, keep current speed
        if(ego_keep_distance_saved == False):
            ego_keep_distance_saved = True
            ego_keep_distance_speed = velocity_kmh - gKeepDistanceSpeedSubtract # maintain a speed that is a little bit less than current speed
        
        pid_control = pidLongitudinalController.run_step(ego_keep_distance_speed)
        print("Keep distance " + str(pid_control))
        print("Saved speed " + str(ego_keep_distance_speed))
    else: 
        # it is not safe to accelerate, brake
        pid_control = pidLongitudinalController.run_step(velocity_kmh - gBrakeKMHStep)

        # check different thresholds of not safeToAcc and apply breaking accordingly 

        # if ego_keep_distance_saved is True there is no need to restore it to False
        # This is benefitial in a way that if the vehicle came to a complete stop
        # because leading vehicle is stopped, we want it to begin moving after
        # leading vehicle started moving again

        print("Brake " + str(pid_control * lSafeToAcc))
    
    print("Speed: " + str(velocity_kmh))

    # if vehicle velocity is less than road limit apply throttle or brake and steer
    if(velocity_kmh < ego_speed_limit):
        
        # control of steering for town06
        # at specific location appyl steering to stay in lane
        vehicle_loc_x = vehicle.get_location().x 

        steer_in = 0.0

        # if location x <= 140 and x > 139 add some steering to the left to keep in lane
        if((vehicle_loc_x <= 140 and vehicle_loc_x > 139)):
            steer_in = -0.085

        # apply control obtained trough PID Longitudinal Controller based on the outputs of the CNN
        if(pid_control >= 0.0):
            vehicle.apply_control(carla.VehicleControl(throttle = pid_control, steer = steer_in))
            print("Throttle")
        else:
            vehicle.apply_control(carla.VehicleControl(brake = abs(pid_control) * lSafeToAcc, steer = steer_in))
            print("Brake")
    else:
        vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0))

def ego_vehicle_manual_control(vehicle, keys):
    milliseconds = pygame.time.Clock().get_time()

    ctrl_throttle = 0.0 
    ctrl_brake = 0.0
    ctrl_steer = 0.0

    # get vehicle properties
    control = vehicle.get_control()
    throttle = control.throttle
    brake = control.brake
    steer = control.steer

    if keys[K_UP] or keys[K_w]:
        ctrl_throttle = min(throttle + 0.01, 1.00)

    if keys[K_DOWN] or keys[K_s]:
        ctrl_brake= min(brake + 0.2, 1)

    steer_increment = 5e-4 * 16

    if keys[K_LEFT] or keys[K_a]:
        if steer > 0:
            steer = 0
        else:
            steer -= steer_increment
    elif keys[K_RIGHT] or keys[K_d]:
        if steer < 0:
            steer = 0
        else:
            steer += steer_increment
    else:
        steer = 0.0

    ctrl_steer = min(0.7, max(-0.7, steer))
    
    vehicle.apply_control(carla.VehicleControl(throttle = ctrl_throttle, brake = ctrl_brake, steer = ctrl_steer))

def collect_data(sim_data, vehicle, camera):
    # save images from camera to disk
    if camera.get_save_to_disk() == False:
        camera.set_save_to_disk(True)

    # collect vehicle data
    control = vehicle.get_control()
    sim_data.add_data(control.throttle, control.brake, control.steer)

def save_data(sim_data, camera):
    if camera.get_save_to_disk() == True:
        camera.set_save_to_disk(False)

    # save images to disk
    camera.save_rgb_to_disk(datasetSavePath)

    # export sim_data with image names to .csv 
    sim_data.export_csv(datasetSavePath)

def cutImage(img, targetW, targetH):

    imgWidth, imgHeight = nativeImgWidth, nativeImgHeight

    widthCrop = imgWidth - targetW

    leftPoint = (imgWidth - targetImgWidth ) // 2
    rightPoint = leftPoint + targetW
    upperPoint = imgHeight - targetImgHeight
    lowerPoint = upperPoint + targetImgHeight

    #img = Image.fromarray(img).crop((leftPoint, upperPoint, rightPoint, lowerPoint))
    img = img[upperPoint : lowerPoint, leftPoint : rightPoint]
    #crop_img = img[y:y+h, x:x+w]
    return img

def cnn_processing(cnn_model, current_frame, smaPredictions):
    # cut first 350 pixels in all directions as not whole image is needed for processing
    # processed image size is (targetImgWidth x targetImageHeight) pixels
    current_frame = cutImage(current_frame, targetImgWidth, targetImgHeight)

    # preprocess data, scale, greyscale, etc.
    current_frame = cv2.resize(current_frame, dsize=(in_width, in_heigth), interpolation=cv2.INTER_CUBIC)
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # normalize image data in range [0, 1]    
    current_frame = current_frame / 255

    # reshape test dataset to appropriate dimensions of input layer of the trained CNN
    # reshape to (1, 100, 100, 1)
    # first dimension that is set to value=1 marks batch size
    cnn_img = current_frame.reshape(1, in_width, in_heigth, in_channels)

    # predict on reshaped frame
    cnn_predictions = cnn_model.predict(cnn_img, verbose = 0)

    # TO-DO: apply moving average on cnn predictions to minimize great oscillations in predicted values.
    smaPredictions.addToBuffer(cnn_predictions)

    return smaPredictions.getSMABuffer()

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    global gRecord_data
    global gData_collected

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        # Instanciating SimulationData in which we will save vehicle state if 'R' is pressed
        sim_data = SimulationData()

        # Instanciating the vehicle to which we attached the sensors
        bp = random.choice(world.get_blueprint_library().filter('vehicle.bmw.*'))
        # town06 location
        ego_spawn_loc = carla.Transform(ego_location_spawn, ego_transform_spawn)
        vehicle = world.spawn_actor(bp, ego_spawn_loc)
        
        # town0x random location
        #spawn_points = world.get_map().get_spawn_points()
        #random.shuffle(spawn_points)
        #ego_transform = spawn_points[0]
        #vehicle = world.spawn_actor(bp, ego_transform)
        
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(False)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[args.width, args.height])

        # Then, RGBCamera can be used to spawn RGB Camera as needed
        # and assign each to a grid position 
        cam = RGBCamera(world, display_manager, carla.Transform(carla.Location(x=0, z=1.7), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 0], saveResolution=7)

        # Spawn vehicles around ego vehicle
        #spawn_points = world.get_map().get_spawn_points()
        #spawn_vehicles_around_ego_vehicles(client=client, world=world, ego_vehicle=vehicle, radius=30, spawn_points=spawn_points, numbers_of_vehicles=10)

        if args.training == 'simulation':
            # create CNN model
            cnn_model = create_cnn_model(in_width, in_heigth, in_channels, output_no)
            # create CNN model
            #cnn_model = tf.keras.models.load_model(os.path.join(model_path, model_name))
            cnn_model.load_weights(os.path.join(model_path, model_name))

            # create PIDLongitudinalController for ego vehicle 'vehicle'
            pidLongitudinalController = PIDLongitudinalController(vehicle)

            smaPredictions = SimpleMovingAverage(gSMABufferLen, output_no)

        # array to store current RGBCamera frame to be sent to CNN
        current_frame = []

        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:

            # TO-DO:
            # add different scenarios, reset simulation, ego vehicle and other traffic participants

            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            # get latest frame from RGBCamera
            current_frame = np.asarray(cam.current_frame)

            # simulation or training mode
            # check if simulation loaded by checking dimension of current_frame
            if args.training == 'simulation' and current_frame.shape[0] != 0:
                # process current RGBCamera frame trough CNN
                cnn_predictions = cnn_processing(cnn_model, current_frame, smaPredictions)

                # control ego vehicle based on predictions
                ego_vehicle_control(vehicle, cnn_predictions, pidLongitudinalController)

                # DEBUG: print predictions
                cnn_predictions = np.round(cnn_predictions, decimals = 3)
                print(cnn_predictions)
            else:
                ego_vehicle_manual_control(vehicle, pygame.key.get_pressed())

            # record simulation data 
            if gRecord_data == True:
                collect_data(sim_data, vehicle, cam)
            else:
                if gData_collected == True:
                    save_data(sim_data, cam)
                    gData_collected = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break
                    elif event.key == K_r:
                        gRecord_data = not gRecord_data
                        gData_collected = True

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-t', '--training',
        metavar='t',
        dest='training',
        default='simulation',
        help='Training mode execution')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
