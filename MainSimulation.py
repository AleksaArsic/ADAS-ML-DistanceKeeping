"""
Script that render multiple rgb cameras in the same pygame window

By default, it renders four cameras, left mirror, right mirror,
center and back
"""

import glob
import os
import sys

from matplotlib.transforms import Transform

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

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

""" Import all essential custom classes 
"""
from scripts.CustomTimer import CustomTimer
from scripts.DisplayManager import DisplayManager
from scripts.RGBCamera import RGBCamera

#################################################################################################################
# constants
ms_to_kmh_ratio = 3600/1000 # m/s to km/h ratio
#################################################################################################################

#################################################################################################################
# ego vehicle parameters
ego_spawn_point = 2 # ego vehicle initial spawn point
ego_location_spawn = carla.Location(x=586.856873, y=-17.063015, z=0.300000) #ego vehicle initial spawn location
ego_transform_spawn = carla.Rotation(pitch=0.000000, yaw=-180.035, roll=0.000000) # ego vehicle initial spawn rotation
ego_speed_limit = 80 # ego vehicle speed limit in km/h
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

def ego_vehicle_control(vehicle):
    abs_vector = math.sqrt(vehicle.get_velocity().x ** 2 + vehicle.get_velocity().y ** 2 + vehicle.get_velocity().z ** 2)
    velocity_kmh = abs(abs_vector * ms_to_kmh_ratio)

    vehicle_loc_x = vehicle.get_location().x 

    if(velocity_kmh < ego_speed_limit):
        steer_in = 0.0

        # if location x <= 140 add some steering to the left to keep in lane
        if((vehicle_loc_x <= 140 and vehicle_loc_x > 139)):
            steer_in = -0.085

        vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = steer_in))
    else:
        vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0))

    

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

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

        # print all spawn points 
        #spawn_points = world.get_map().get_spawn_points()

        #for i in range(len(spawn_points)):
        #    print("Index: " + str(i) + " " +str(spawn_points[i]))

        # Instanciating the vehicle to which we attached the sensors
        bp = random.choice(world.get_blueprint_library().filter('vehicle.bmw.*'))
        ego_spawn_loc = carla.Transform(ego_location_spawn, ego_transform_spawn)
        vehicle = world.spawn_actor(bp, ego_spawn_loc)
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(False)


        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[args.width, args.height])

        # Then, RGBCamera can be used to spawn RGB Camera as needed
        # and assign each to a grid position 
        cam = RGBCamera(world, display_manager, carla.Transform(carla.Location(x=0, z=1.7), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 0])

        # Spawn vehicles around ego vehicle
        #spawn_points = world.get_map().get_spawn_points()
        #spawn_vehicles_around_ego_vehicles(client=client, world=world, ego_vehicle=vehicle, radius=30, spawn_points=spawn_points, numbers_of_vehicles=10)


        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:

            # ego vehicle control
            ego_vehicle_control(vehicle)

            print(vehicle.get_location())

            # TO-DO:
            # add different scenarios, reset simulation, ego vehicle and other traffic participants

            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

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
