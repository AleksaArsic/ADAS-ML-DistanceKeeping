
'''
    HUD.py - class responsible for giving information status about the vehicle in the pygame window of the main simulation.
'''
import pygame
import carla
import datetime
import math
import os
import numpy as np

class HUD(object):
    def __init__(self, width, height):
        pygame.font.init()
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._client_clock = pygame.time.Clock()

        # format:
        # [vehicle, cnn_predictions, sma_predictions, ego_pid_control[accelerate_rate, brake_rate]]
        self.sim_data = []

    def setSimData(self, sim_data):
        self.sim_data = sim_data

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world):
        self._client_clock.tick()
        self._notifications.tick(world, self._client_clock)
        if not self._show_info:
            return
        
        t = self.sim_data[0].get_transform()
        v = self.sim_data[0].get_velocity()
        c = self.sim_data[0].get_control()

        steer_rate = 0.0

        if len(self.sim_data[3]):
            cnn_predictions = self.sim_data[1]
            sma_predictions = self.sim_data[2]

            safe_to_accelerate = sma_predictions[-1]
            accelerate_rate = self.sim_data[3][0]
            brake_rate = self.sim_data[3][1]
            keep_speed_rate = self.sim_data[3][2]

            keep_distance = 1 if sma_predictions[-1] >= 0.4 and sma_predictions[-1] <= 0.6 else 0
        else:
            cnn_predictions = [0, 0, 0, 0, 0]
            sma_predictions = [0, 0, 0, 0, 0]

            safe_to_accelerate = 0.0
            accelerate_rate = 0.0
            brake_rate = 0.0
            keep_speed_rate = 0.0

            keep_distance = 0

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % self._client_clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(self.sim_data[0], truncate=20),
            'Map:     % 20s' % world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y))
            ]
    
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
                '']

        self._info_text += [
            'CNN predictions: ' + str(np.round(cnn_predictions, decimals = 3)),
            'SMA predictions: ' + str(np.round(sma_predictions, decimals = 3)),
            'Throttle:           % 16.1f' % accelerate_rate,
            'Break:              % 16.1f' % brake_rate,
            'Steer:              % 16.1f' % steer_rate,
            'Safe to accelerate: % 16.3f' % safe_to_accelerate,
            ''
        ]

        self._info_text += [
            'Vehicles present:            ' + str(1 - round(sma_predictions[0])),
            'Vehicles present left:       ' + str(1 - round(sma_predictions[1])),
            'Vehicles present right:      ' + str(1 - round(sma_predictions[2])),
            'Vehicles present center:     ' + str(1 - round(sma_predictions[3])),
            'Keep distance:             %s' % {0: 'False', 1: 'True'}.get(keep_distance, keep_distance),
            'Saved speed:                ' + str(round(keep_speed_rate)),
            '',
        ]

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((self.dim[0], 115))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            h_offset = 105
            bar_h_offset = 435
            bar_width = 106

            v_cnt = 0
            h_cnt = 0
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    #v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 4), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 4), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset+ f * (bar_width - 6), v_offset + 4), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 4), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (h_offset, v_offset))
                
                v_cnt += 1

                if v_cnt % 7 == 0:
                    if h_cnt == 2:
                        h_offset += 380
                        h_cnt = 0
                    else:
                        h_offset += 240

                    v_offset = 4
                    v_cnt = 0
                    h_cnt += 1
                else:
                    v_offset += 18

        self._notifications.render(display)

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name