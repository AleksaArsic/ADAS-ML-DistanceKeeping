import carla
import numpy as np
from scripts.CustomTimer import CustomTimer

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class RGBCamera:
    def __init__(self, world, display_man, transform, attached, sensor_options, display_pos, save_to_disk = False):
        self.save_to_disk = save_to_disk
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, transform, attached, sensor_options = None):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        disp_size = self.display_man.get_display_size()
        camera_bp.set_attribute('image_size_x', str(disp_size[0]))
        camera_bp.set_attribute('image_size_y', str(disp_size[1]))

        if sensor_options != None:
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.save_rgb_image)

        return camera

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)

        if(self.save_to_disk == True):
            image.save_to_disk('camera_sensors_output/left_mirror/%.6d.jpg' % image.frame)

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()