import carla
import pygame
import numpy as np

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Get world and load Town03
world = client.load_world("Town03")

blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = blueprints.filter("model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

print("Vehicle spawned:", vehicle.id)

# ----- CAMERA SETUP -----
camera_bp = blueprints.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)  # front hood position
)

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Global frame buffer
frame = None

def camera_callback(image):
    global frame
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((600, 800, 4))[:, :, :3]
    frame = array

camera.listen(camera_callback)

# ----- PYGAME WINDOW -----
pygame.init()
display = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# ----- MAIN LOOP -----
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if frame is not None:
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        display.blit(surface, (0, 0))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
camera.stop()
vehicle.destroy()
print("Camera and vehicle destroyed.")
