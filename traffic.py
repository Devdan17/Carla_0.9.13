#!/usr/bin/env python3
"""
Low-GPU ego vehicle demo in Town03:

- Ego vehicle with autopilot (Traffic Manager) following lanes & rules.
- Extra traffic vehicles.
- Extra pedestrians walking around and randomly crossing roads.
- Ego detects pedestrians directly ahead and reacts:
    - Brakes to a stop when a pedestrian is too close ahead.
    - Resumes autopilot when path is clear.
- Pygame visualization:
    - Full-window third-person camera behind the ego.
    - Small front camera inset at bottom-right.
"""

import carla
import random
import time
import numpy as np
import pygame
import math

# ---------------- CONFIG (tuned for your laptop) ----------------
DISPLAY_W, DISPLAY_H = 960, 540     # pygame window & third-person camera size
FRONT_W, FRONT_H = 240, 135        # small front camera inset

NUM_TRAFFIC = 14                   # number of extra traffic vehicles
NUM_WALKERS = 22                   # number of pedestrians

FPS = 10                           # lower FPS = less GPU load
DETECTION_RADIUS = 25.0            # meters, how far we "look" ahead
STOP_DISTANCE = 10.0               # if a pedestrian is this close ahead -> brake
CLEAR_DISTANCE = 15.0              # distance to consider path clear to resume
RETARGET_INTERVAL = 20.0           # seconds between assigning new random goals
# -----------------------------------------------------------------


def img_to_array(image):
    """Convert a CARLA image to HxWx3 uint8 numpy array (RGB)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3]


def is_in_front(ego_transform, target_location):
    """Return True if target is roughly in front of the ego."""
    yaw = math.radians(ego_transform.rotation.yaw)
    forward = np.array([math.cos(yaw), math.sin(yaw)])

    dx = target_location.x - ego_transform.location.x
    dy = target_location.y - ego_transform.location.y
    vec = np.array([dx, dy])

    dot = forward.dot(vec)
    return dot > 0  # positive dot product -> in front half-space


def lateral_offset(ego_transform, target_location):
    """
    Signed lateral offset of target from ego's forward direction (in meters).
    ~0 means roughly in same lane; positive/negative are left/right.
    """
    yaw = math.radians(ego_transform.rotation.yaw)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([math.sin(yaw), -math.cos(yaw)])  # 90° rotated

    dx = target_location.x - ego_transform.location.x
    dy = target_location.y - ego_transform.location.y
    vec = np.array([dx, dy])

    return right.dot(vec)


def main():
    pygame.init()
    display = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Ego Autonomy - Pedestrian Detection & Avoidance")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.load_world("Town03")

    # Use ASYNCHRONOUS mode (safer for weaker GPUs)
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(False)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.global_percentage_speed_difference(0.0)
    traffic_manager.set_random_device_seed(1)

    blueprint_library = world.get_blueprint_library()

    actors = []
    walkers = []
    walker_controllers = []

    front_frame = None
    third_frame = None

    manual_override = False    # True when we are manually braking
    last_brake_time = 0.0
    last_retarget_time = 0.0

    try:
        # ---------- Spawn Ego Vehicle ----------
        spawn_points = world.get_map().get_spawn_points()
        ego_spawn = random.choice(spawn_points)
        ego_bp = blueprint_library.filter("model3")[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        actors.append(ego)
        print("Ego vehicle ID:", ego.id)

        ego.set_autopilot(True, traffic_manager.get_port())

        # ---------- Third-person Camera (fills whole window) ----------
        third_bp = blueprint_library.find("sensor.camera.rgb")
        third_bp.set_attribute("image_size_x", str(DISPLAY_W))
        third_bp.set_attribute("image_size_y", str(DISPLAY_H))
        third_bp.set_attribute("fov", "90")

        third_transform = carla.Transform(
            carla.Location(x=-6.0, z=3.0),
            carla.Rotation(pitch=-15.0)
        )

        third_cam = world.spawn_actor(third_bp, third_transform, attach_to=ego)
        actors.append(third_cam)

        def third_callback(image):
            nonlocal third_frame
            third_frame = img_to_array(image)

        third_cam.listen(third_callback)

        # ---------- Front Camera (small inset) ----------
        front_bp = blueprint_library.find("sensor.camera.rgb")
        front_bp.set_attribute("image_size_x", str(FRONT_W))
        front_bp.set_attribute("image_size_y", str(FRONT_H))
        front_bp.set_attribute("fov", "90")

        front_transform = carla.Transform(carla.Location(x=1.5, z=2.2))

        front_cam = world.spawn_actor(front_bp, front_transform, attach_to=ego)
        actors.append(front_cam)

        def front_callback(image):
            nonlocal front_frame
            front_frame = img_to_array(image)

        front_cam.listen(front_callback)

        # ---------- Traffic Vehicles ----------
        vehicle_bps = blueprint_library.filter("vehicle.*")
        traffic_vehicles = []
        random.shuffle(spawn_points)
        for sp in spawn_points:
            if len(traffic_vehicles) >= NUM_TRAFFIC:
                break
            bp = random.choice(vehicle_bps)
            v = world.try_spawn_actor(bp, sp)
            if v and v.id != ego.id:
                v.set_autopilot(True, traffic_manager.get_port())
                actors.append(v)
                traffic_vehicles.append(v)

        print("Traffic vehicles spawned:", len(traffic_vehicles))

        # ---------- Pedestrians + Controllers ----------
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        spawn_transforms = []
        for _ in range(NUM_WALKERS * 2):  # try extra locations
            loc = world.get_random_location_from_navigation()
            if loc:
                spawn_transforms.append(carla.Transform(loc))

        for tr in spawn_transforms:
            if len(walkers) >= NUM_WALKERS:
                break
            wb = random.choice(walker_bps)
            w = world.try_spawn_actor(wb, tr)
            if w:
                walkers.append(w)
                actors.append(w)

        print("Pedestrians spawned:", len(walkers))

        for w in walkers:
            c = world.spawn_actor(controller_bp, carla.Transform(), w)
            walker_controllers.append(c)
            actors.append(c)

        time.sleep(1.0)

        # Start walkers with random goals
        for c in walker_controllers:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.2 + random.random())  # 1.2–2.2 m/s

        print("Walker controllers started:", len(walker_controllers))

        # ---------- Main Loop ----------
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()

            # Periodically retarget pedestrians to new random locations
            if now - last_retarget_time > RETARGET_INTERVAL:
                for c in walker_controllers:
                    loc = world.get_random_location_from_navigation()
                    if loc:
                        # random wandering -> often crosses roads naturally
                        c.go_to_location(loc)
                last_retarget_time = now

            # --- Pedestrian detection in front of ego ---
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            actors_all = world.get_actors()
            walkers_all = actors_all.filter("walker.pedestrian.*")

            closest_ped = None
            closest_dist = 1e9

            for w in walkers_all:
                loc = w.get_location()
                dist = loc.distance(ego_loc)
                if dist > DETECTION_RADIUS:
                    continue
                if not is_in_front(ego_tf, loc):
                    continue
                lat = abs(lateral_offset(ego_tf, loc))
                if lat > 3.0:  # too far to the side (not in our lane)
                    continue
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ped = w

            if closest_ped and closest_dist < STOP_DISTANCE:
                # Too close: override autopilot and brake hard
                if not manual_override:
                    print(f"[BRAKE] Pedestrian ahead at {closest_dist:.1f} m")
                    ego.set_autopilot(False)
                    manual_override = True
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                last_brake_time = now
            else:
                # No close pedestrian ahead
                if manual_override and (now - last_brake_time) > 2.0 and closest_dist > CLEAR_DISTANCE:
                    # Path has been clear for a bit: resume autopilot
                    print("[RESUME] Path clear, resuming autopilot")
                    ego.set_autopilot(True, traffic_manager.get_port())
                    manual_override = False

            # ---------- Draw third-person camera full window ----------
            if third_frame is not None:
                surf = pygame.surfarray.make_surface(np.rot90(third_frame))
                display.blit(surf, (0, 0))

            # ---------- Draw front camera inset ----------
            if front_frame is not None:
                surf_f = pygame.surfarray.make_surface(np.rot90(front_frame))
                inset_x = DISPLAY_W - FRONT_W - 10
                inset_y = DISPLAY_H - FRONT_H - 10
                display.blit(surf_f, (inset_x, inset_y))

            # ---------- Text overlay ----------
            lines = [
                f"Ego ID: {ego.id}",
                f"Pedestrian ahead: {'YES' if closest_ped else 'NO'}",
                f"Closest ped distance: {closest_dist:.1f} m" if closest_ped else "",
                f"Mode: {'BRAKING (override)' if manual_override else 'AUTOPILOT'}",
            ]
            y = 10
            for line in lines:
                if not line:
                    continue
                txt = font.render(line, True, (255, 255, 255))
                display.blit(txt, (10, y))
                y += 20

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        print("Cleaning up actors...")
        for c in walker_controllers:
            try:
                c.stop()
            except RuntimeError:
                pass

        client.apply_batch([carla.command.DestroyActor(a) for a in actors])

        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
