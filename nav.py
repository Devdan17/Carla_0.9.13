#!/usr/bin/env python3
"""
Town03 ego-vehicle demo (low GPU) with:
- Route navigation (BasicAgent) from random start to far destination.
- BasicAgent handles steering (no custom steering overrides).
- Lane changes disabled for ego -> single-lane lane-following.
- NPC traffic via Traffic Manager.
- Pedestrians walking near roads and randomly crossing ahead of ego.
- Ego detects pedestrians in front and brakes, then resumes the route.
- Pygame visualization:
    * Full-window third-person camera.
    * Small front camera inset (bottom-right).
    * Small minimap (top-left) with road graph, route, ego, and destination.
"""

import carla
import random
import math
import time
import os
import sys

import numpy as np
import pygame

# ---------- EDIT THIS IF YOUR PATH IS DIFFERENT ----------
CARLA_ROOT = r"C:\CARLA_0.9.13\WindowsNoEditor\PythonAPI"
sys.path.append(CARLA_ROOT)
sys.path.append(os.path.join(CARLA_ROOT, "agents"))

from agents.navigation.basic_agent import BasicAgent  # noqa: E402

# ---------------- CONFIG (tuned for your laptop) ----------------------
DISPLAY_W, DISPLAY_H = 960, 540     # pygame window & third-person camera size
FRONT_W, FRONT_H = 240, 135        # small front camera inset

NUM_TRAFFIC = 14                   # number of extra traffic vehicles
NUM_WALKERS = 22                   # number of pedestrians

FPS = 10                           # lower FPS = less GPU load
DETECTION_RADIUS = 25.0            # meters, how far we "look" ahead
STOP_DISTANCE = 10.0               # if a pedestrian is this close ahead -> brake
CLEAR_DISTANCE = 15.0              # distance to consider path clear to resume
RETARGET_INTERVAL = 20.0           # seconds between assigning new random goals

MINIMAP_W, MINIMAP_H = 220, 220    # small map size (top-left)
MINIMAP_MARGIN = 10
# ---------------------------------------------------------------------


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


def distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# ---------------- MINIMAP HELPERS ----------------

def build_minimap(world_map):
    """
    Build static minimap data:
    - road segments from topology
    - bounds (min/max x, y) for coordinate mapping
    Returns (segments, bounds) where:
    segments = [((x1,y1), (x2,y2)), ...]
    bounds = (min_x, max_x, min_y, max_y)
    """
    topology = world_map.get_topology()
    segments = []

    xs, ys = [], []

    for wp_start, wp_end in topology:
        loc1 = wp_start.transform.location
        loc2 = wp_end.transform.location
        x1, y1 = loc1.x, loc1.y
        x2, y2 = loc2.x, loc2.y
        segments.append(((x1, y1), (x2, y2)))
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if not xs:
        bounds = (-200.0, 200.0, -200.0, 200.0)
    else:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = 0.05 * (max_x - min_x)
        pad_y = 0.05 * (max_y - min_y)
        bounds = (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)

    return segments, bounds


def world_to_minimap(x, y, bounds):
    """
    Map world (x,y) into [0, MINIMAP_W] x [0, MINIMAP_H].
    Keeps orientation roughly consistent (north-up).
    """
    min_x, max_x, min_y, max_y = bounds
    if max_x - min_x < 1e-3 or max_y - min_y < 1e-3:
        return 0, 0

    nx = (x - min_x) / (max_x - min_x)
    ny = (y - min_y) / (max_y - min_y)

    px = int(nx * MINIMAP_W)
    py = MINIMAP_H - int(ny * MINIMAP_H)
    return px, py


def prerender_minimap_bg(segments, bounds):
    """
    Render a static minimap background surface with road segments.
    """
    surf = pygame.Surface((MINIMAP_W, MINIMAP_H))
    surf.fill((10, 10, 10))  # dark background

    road_color = (80, 80, 80)

    for (x1, y1), (x2, y2) in segments:
        px1, py1 = world_to_minimap(x1, y1, bounds)
        px2, py2 = world_to_minimap(x2, y2, bounds)
        pygame.draw.line(surf, road_color, (px1, py1), (px2, py2), 1)

    return surf


def extract_route_points(agent):
    """
    Extract list of locations along the global route planned by BasicAgent.
    """
    pts = []
    if hasattr(agent, "_global_plan") and agent._global_plan:
        for wp, _ in agent._global_plan:
            loc = wp.transform.location
            pts.append(loc)
    return pts


# ---------------- MAIN ----------------

def main():
    pygame.init()
    display = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Town03 Route + Ped Avoid + Minimap (Low GPU)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.load_world("Town03")
    world_map = world.get_map()

    # Keep ASYNCHRONOUS mode (lighter on GPU) – let CARLA tick itself.
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

    manual_brake = False
    last_brake_time = 0.0
    last_retarget_time = 0.0

    # Minimap data
    minimap_segments, minimap_bounds = build_minimap(world_map)
    minimap_bg = prerender_minimap_bg(minimap_segments, minimap_bounds)
    route_points = []
    goal_location = None

    try:
        # ---------- Choose start & goal spawn points (far apart) ----------
        spawn_points = world_map.get_spawn_points()
        if len(spawn_points) < 2:
            print("Not enough spawn points in this map.")
            return

        start = random.choice(spawn_points)
        goal = random.choice(spawn_points)

        attempts = 0
        while distance(start.location, goal.location) < 150.0 and attempts < 20:
            goal = random.choice(spawn_points)
            attempts += 1

        goal_location = goal.location

        print("Start location:", start.location)
        print("Destination   :", goal.location)

        # ---------- Spawn Ego Vehicle ----------
        ego_bp = blueprint_library.filter("model3")[0]
        ego = world.try_spawn_actor(ego_bp, start)
        if ego is None:
            print("Failed to spawn ego vehicle. Try again.")
            return

        actors.append(ego)
        print("Ego vehicle ID:", ego.id)

        ego.set_autopilot(False)  # BasicAgent will control

        # ---------- Create BasicAgent for route navigation ----------
        agent = BasicAgent(ego, target_speed=35)  # km/h
        agent.set_destination(goal.location)
        print("BasicAgent destination set. Route planned.")

        # Just turn OFF lane changing; keep all other tuning default.
        try:
            lp = agent._local_planner
            if hasattr(lp, "_lane_change_allowed"):
                lp._lane_change_allowed = False
            if hasattr(lp, "_offset"):
                lp._offset = 0.0
        except Exception:
            pass

        # Densify global route slightly
        try:
            if hasattr(agent, "_global_route_planner"):
                agent._global_route_planner._sampling_resolution = 1.0
        except Exception:
            pass

        route_points = extract_route_points(agent)

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

        # ---------- Traffic Vehicles (TM autopilot) ----------
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

        # ---------- Pedestrians + Controllers (spawn near roads) ----------
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        side_wps = world_map.generate_waypoints(2.0)
        random.shuffle(side_wps)

        for wp in side_wps:
            if len(walkers) >= NUM_WALKERS:
                break
            if wp.lane_type != carla.LaneType.Sidewalk:
                continue

            spawn_loc = wp.transform.location
            spawn_loc.z += 0.5

            wb = random.choice(walker_bps)
            w = world.try_spawn_actor(wb, carla.Transform(spawn_loc))
            if w:
                walkers.append(w)
                actors.append(w)

        print("Pedestrians spawned near roads:", len(walkers))

        for w in walkers:
            c = world.spawn_actor(controller_bp, carla.Transform(), w)
            walker_controllers.append(c)
            actors.append(c)

        time.sleep(1.0)

        for c in walker_controllers:
            loc = world.get_random_location_from_navigation()
            if loc:
                c.start()
                c.go_to_location(loc)
                c.set_max_speed(1.2 + random.random())

        print("Walker controllers started:", len(walker_controllers))

        # ---------- Main Loop ----------
        running = True
        reached_destination = False
        idle_after_reach = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()

            # Periodically retarget pedestrians
            if now - last_retarget_time > RETARGET_INTERVAL:
                for c in walker_controllers:
                    loc = world.get_random_location_from_navigation()
                    if loc:
                        c.go_to_location(loc)
                last_retarget_time = now

            # Force some walkers to cross ahead of ego
            for c in walker_controllers[:8]:
                if random.random() < 0.03:
                    dist_ahead = random.uniform(10.0, 20.0)
                    ahead_loc = carla.Location(
                        x=ego_loc.x + ego_forward.x * dist_ahead,
                        y=ego_loc.y + ego_forward.y * dist_ahead,
                        z=ego_loc.z
                    )
                    cross_target = carla.Location(
                        x=ahead_loc.x + random.uniform(-4.0, 4.0),
                        y=ahead_loc.y + random.uniform(-4.0, 4.0),
                        z=ahead_loc.z
                    )
                    c.go_to_location(cross_target)

            # Pedestrian detection
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
                if lat > 3.0:
                    continue
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ped = w

            # Decide control
            if closest_ped and closest_dist < STOP_DISTANCE:
                if not manual_brake:
                    print(f"[BRAKE] Pedestrian ahead at {closest_dist:.1f} m")
                    manual_brake = True
                control = carla.VehicleControl(throttle=0.0, brake=1.0)
                last_brake_time = now
            else:
                if manual_brake and (now - last_brake_time) > 2.0 and closest_dist > CLEAR_DISTANCE:
                    print("[RESUME] Path clear, resuming route control")
                    manual_brake = False

                if not reached_destination:
                    control = agent.run_step()
                    if agent.done():
                        print("Destination reached. Holding position...")
                        reached_destination = True
                        idle_after_reach = 0
                else:
                    control = carla.VehicleControl(throttle=0.0, brake=0.7)
                    idle_after_reach += 1
                    if idle_after_reach > 50:
                        print("Route navigation complete. Exiting loop.")
                        running = False

            ego.apply_control(control)

            # ---------- Draw third-person ----------
            if third_frame is not None:
                surf = pygame.surfarray.make_surface(np.rot90(third_frame))
                display.blit(surf, (0, 0))

            # ---------- Draw front camera ----------
            if front_frame is not None:
                surf_f = pygame.surfarray.make_surface(np.rot90(front_frame))
                inset_x = DISPLAY_W - FRONT_W - 10
                inset_y = DISPLAY_H - FRONT_H - 10
                display.blit(surf_f, (inset_x, inset_y))

            # ---------- Minimap ----------
            minimap = minimap_bg.copy()

            if route_points:
                prev_px, prev_py = None, None
                for loc in route_points[::5]:
                    px, py = world_to_minimap(loc.x, loc.y, minimap_bounds)
                    if prev_px is not None:
                        pygame.draw.line(minimap, (0, 120, 255),
                                         (prev_px, prev_py), (px, py), 2)
                    prev_px, prev_py = px, py

            if goal_location is not None:
                gx, gy = world_to_minimap(goal_location.x, goal_location.y, minimap_bounds)
                pygame.draw.circle(minimap, (0, 255, 0), (gx, gy), 4)

            ex, ey = world_to_minimap(ego_loc.x, ego_loc.y, minimap_bounds)
            pygame.draw.circle(minimap, (255, 0, 0), (ex, ey), 4)

            display.blit(minimap, (MINIMAP_MARGIN, MINIMAP_MARGIN))

            # ---------- HUD ----------
            lines = [
                f"Ego ID: {ego.id}",
                f"Pedestrian ahead: {'YES' if closest_ped else 'NO'}",
                f"Closest ped distance: {closest_dist:.1f} m" if closest_ped else "",
                f"Mode: {'BRAKING (override)' if manual_brake else 'ROUTE FOLLOWING'}",
                f"Reached destination: {'YES' if reached_destination else 'NO'}",
            ]
            y = 10
            for line in lines:
                if not line:
                    continue
                txt = font.render(line, True, (255, 255, 255))
                display.blit(txt, (DISPLAY_W - 10 - txt.get_width(), y))
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
