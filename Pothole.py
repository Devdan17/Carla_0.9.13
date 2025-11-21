#!/usr/bin/env python3
"""
Town03 ego-vehicle demo (CARLA 0.9.13) with:

- Route navigation using BasicAgent (start → far destination)
- Global route computed via GlobalRoutePlanner (0.9.13 version, no DAO)
- Potholes placed ON the BasicAgent route (driving lane center)
- Stable pothole avoidance (no spinning)
- Curve slowdown (ego reduces speed in sharp turns)
- Traffic vehicles on autopilot
- Pedestrians crossing in front of ego (detected + brake)
- Pygame visualization:
    * Third-person camera (full window)
    * Front camera inset (bottom-right)
    * Minimap (top-left)
"""

import carla
import random
import math
import time
import os
import sys

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# PATH FIX FOR CARLA 0.9.13
# ---------------------------------------------------------------------------
CARLA_ROOT = r"C:\CARLA_0.9.13\WindowsNoEditor\PythonAPI"
sys.path.append(CARLA_ROOT)
sys.path.append(os.path.join(CARLA_ROOT, "agents"))

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner


# ---------------- CONFIG ----------------
DISPLAY_W, DISPLAY_H = 960, 540
FRONT_W, FRONT_H = 240, 135
FPS = 10

NUM_TRAFFIC = 12
NUM_WALKERS = 20

DETECTION_RADIUS = 25.0
STOP_DISTANCE = 10.0
CLEAR_DISTANCE = 14.0

RETARGET_INTERVAL = 20.0

# ===== FINAL STABLE POTHOLE SETTINGS =====
POTHOLE_DETECT_DIST = 22.0       # detection distance ahead
POTHOLE_LANE_WIDTH = 3.0
POTHOLE_AVOID_OFFSET = 2.5       # lateral sidestep
POTHOLE_STEER_GAIN = 0.9         # steering gain for avoidance
POTHOLE_END_DIST = 7.0           # distance after which we end avoidance
NUM_POTHOLES = 6

# Minimap
MINIMAP_W, MINIMAP_H = 220, 220
MINIMAP_MARGIN = 10


# ---------------- HELPERS ----------------
def img_to_array(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3]


def is_in_front(ego_tf, target_loc):
    yaw = math.radians(ego_tf.rotation.yaw)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    dx = target_loc.x - ego_tf.location.x
    dy = target_loc.y - ego_tf.location.y
    return fwd.dot(np.array([dx, dy])) > 0.0


def lateral_offset(ego_tf, target_loc):
    yaw = math.radians(ego_tf.rotation.yaw)
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    dx = target_loc.x - ego_tf.location.x
    dy = target_loc.y - ego_tf.location.y
    return right.dot(np.array([dx, dy]))


def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


# ---------------- MINIMAP ----------------
def build_minimap(world_map):
    topo = world_map.get_topology()
    segs = []
    xs, ys = [], []
    for wp1, wp2 in topo:
        l1 = wp1.transform.location
        l2 = wp2.transform.location
        segs.append(((l1.x, l1.y), (l2.x, l2.y)))
        xs += [l1.x, l2.x]
        ys += [l1.y, l2.y]

    if not xs:
        return [], (-200, 200, -200, 200)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = 0.05 * (max_x - min_x)
    pad_y = 0.05 * (max_y - min_y)
    bounds = (min_x - pad_x, max_x + pad_x,
              min_y - pad_y, max_y + pad_y)

    return segs, bounds


def world_to_minimap(x, y, bounds):
    min_x, max_x, min_y, max_y = bounds
    nx = (x - min_x) / max(1e-6, (max_x - min_x))
    ny = (y - min_y) / max(1e-6, (max_y - min_y))
    px = int(nx * MINIMAP_W)
    py = MINIMAP_H - int(ny * MINIMAP_H)
    return px, py


def prerender_minimap_bg(segments, bounds):
    surf = pygame.Surface((MINIMAP_W, MINIMAP_H))
    surf.fill((10, 10, 10))
    for (x1, y1), (x2, y2) in segments:
        p1 = world_to_minimap(x1, y1, bounds)
        p2 = world_to_minimap(x2, y2, bounds)
        pygame.draw.line(surf, (80, 80, 80), p1, p2, 1)
    return surf


# ---------------- POTHOLE (0.9.13-safe) ----------------
def draw_pothole(world, loc):
    radius = 1.2
    segments = 18
    angle_step = 2 * math.pi / segments

    pts = []
    for i in range(segments):
        a = i * angle_step
        pts.append(carla.Location(
            loc.x + radius * math.cos(a),
            loc.y + radius * math.sin(a),
            loc.z
        ))

    for i in range(segments):
        s = pts[i]
        e = pts[(i + 1) % segments]
        world.debug.draw_line(
            s, e, thickness=0.10, life_time=0.10,
            color=carla.Color(30, 30, 30)
        )

    for _ in range(6):
        ang = random.uniform(0, 2*math.pi)
        ln = random.uniform(0.6, 1.5)
        sx = loc.x + 0.3 * math.cos(ang)
        sy = loc.y + 0.3 * math.sin(ang)
        ex = loc.x + ln * math.cos(ang)
        ey = loc.y + ln * math.sin(ang)
        world.debug.draw_line(
            carla.Location(sx, sy, loc.z),
            carla.Location(ex, ey, loc.z),
            thickness=0.06,
            life_time=0.10,
            color=carla.Color(0, 0, 0)
        )


def choose_potholes_from_route(route_points, num):
    if not route_points:
        return []
    step = max(1, len(route_points)//num)
    return [route_points[i] for i in range(0, len(route_points), step)][:num]


# ---------------- MAIN ----------------
def main():

    pygame.init()
    display = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Town03 Route + Ped + Potholes (0.9.13 FINAL)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.load_world("Town03")
    world_map = world.get_map()
    bp = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    actors = []
    walkers = []
    walker_controllers = []

    front_frame = None
    third_frame = None

    manual_brake = False
    last_brake_time = 0
    last_retarget_time = 0

    potholes = []
    active_pothole = None
    avoid_side = 1

    # ---------------- Start/Goal ----------------
    sp = world_map.get_spawn_points()
    start = random.choice(sp)
    goal = random.choice(sp)
    while distance(start.location, goal.location) < 150:
        goal = random.choice(sp)

    print("Start:", start.location)
    print("Goal :", goal.location)

    # ---------------- Route ----------------
    grp = GlobalRoutePlanner(world_map, 2.0)
    route = grp.trace_route(start.location, goal.location)
    route_points = [wp.transform.location for wp, _ in route]
    print("Route points:", len(route_points))

    # ---------------- Ego ----------------
    ego_bp = bp.filter("vehicle.tesla.model3")[0]
    ego = world.try_spawn_actor(ego_bp, start)
    actors.append(ego)

    agent = BasicAgent(ego, target_speed=35)
    agent.set_destination(goal.location)

    try:
        lp = agent._local_planner
        if hasattr(lp, "_lane_change_allowed"):
            lp._lane_change_allowed = False
        if hasattr(lp, "_offset"):
            lp._offset = 0.0
    except Exception:
        pass

    # ---------------- Potholes ----------------
    raw = choose_potholes_from_route(route_points, NUM_POTHOLES)
    for loc in raw:
        wp = world_map.get_waypoint(loc, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp:
            p = wp.transform.location
            p.z += 0.27
            potholes.append(p)

    print("Potholes placed:", len(potholes))

    # ---------------- Cameras ----------------
    third_bp = bp.find("sensor.camera.rgb")
    third_bp.set_attribute("image_size_x", str(DISPLAY_W))
    third_bp.set_attribute("image_size_y", str(DISPLAY_H))
    third = world.spawn_actor(
        third_bp,
        carla.Transform(carla.Location(x=-7, z=3), carla.Rotation(pitch=-15)),
        attach_to=ego)
    actors.append(third)

    def third_cb(image):
        nonlocal third_frame
        third_frame = img_to_array(image)

    third.listen(third_cb)

    front_bp = bp.find("sensor.camera.rgb")
    front_bp.set_attribute("image_size_x", str(FRONT_W))
    front_bp.set_attribute("image_size_y", str(FRONT_H))
    front = world.spawn_actor(
        front_bp,
        carla.Transform(carla.Location(x=1.5, z=2.1)),
        attach_to=ego)
    actors.append(front)

    def front_cb(image):
        nonlocal front_frame
        front_frame = img_to_array(image)

    front.listen(front_cb)

    # ---------------- Traffic ----------------
    veh_bps = bp.filter("vehicle.*")
    traffic = []
    random.shuffle(sp)
    for s in sp:
        if len(traffic) >= NUM_TRAFFIC:
            break
        v = world.try_spawn_actor(random.choice(veh_bps), s)
        if v:
            actors.append(v)
            v.set_autopilot(True)
            traffic.append(v)
    print("Traffic:", len(traffic))

    # ---------------- Pedestrians ----------------
    walker_bps = bp.filter("walker.pedestrian.*")
    ctrl_bp = bp.find("controller.ai.walker")

    spawn_trans = []
    tries = 0
    while len(spawn_trans) < NUM_WALKERS and tries < 200:
        loc = world.get_random_location_from_navigation()
        tries += 1
        if loc:
            loc.z += 0.5
            spawn_trans.append(carla.Transform(loc))

    for tr in spawn_trans:
        w = world.try_spawn_actor(random.choice(walker_bps), tr)
        if w:
            actors.append(w)
            walkers.append(w)

    print("Walkers:", len(walkers))

    for w in walkers:
        c = world.spawn_actor(ctrl_bp, carla.Transform(), w)
        actors.append(c)
        walker_controllers.append(c)

    time.sleep(1.0)
    for c in walker_controllers:
        dest = world.get_random_location_from_navigation()
        if dest:
            c.start()
            c.go_to_location(dest)
            c.set_max_speed(1.0 + random.random())

    # ---------------- Minimap ----------------
    segs, bounds = build_minimap(world_map)
    minimap_bg = prerender_minimap_bg(segs, bounds)

    # ---------------- MAIN LOOP ----------------
    running = True
    reached_destination = False
    idle_after_reach = 0

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        now = time.time()

        # ---------------- Retarget walkers ----------------
        if now - last_retarget_time > RETARGET_INTERVAL:
            last_retarget_time = now
            for c in walker_controllers:
                dest = world.get_random_location_from_navigation()
                if dest:
                    c.go_to_location(dest)

        # ---------------- Random crossing walkers ----------------
        for c in walker_controllers[:6]:
            if random.random() < 0.03:
                fwd_vec = ego_tf.get_forward_vector()
                dist_ahead = random.uniform(10, 20)
                ahead = carla.Location(
                    ego_loc.x + fwd_vec.x * dist_ahead,
                    ego_loc.y + fwd_vec.y * dist_ahead,
                    ego_loc.z
                )
                cross = carla.Location(
                    ahead.x + random.uniform(-4, 4),
                    ahead.y + random.uniform(-4, 4),
                    ahead.z
                )
                c.go_to_location(cross)

        # ---------------- Pedestrian detection ----------------
        walkers_all = world.get_actors().filter("walker.pedestrian.*")
        closest_ped = None
        closest_dist = 999

        for w in walkers_all:
            loc = w.get_location()
            dist = loc.distance(ego_loc)
            if dist > DETECTION_RADIUS:
                continue
            if not is_in_front(ego_tf, loc):
                continue
            if abs(lateral_offset(ego_tf, loc)) > 3:
                continue
            if dist < closest_dist:
                closest_dist = dist
                closest_ped = w

        # ---------------- Pothole detection ----------------
        nearest_poth = None
        nearest_dist = 999

        for ph in potholes:
            dist = ph.distance(ego_loc)
            if dist > POTHOLE_DETECT_DIST:
                continue
            if not is_in_front(ego_tf, ph):
                continue
            if abs(lateral_offset(ego_tf, ph)) > POTHOLE_LANE_WIDTH:
                continue
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_poth = ph

        # Start avoidance (dynamic side choice)
        if active_pothole is None and nearest_poth is not None:
            active_pothole = nearest_poth

            # Pick right/left based on lane topology
            wp_ph = world_map.get_waypoint(
                active_pothole,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            avoid_side = 1  # default right
            if wp_ph is not None:
                right_wp = wp_ph.get_right_lane()
                left_wp = wp_ph.get_left_lane()

                if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                    avoid_side = 1   # right
                elif left_wp and left_wp.lane_type == carla.LaneType.Driving:
                    avoid_side = -1  # left
                else:
                    avoid_side = 1   # fallback

            print("[POTHOLE] Avoid:", active_pothole,
                  "side:", "RIGHT" if avoid_side == 1 else "LEFT")

        # End avoidance
        if active_pothole:
            dist = ego_loc.distance(active_pothole)
            vec = active_pothole - ego_loc
            yaw = math.radians(ego_tf.rotation.yaw)
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            dotp = fwd.dot(np.array([vec.x, vec.y]))

            if dist < POTHOLE_END_DIST and dotp < 0:
                print("[POTHOLE] Passed")
                active_pothole = None

        # ---------------- CONTROL ----------------
        if closest_ped and closest_dist < STOP_DISTANCE:
            manual_brake = True
            last_brake_time = now
            control = carla.VehicleControl(throttle=0, brake=1.0)

        else:
            if manual_brake and now - last_brake_time > 2.0 and closest_dist > CLEAR_DISTANCE:
                manual_brake = False

            if not reached_destination:
                control = agent.run_step()

                # -------- Curve Slowdown --------
                yaw = math.radians(ego_tf.rotation.yaw)
                fwd = np.array([math.cos(yaw), math.sin(yaw)])
                ahead_loc = carla.Location(
                    ego_loc.x + fwd[0] * 8,
                    ego_loc.y + fwd[1] * 8,
                    ego_loc.z
                )
                wp_ahead = world_map.get_waypoint(ahead_loc, project_to_road=True)

                if wp_ahead:
                    road_yaw = math.radians(wp_ahead.transform.rotation.yaw)
                    diff = abs((road_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
                    if diff > math.radians(25):
                        control.throttle = min(control.throttle, 0.25)

                # -------- Strong pothole avoidance --------
                if active_pothole is not None:
                    right_vec = np.array([math.sin(yaw), -math.cos(yaw)])

                    # Target point offset to side of pothole
                    target = carla.Location(
                        x=active_pothole.x + avoid_side * right_vec[0] * POTHOLE_AVOID_OFFSET,
                        y=active_pothole.y + avoid_side * right_vec[1] * POTHOLE_AVOID_OFFSET,
                        z=ego_loc.z
                    )

                    dx = target.x - ego_loc.x
                    dy = target.y - ego_loc.y
                    vec = np.array([dx, dy])
                    dotv = fwd.dot(vec)
                    detv = fwd[0] * vec[1] - fwd[1] * vec[0]
                    angle = math.atan2(detv, dotv)

                    # Override steering instead of adding
                    raw_steer = POTHOLE_STEER_GAIN * angle
                    raw_steer = max(-0.5, min(0.5, raw_steer))
                    control.steer = raw_steer

                    # Slow down while avoiding
                    control.throttle = min(control.throttle, 0.20)

                    # Extra brake when very close
                    dist_ph = ego_loc.distance(active_pothole)
                    if dist_ph < 6.0:
                        control.brake = max(control.brake, 0.3)

                if agent.done():
                    reached_destination = True
                    idle_after_reach = 0

            else:
                control = carla.VehicleControl(throttle=0, brake=0.7)
                idle_after_reach += 1
                if idle_after_reach > 60:
                    running = False

        ego.apply_control(control)

        # ---------------- Draw potholes ----------------
        for ph in potholes:
            draw_pothole(world, ph)

        # ---------------- Cameras ----------------
        if third_frame is not None:
            surf = pygame.surfarray.make_surface(np.rot90(third_frame))
            display.blit(surf, (0, 0))

        if front_frame is not None:
            surf2 = pygame.surfarray.make_surface(np.rot90(front_frame))
            display.blit(surf2,
                         (DISPLAY_W - FRONT_W - 10,
                          DISPLAY_H - FRONT_H - 10))

        # ---------------- Minimap ----------------
        mm = minimap_bg.copy()

        prev = None
        for loc in route_points[::5]:
            p = world_to_minimap(loc.x, loc.y, bounds)
            if prev:
                pygame.draw.line(mm, (0, 150, 255), prev, p, 2)
            prev = p

        for ph in potholes:
            px, py = world_to_minimap(ph.x, ph.y, bounds)
            col = (255, 255, 0) if ph == active_pothole else (255, 0, 0)
            pygame.draw.circle(mm, col, (px, py), 4)

        gx, gy = world_to_minimap(goal.location.x, goal.location.y, bounds)
        pygame.draw.circle(mm, (0, 255, 0), (gx, gy), 4)

        ex, ey = world_to_minimap(ego_loc.x, ego_loc.y, bounds)
        pygame.draw.circle(mm, (255, 255, 255), (ex, ey), 4)

        display.blit(mm, (MINIMAP_MARGIN, MINIMAP_MARGIN))

        # HUD
        hud = [
            f"Ego: {ego.id}",
            f"Ped: {'YES' if closest_ped else 'NO'}",
            f"Pothole Avoid: {'YES' if active_pothole else 'NO'}",
        ]
        y = 10
        for line in hud:
            text = font.render(line, True, (255, 255, 255))
            display.blit(text, (DISPLAY_W - text.get_width() - 10, y))
            y += 18

        pygame.display.flip()
        clock.tick(FPS)

    # --------------- Cleanup ----------------
    for c in walker_controllers:
        try:
            c.stop()
        except Exception:
            pass

    world.tick()

    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
