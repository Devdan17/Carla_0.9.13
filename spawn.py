import carla
import random
import time

def main():
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Load Town03
    world = client.load_world("Town03")

    blueprint_library = world.get_blueprint_library()

    # Choose a vehicle blueprint (Tesla Model 3)
    vehicle_bp = blueprint_library.filter("model3")[0]

    # Get all spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Random spawn point
    spawn_point = random.choice(spawn_points)

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    print("Vehicle spawned with ID:", vehicle.id)

    # Keep it alive briefly
    time.sleep(2)

if __name__ == "__main__":
    main()
