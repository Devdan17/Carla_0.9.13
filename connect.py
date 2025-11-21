import carla
import time


def connect_to_town03(max_retries: int = 5):
    """
    Connect to CARLA 0.9.12 Simulator and load Town03.
    Retries several times if the simulator is still loading.
    """

    print("Connecting to CARLA server on localhost:2000 ...")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\nAttempt {attempt}/{max_retries} ...")

            # Create client
            client: carla.Client = carla.Client("localhost", 2000)
            client.set_timeout(30.0)  # 30 seconds timeout

            # Load the world
            world: carla.World = client.load_world("Town03")

            print("Connected to CARLA successfully!")
            print("Loaded Map:", world.get_map().name)

            return client, world

        except Exception as e:
            print(f"Failed: {e}")
            if attempt < max_retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("\nCARLA connection failed after all retries.")
                print("Make sure:")
                print(" - CarlaUE4.exe is fully loaded")
                print(" - Town03 is visible")
                print(" - The engine finished loading")
                print(" - Python API version matches simulator")
                return None, None


if __name__ == "__main__":
    connect_to_town03()
