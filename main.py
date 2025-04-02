from camera.ai_deck_stream import AIDeckStream
from tracking.threshold_tracker import ThresholdTracker
from drone.controller import DroneController

def main():
    camera = AIDeckStream()
    tracker = ThresholdTracker()
    drone = DroneController()

    drone.connect()

    try:
        while True:
            frame = camera.get_frame()
            result = tracker.detect_red_object(frame)

            if result:
                direction = tracker.compute_position(result['center'], frame.shape)
                drone.send_command(direction)
    finally:
        drone.stop()

if __name__ == "__main__":
    main()
