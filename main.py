import cv2
import time
from ultralytics import YOLO


def main():
    # Load model
    model = YOLO("yolo26n.onnx")

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to close.")

    # Initialize FPS timer
    prev_time = time.time()

    # Main loop
    while True:

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model.predict(source=frame_rgb, conf=0.25, verbose=False)

        # Draw bounding boxes and convert back to BGR
        annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # Count detected objects
        num_boxes = len(results[0].boxes) if results[0].boxes is not None else 0

        # Overlay FPS and object count
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Objects: {num_boxes}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # Display frame
        cv2.imshow('YOLO26', annotated)

        # Exit on 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()