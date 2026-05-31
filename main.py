import cv2

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to close the camera.")
    
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()