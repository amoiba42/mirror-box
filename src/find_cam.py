import cv2

print("Scanning for cameras...")

# Scan the first 10 indices
for index in range(10):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera found at Index {index}!")
            # Save a test image to prove it works
            cv2.imwrite(f"cam_test_{index}.jpg", frame)
            cap.release()
        else:
            print(f"⚠️  Index {index} opened, but returned no frame.")
            cap.release()
    else:
        pass # Port closed

print("Scan complete.")