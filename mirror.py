# modified_mirror_flip.py
import cv2

# --- Configuration ---
CAMERA_INDEX = 0
WINDOW_TITLE = "Interactive Mirror Feed"
# 0: vertical, 1: horizontal, -1: diagonal, 2: none (original)
FLIP_MODES = {
    ord('h'): (1, "Horizontal Flip (Mirror)"),
    ord('v'): (0, "Vertical Flip"),
    ord('d'): (-1, "Diagonal Flip"),
    ord('n'): (2, "No Flip"),
}
QUIT_KEY = ord('q')

# --- Initialization ---
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise SystemExit(f"Cannot open camera at index {CAMERA_INDEX}")

# Initial state
current_flip_code = 1  # Start with no flip
current_mode_name = "Horizontal Flip (Mirror)"

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Store original frame for consistent transformations
    original_frame = frame.copy()
    
    # 1. Apply Flip directly to original frame
    if current_flip_code in [0, 1, -1]:
        processed_frame = cv2.flip(original_frame, current_flip_code)
    else: # No flip (code 2)
        processed_frame = original_frame.copy()
        
    # 2. Add Text Overlay
    text = f"Mode: {current_mode_name} | Press 'h', 'v', 'd', 'n' to change | 'q' to quit"
    cv2.putText(
        processed_frame, 
        text, 
        (10, 30), # Position
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7,      # Font scale
        (255, 0, 0), # BGR Color (Blue)
        1         # Thickness
    )

    # 3. Display
    cv2.imshow(WINDOW_TITLE, processed_frame)
    
    # 4. Handle Key Press
    key = cv2.waitKey(1) & 0xFF
    
    if key == QUIT_KEY:
        break
    
    if key in FLIP_MODES:
        current_flip_code, current_mode_name = FLIP_MODES[key]
        print(f"Switched to: {current_mode_name}")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()