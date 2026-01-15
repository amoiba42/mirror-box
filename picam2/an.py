from picamera2 import Picamera2
from libcamera import controls

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# --- SETTING CONTROLS FOR NATURAL LOOK ---
picam2.set_controls({
    # 1. Reduce Saturation (Default is 1.0). 
    # Lowering this counteracts the "popping" colors caused by IR sensitivity.
    "Saturation": 0.6, 

    # 2. Set Sharpness (Default is 1.0).
    # Reducing slightly to 0.5 - 0.8 removes the "digital" look of edges.
    "Sharpness": 0.75,

    # 3. Ensure Auto White Balance is on Normal/Auto
    "AwbMode": controls.AwbModeEnum.Auto,
    
    # 4. Brightness (Default is 0.0). 
    # Leave at 0.0 unless the image is consistently too dark/bright.
    "Brightness": 0.0,

    # 5. Contrast (Default is 1.0).
    # Sometimes IR wash makes images look flat; if so, bump this to 1.1 or 1.2.
    # For now, keep it standard to avoid "popping".
    "Contrast": 1.0
})

# Keep the script running to view the feed
input("Press Enter to stop...")
picam2.stop()