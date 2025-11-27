MODEL_PATH = "../Model_1_Simple_CNN_200_50_20250606_113059.h5"   # your trained model
INPUT_H, INPUT_W = 50, 200     # model expects 200×50
PADDING = 20                   # crop padding

# Laptop screen parameters
SCREEN_W = 1366                # px
SCREEN_H = 768                 # px

SCREEN_W_MM = 303.0            # mm (physical)
SCREEN_H_MM = 171.0            # mm

DIST_MM = 500.0                # 50 cm user–screen distance

LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]

# Calibration parameters (correction offsets)
CALIB_OFFSETS = {
    "top_left": (0.0, 0.0),
    "top_right": (0.0, 0.0),
    "bottom_left": (0.0, 0.0),
    "bottom_right": (0.0, 0.0),
    "center": (0.0, 0.0)
}

CALIBRATION_FILE = "calibration_params.json"
CALIBRATION_TIME = 5  # seconds per corner