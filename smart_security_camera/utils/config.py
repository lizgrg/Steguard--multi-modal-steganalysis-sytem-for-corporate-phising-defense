# utils/config.py
"""Central configuration — Smart Security Camera CSY3058"""

# ── Video Source ──────────────────────────────────────────────────────────────
VIDEO_SOURCE        = 0
FRAME_WIDTH         = 854
FRAME_HEIGHT        = 480
TARGET_FPS          = 30

# ── MOG2 Background Subtractor ────────────────────────────────────────────────
MOG2_HISTORY        = 200
MOG2_VAR_THRESHOLD  = 40          # lowered: more sensitive to motion
MOG2_DETECT_SHADOWS = True
MOG2_LEARNING_RATE  = 0.02

# ── Motion Detection Thresholds ───────────────────────────────────────────────
MIN_CONTOUR_AREA    = 3500         # lowered: catches smaller movements
MAX_FOREGROUND_RATIO = 0.55
MOTION_BLUR_KERNEL  = (11, 11)
MORPH_KERNEL_SIZE   = (5, 5)       # smaller: less aggressive merging

# ── Object Classification ─────────────────────────────────────────────────────
HOG_WIN_STRIDE      = (4, 4)       # finer stride: more detection passes
HOG_PADDING         = (8, 8)       # more padding around crop
HOG_SCALE           = 1.05
HOG_HIT_THRESHOLD   = -0.5         # negative: much more aggressive HOG
VEHICLE_MIN_AREA    = 90000        # raised: only very large blobs = vehicle
HUMAN_MIN_ASPECT    = 0.8          # lowered: forgiving for webcam angles

# ── Centroid Tracker ──────────────────────────────────────────────────────────
MAX_DISAPPEARED     = 15
MAX_DISTANCE        = 90

# ── Video Saving ──────────────────────────────────────────────────────────────
OUTPUT_DIR          = "output"
CLIP_BUFFER_BEFORE  = 20
CLIP_BUFFER_AFTER   = 40
VIDEO_CODEC         = "XVID"
CLIP_FPS            = 20

# ── Camera Calibration ────────────────────────────────────────────────────────
APPLY_UNDISTORTION  = False
CAMERA_MATRIX       = None
DIST_COEFFICIENTS   = None

# ── UI Theme ──────────────────────────────────────────────────────────────────
ALERT_COLOURS = {
    "HUMAN":   "#00FFA3",
    "VEHICLE": "#FFB547",
    "UNKNOWN": "#888899",
}
DASHBOARD_THEME = {
    "bg":      "#0A0A0F",
    "panel":   "#13131A",
    "border":  "#1E1E2E",
    "text":    "#E0E0F0",
    "subtext": "#666680",
    "accent":  "#00FFA3",
    "danger":  "#FF4D6D",
    "warning": "#FFB547",
}