"""Q3: Calibrate once, save, then undistort a live webcam feed.
Step 1: python q3_undistort_live.py calibrate calib_images/   (chessboard photos, *.jpg/png)
        or: python q3_undistort_live.py capture                (press SPACE to grab ~15 views, ESC to finish)
Step 2: python q3_undistort_live.py live                       (press q to quit)
Default chessboard: 9x6 inner corners, 25 mm squares (change below).
"""
import sys
import glob
import cv2
import numpy as np

BOARD = (9, 6)
SQUARE = 25.0
CALIB_FILE = "camera_calib.npz"


def calibrate(images):
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE
    objpts, imgpts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, c = cv2.findChessboardCorners(gray, BOARD)
        if ok:
            c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
            objpts.append(objp); imgpts.append(c)
    if len(objpts) < 5:
        sys.exit(f"Only {len(objpts)} usable views; need at least ~10.")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    np.savez(CALIB_FILE, K=K, dist=dist, size=size)
    print(f"Used {len(objpts)} views, RMS reprojection error = {rms:.3f}px\nK=\n{K}\ndist={dist.ravel()}")
    print("Saved", CALIB_FILE)


def capture():
    cap = cv2.VideoCapture(0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        found, c = cv2.findChessboardCorners(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), BOARD)
        v = f.copy()
        cv2.drawChessboardCorners(v, BOARD, c, found)
        cv2.putText(v, f"SPACE=grab ({len(frames)})  ESC=done", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("capture", v)
        k = cv2.waitKey(1) & 0xFF
        if k == 32 and found:
            frames.append(f)
        elif k == 27:
            break
    cap.release(); cv2.destroyAllWindows()
    calibrate(frames)


def live():
    data = np.load(CALIB_FILE)
    K, dist = data["K"], data["dist"]
    cap = cv2.VideoCapture(0)
    ok, f = cap.read()
    h, w = f.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        und = cv2.undistort(f, K, dist, None, newK)
        x, y, rw, rh = roi
        und = und[y:y + rh, x:x + rw]                      # crop black borders
        cv2.imshow("Original | Undistorted",
                   np.hstack([f, cv2.resize(und, (w, h))]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "calibrate":
        files = sorted(glob.glob(sys.argv[2] + "/*.jpg") + glob.glob(sys.argv[2] + "/*.png"))
        calibrate([cv2.imread(p) for p in files])
    elif mode == "capture":
        capture()
    else:
        live()
