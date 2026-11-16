"""Q2: Perspective correction (unwarping).
Click the 4 corners of the tilted document (TL, TR, BR, BL order), press ENTER/SPACE to warp.
Usage: python q2_perspective_correction.py tilted.jpg [out.jpg]
Or non-interactive: pass 8 numbers after the image: x1 y1 x2 y2 x3 y3 x4 y4
"""
import sys
import cv2
import numpy as np


def order_points(pts):
    """Return points ordered TL, TR, BR, BL."""
    pts = np.array(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def apply_perspective_transform(image, src_pts, dst_pts, out_size):
    """Inverse use: src = tilted quadrilateral in the photo, dst = perfect rectangle."""
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpPerspective(image, M, out_size), M


def unwarp(image, corners):
    tl, tr, br, bl = order_points(corners)
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    rect = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]   # destination: perfect rectangle
    return apply_perspective_transform(image, [tl, tr, br, bl], rect, (W, H))


def pick_corners(image):
    pts, disp = [], image.copy()

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Select 4 corners", disp)

    cv2.imshow("Select 4 corners", disp)
    cv2.setMouseCallback("Select 4 corners", cb)
    while len(pts) < 4 or cv2.waitKey(0) not in (13, 32):
        if cv2.waitKey(20) == 27:
            sys.exit()
    cv2.destroyAllWindows()
    return pts


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit("Could not read image")
    if len(sys.argv) >= 10:
        corners = np.float32(sys.argv[2:10]).reshape(4, 2)
    else:
        corners = pick_corners(img)
    out, M = unwarp(img, corners)
    print("Homography:\n", M)
    cv2.imwrite(sys.argv[2] if len(sys.argv) == 3 else "unwarped.jpg", out)
    cv2.imshow("Original", img)
    cv2.imshow("Unwarped (head-on view)", out)
    cv2.waitKey(0)
