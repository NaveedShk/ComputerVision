"""Q1: Image transformations (a-e).
Usage: python q1_transformations.py [image_path]
pip install opencv-python numpy matplotlib
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load(path):
    img = cv2.imread(path) if path else None
    if img is None:  # fallback synthetic image (checkerboard + text)
        img = np.full((300, 400, 3), 255, np.uint8)
        for i in range(0, 400, 50):
            for j in range(0, 300, 50):
                if (i // 50 + j // 50) % 2 == 0:
                    cv2.rectangle(img, (i, j), (i + 50, j + 50), (200, 120, 40), -1)
        cv2.putText(img, "LAB 07", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return img


def show(items, cols=3, title=""):
    rows = (len(items) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for k, (name, im) in enumerate(items):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()


img = load(sys.argv[1] if len(sys.argv) > 1 else None)
h, w = img.shape[:2]

# ---------------- (a) 2D transformations ----------------
T = np.float32([[1, 0, 80], [0, 1, 40]])                       # translation
R = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)           # rotation 30 deg
S = np.float32([[0.6, 0, 0], [0, 0.6, 0]])                     # scaling
SH = np.float32([[1, 0.4, 0], [0, 1, 0]])                      # shearing (x)
show([("Original", img),
      ("Translation", cv2.warpAffine(img, T, (w, h))),
      ("Rotation 30°", cv2.warpAffine(img, R, (w, h))),
      ("Scaling 0.6x", cv2.warpAffine(img, S, (w, h))),
      ("Shear x=0.4", cv2.warpAffine(img, SH, (int(w * 1.4), h)))],
     title="Q1(a) 2D transformations")

# ---------------- (b) 3D transformation ----------------
# Unit cube, rotated and translated along x, y, z, then projected with a pinhole camera.
cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.sum(cube[i] != cube[j]) == 1]


def rot_xyz(ax, ay, az):
    ax, ay, az = np.radians([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project(P, f=3.0):
    return f * P[:, :2] / P[:, 2:3]


fig = plt.figure(figsize=(15, 4))
cases = [("Original (z+6)", np.eye(3), [0, 0, 6]),
         ("Rotate x=40°", rot_xyz(40, 0, 0), [0, 0, 6]),
         ("Rotate y=40°", rot_xyz(0, 40, 0), [0, 0, 6]),
         ("Rotate z=40° + move x,y,z", rot_xyz(0, 0, 40), [2, 1, 8])]
for k, (name, Rm, t) in enumerate(cases):
    P = cube @ Rm.T + np.array(t, float)   # P' = R P + t
    p = project(P)
    ax = fig.add_subplot(1, 4, k + 1)
    for i, j in edges:
        ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], "b-")
    ax.set_title(name); ax.set_aspect("equal"); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
fig.suptitle("Q1(b) 3D cube transformations (perspective-projected)")

# ---------------- (c) Homogeneous coordinates ----------------
pt = np.array([50, 30])
pt_h = np.array([50, 30, 1])
print("(c) 2D point", pt, "-> homogeneous", pt_h)
T3 = np.array([[1, 0, 80], [0, 1, 40], [0, 0, 1]])
print("    translation as ONE matrix multiply:", T3 @ pt_h)
print("    scaled homogeneous (100,60,2) is the same point:", np.array([100, 60, 2])[:2] / 2)
print("""    WHY: translation is not a linear map in (x,y), so it can't be a 2x2 matrix. The extra
    coordinate lets translation, rotation, scale, shear AND perspective all be 3x3 matrices that
    compose by matrix multiplication; it also represents points at infinity (w=0) and
    makes perspective division (x/w, y/w) possible.""")

# ---------------- (d) Affine transformation ----------------
src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
dst = np.float32([[40, 30], [w - 80, 60], [90, h - 40]])
A = cv2.getAffineTransform(src, dst)
aff = cv2.warpAffine(img, A, (w, h))
show([("Original", img), ("Affine", aff)], cols=2, title="Q1(d) Affine transform")
print("(d) Affine matrix:\n", A)
print("    Straight lines stay straight, parallel lines stay parallel (ratios along lines preserved);"
      " angles/lengths are NOT preserved.")

# ---------------- (e) Projective transform (homography) ----------------
# Simulate a slanted book/page, then fix it with a homography.
tilt_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
tilt_dst = np.float32([[60, 50], [w - 30, 10], [w - 90, h - 20], [20, h - 70]])
Htilt = cv2.getPerspectiveTransform(tilt_src, tilt_dst)
tilted = cv2.warpPerspective(img, Htilt, (w, h))
Hfix = cv2.getPerspectiveTransform(tilt_dst, tilt_src)      # inverse mapping
fixed = cv2.warpPerspective(tilted, Hfix, (w, h))
show([("Original", img), ("Tilted", tilted), ("Corrected (homography)", fixed)],
     title="Q1(e) Perspective correction")

plt.show()
