"""
The Tiny Fingerprint — Linear Scale
------------------------------------
A digital fingerprint from the crime scene is too small to read.
This script:
  1. Manually builds a 2x2 scaling matrix (300% on x, 300% on y).
  2. Applies it to the fingerprint's point coordinates.
  3. Calculates the translation needed to keep the fingerprint
     perfectly centered on screen after it grows.

No external image is required — the fingerprint is represented as a
set of 2D points (x, y), which is exactly how scaling math works
whether the points come from a synthetic shape or real pixel/vector
data extracted from a photo.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# STEP 1: Build the fingerprint (a set of 2D points)
# ---------------------------------------------------------------
# For demonstration we generate a simple spiral of concentric arcs
# to *look* like a fingerprint ridge pattern. Swap this out for real
# (x, y) coordinates extracted from an actual image if you have one.

def make_fingerprint(num_rings=6, points_per_ring=200, base_radius=1.0):
    points = []
    for ring in range(1, num_rings + 1):
        radius = base_radius * ring / num_rings
        theta = np.linspace(0, 2 * np.pi * 0.85, points_per_ring)  # arcs, not full circles
        # small radial wobble so it reads as an organic ridge, not a perfect circle
        wobble = 0.03 * np.sin(theta * 5 + ring)
        x = (radius + wobble) * np.cos(theta)
        y = (radius + wobble) * np.sin(theta)
        points.append(np.column_stack([x, y]))
    return np.vstack(points)  # shape: (N, 2)


fingerprint = make_fingerprint()

# ---------------------------------------------------------------
# STEP 2: Manually build the 2x2 scaling matrix
# ---------------------------------------------------------------
# A scaling transform in 2D is represented as:
#
#     | sx   0 |
# S = |        |
#     | 0   sy |
#
# Multiplying S by a point (x, y) gives (sx*x, sy*y).
# "Enlarge by 300%" means the new size is 300% OF the original,
# i.e. a scale factor of 3.0 (not 4.0 — 300% growth would be 4x,
# but "enlarge BY 300%" / "scale TO 300%" both conventionally mean
# a multiplier of 3.0, which we use here).

scale_x = 3.0
scale_y = 3.0

S = np.array([
    [scale_x, 0.0],
    [0.0,     scale_y]
])

print("Scaling matrix S:")
print(S)

# ---------------------------------------------------------------
# STEP 3: Apply the scaling matrix to every point
# ---------------------------------------------------------------
# For a point p = (x, y), the scaled point is S @ p.
# We do this for the whole array at once: (S @ points.T).T

scaled_fingerprint_raw = (S @ fingerprint.T).T

# ---------------------------------------------------------------
# STEP 4: Calculate the centering translation
# ---------------------------------------------------------------
# Scaling multiplies every coordinate relative to the ORIGIN (0, 0),
# not relative to the shape's own center. So unless the shape's
# center already sits exactly on the origin, scaling will push it
# off to one side.
#
# Fix: find the fingerprint's original center C, then compute how
# far the scaled center has drifted from C, and shift everything
# back by that amount.
#
#   C            = center of original points
#   C_scaled     = S @ C   (where the center ends up after scaling)
#   translation  = C - C_scaled   (how far to shift it back)
#
# Equivalently: translation = C * (1 - scale), per axis.

center = fingerprint.mean(axis=0)          # (cx, cy) of the ORIGINAL fingerprint
center_scaled = S @ center                 # where that center point lands after scaling
translation = center - center_scaled       # shift needed to restore the original center

print(f"\nOriginal center:        {center}")
print(f"Center after scaling:   {center_scaled}")
print(f"Recentering translation:{translation}")

# ---------------------------------------------------------------
# STEP 5: Apply the translation to get the final, centered result
# ---------------------------------------------------------------
scaled_fingerprint_centered = scaled_fingerprint_raw + translation

# Sanity check: the new center should equal the original center
new_center = scaled_fingerprint_centered.mean(axis=0)
print(f"New center after fix:   {new_center}")
assert np.allclose(new_center, center, atol=1e-9), "Centering failed!"
print("\n✅ Fingerprint enlarged 300% on x and y, and re-centered on the original spot.")

# ---------------------------------------------------------------
# STEP 6: Visualize original vs. enlarged (before/after centering)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(fingerprint[:, 0], fingerprint[:, 1],
           s=4, color="dimgray", label="Original (100%)")
ax.scatter(scaled_fingerprint_raw[:, 0], scaled_fingerprint_raw[:, 1],
           s=4, color="lightcoral", alpha=0.5, label="Scaled 300% (uncentered)")
ax.scatter(scaled_fingerprint_centered[:, 0], scaled_fingerprint_centered[:, 1],
           s=4, color="steelblue", label="Scaled 300% (re-centered)")

ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.scatter(*center, color="black", marker="x", s=80, zorder=5, label="Original center")

ax.set_aspect("equal")
ax.set_title("Fingerprint Scaling: 300% Enlargement + Recentering")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fingerprint_scaling.png", dpi=150)
print("\nSaved visualization to fingerprint_scaling.png")
