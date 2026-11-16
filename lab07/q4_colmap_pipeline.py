"""Q4: 3D reconstruction with COLMAP (features -> matching -> poses/SfM -> dense point cloud -> mesh).
Install COLMAP (https://colmap.github.io, Windows: download release zip, add COLMAP.bat to PATH).
Dense stereo + fusion needs a CUDA GPU; without one the script stops after the sparse model.

Usage: python q4_colmap_pipeline.py <images_dir> <workspace_dir> [--no-dense]
Put 30-100 overlapping photos of a textured object/scene in <images_dir>.
"""
import subprocess
import sys
import shutil
from pathlib import Path

COLMAP = shutil.which("colmap") or shutil.which("COLMAP.bat") or "colmap"


def run(*args):
    cmd = [COLMAP, *map(str, args)]
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    images, ws = Path(sys.argv[1]), Path(sys.argv[2])
    dense = "--no-dense" not in sys.argv
    db, sparse, dense_dir = ws / "database.db", ws / "sparse", ws / "dense"
    sparse.mkdir(parents=True, exist_ok=True)

    # 1. Detect features (SIFT)
    run("feature_extractor", "--database_path", db, "--image_path", images,
        "--ImageReader.single_camera", 1)
    # 2. Match features across images
    run("exhaustive_matcher", "--database_path", db)
    # 3. Estimate camera poses + sparse 3D points (incremental SfM)
    run("mapper", "--database_path", db, "--image_path", images, "--output_path", sparse)
    model = sparse / "0"
    run("model_converter", "--input_path", model, "--output_path", ws / "sparse.ply",
        "--output_type", "PLY")
    print("Sparse point cloud ->", ws / "sparse.ply")
    if not dense:
        return

    # 4. Dense reconstruction -> point cloud
    run("image_undistorter", "--image_path", images, "--input_path", model,
        "--output_path", dense_dir, "--output_type", "COLMAP")
    run("patch_match_stereo", "--workspace_path", dense_dir,
        "--PatchMatchStereo.geom_consistency", "true")
    run("stereo_fusion", "--workspace_path", dense_dir,
        "--output_path", dense_dir / "fused.ply")
    # 5. Mesh (Poisson; Delaunay also available via delaunay_mesher)
    run("poisson_mesher", "--input_path", dense_dir / "fused.ply",
        "--output_path", dense_dir / "meshed-poisson.ply")
    print("\nDone. Open fused.ply (point cloud) and meshed-poisson.ply (mesh) in MeshLab / COLMAP GUI.")


if __name__ == "__main__":
    main()
