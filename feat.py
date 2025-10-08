import os, glob
import pandas as pd
import numpy as np

os.makedirs("features", exist_ok=True)

fps = 30
window = 60
stride = 30

keep_ids = [
    0,   # Nose
    7, 8,   # Left Ear, Right Ear
    11, 12, # Left Shoulder, Right Shoulder
    13, 14, # Left Elbow, Right Elbow
    15, 16, # Left Wrist, Right Wrist
    17, 18, # Left Pinky, Right Pinky
    19, 20, # Left Index, Right Index
    21, 22, # Left Thumb, Right Thumb
]

def rotate_point(x, y, cx, cy, angle):
    xr = (x - cx) * np.cos(-angle) - (y - cy) * np.sin(-angle)
    yr = (x - cx) * np.sin(-angle) + (y - cy) * np.cos(-angle)
    return xr, yr

def safe_mean(arr):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return np.nanmean(arr)

def safe_std(arr):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return np.nanstd(arr)

def extract_features(win):
    feats = {}

    points = {}
    for idx in keep_ids:
        try:
            x, y, z = win[f"landmark_{idx}_x"], win[f"landmark_{idx}_y"], win[f"landmark_{idx}_z"]
            vis = win[f"landmark_{idx}_vis"]
            mask = vis < 0.5
            x = x.mask(mask)
            y = y.mask(mask)
            z = z.mask(mask)


            lx, ly = win["landmark_11_x"], win["landmark_11_y"]
            rx, ry = win["landmark_12_x"], win["landmark_12_y"]
            msx, msy = (lx + rx) / 2, (ly + ry) / 2

            angle = np.arctan2(ry - ly, rx - lx)
            xr, yr = rotate_point(x, y, msx, msy, angle)

            shoulder_width = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
            shoulder_width = np.where(shoulder_width == 0, 1e-6, shoulder_width)

            xr /= shoulder_width
            yr /= shoulder_width
            zr = z / shoulder_width

            points[idx] = (xr.values, yr.values, zr.values)
        except KeyError:
            points[idx] = (np.full(window, np.nan),
                           np.full(window, np.nan),
                           np.full(window, np.nan))

    def dist3D(id1, id2):
        x1, y1, z1 = points[id1]
        x2, y2, z2 = points[id2]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    feats["dist_wrists_mean"] = safe_mean(dist3D(15, 16))
    feats["noseL_wrist_mean"] = safe_mean(dist3D(0, 15))
    feats["noseR_wrist_mean"] = safe_mean(dist3D(0, 16))

    def angle2D(a, b, c):
        ax, ay, _ = points[a]
        bx, by, _ = points[b]
        cx, cy, _ = points[c]
        v1 = np.stack([ax - bx, ay - by], axis=1)
        v2 = np.stack([cx - bx, cy - by], axis=1)
        dot = np.sum(v1 * v2, axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        ang = np.arccos(np.clip(dot / norm, -1, 1))
        return ang

    feats["elbowL_mean"] = safe_mean(angle2D(11, 13, 15))
    feats["elbowR_mean"] = safe_mean(angle2D(12, 14, 16))

    for side, (w, p, i, t) in zip(
        ["L", "R"],
        [(15, 17, 19, 21), (16, 18, 20, 22)]
    ):
        feats[f"{side}_wrist_pinky_mean"] = safe_mean(dist3D(w, p))
        feats[f"{side}_wrist_index_mean"] = safe_mean(dist3D(w, i))
        feats[f"{side}_wrist_thumb_mean"] = safe_mean(dist3D(w, t))

    for idx in [15, 16, 17, 18, 19, 20, 21, 22]:
        x, y, z = points[idx]
        vx, vy, vz = np.diff(x), np.diff(y), np.diff(z)
        vel = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        feats[f"landmark_{idx}_vel_mean"] = safe_mean(vel)
        feats[f"landmark_{idx}_vel_std"] = safe_std(vel)

    return feats

input_files = glob.glob("output csv/*.csv")

for input_csv in input_files:
    print(f"Processing {input_csv}")
    df = pd.read_csv(input_csv)

    features = []
    for start in range(0, len(df) - window, stride):
        win = df.iloc[start:start + window]
        feat_dict = extract_features(win)
        feat_dict = {"start_frame": start, "end_frame": start + window, **feat_dict}
        features.append(feat_dict)

    feat_df = pd.DataFrame(features).fillna(0.0)

    base_name = os.path.basename(input_csv)
    out_name = f"features/features_{os.path.splitext(base_name)[0]}.csv"
    feat_df.to_csv(out_name, index=False)
    print(f"Saved {out_name}")