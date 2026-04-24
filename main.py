import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO

from projection import (
    load_image, load_lidar, load_calibration, load_labels,
    project_lidar_to_image, filter_to_image, depth_to_color_rainbow,
    distance_from_pointcloud, gt_distance,
    get_3d_box_corners_cam, cam_to_lidar,
    evaluate_sub_pointcloud, compute_iou_2d,
)

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    XLSX_OK = True
except ImportError:
    XLSX_OK = False
    print("WARNING: openpyxl not installed. Run: pip install openpyxl")



BASE_PATH = r"D:\Lidar_Radar\KITTI-Selection_incl_LiDAR_2026"

IMAGE_DIR = os.path.join(BASE_PATH, "data_object_image_2", "training", "image_2")
LIDAR_DIR = os.path.join(BASE_PATH, "data_object_velodyne", "training", "velodyne")
CALIB_DIR = os.path.join(BASE_PATH, "data_object_calib", "training", "calib")
LABEL_DIR = os.path.join(BASE_PATH, "data_object_label_2", "training", "label_2")

OUT_DIR  = r"D:\Lidar_Radar\Output"
PROJ_DIR = r"D:\Lidar_Radar\Projection_images"
DET_DIR  = r"D:\Lidar_Radar\car_detections"

YOLO_MODEL_PATH = "yolov8n-seg.pt"
CAR_CLASS_ID    = 2
CONF_THRESHOLD  = 0.18
IOU_THRESHOLD   = 0.45
MASK_THRESHOLD  = 0.35
MASK_DILATION   = 1
AP_THRESHOLDS   = np.arange(0.5, 0.96, 0.05)
AP_CURVE_PATH   = os.path.join(OUT_DIR, "AP_vs_IoU.png")

BEV_COLORS = [
    'red', 'blue', 'green', 'yellow',
    'magenta', 'cyan', 'orange', 'purple',
]

ALL_SCORES      = []
TOTAL_GT_CARS   = 0
ALL_PREDICTIONS = []
ALL_GT_BOXES    = []




def startup_checks():
    """Verify all required directories exist."""
    print("=" * 60)
    print("  LiDAR + Camera Sensor Fusion  |  KITTI Dataset")
    print("=" * 60)

    ok = True
    checks = [
        ("BASE_PATH",  BASE_PATH),
        ("IMAGE_DIR",  IMAGE_DIR),
        ("LIDAR_DIR",  LIDAR_DIR),
        ("CALIB_DIR",  CALIB_DIR),
        ("LABEL_DIR",  LABEL_DIR),
    ]
    for name, path in checks:
        exists = os.path.isdir(path)
        tag = "OK" if exists else "MISSING"
        print(f"  [{tag:7s}]  {name}: {path}")
        if not exists:
            ok = False

    if not ok:
        print("\nERROR: One or more folders not found.")
        print("Fix BASE_PATH in main.py and try again.")
        exit(1)

    for d in [OUT_DIR, PROJ_DIR, DET_DIR]:
        os.makedirs(d, exist_ok=True)
    print("\nOutput folders ready.")
    print()


def get_scene_ids():
    """Get list of scene IDs from image directory."""
    ids = []
    image_dir_training = os.path.join(BASE_PATH, "data_object_image_2", "training", "image_2")
    if os.path.isdir(image_dir_training):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found = sorted(glob.glob(os.path.join(image_dir_training, ext)))
            ids += [os.path.splitext(os.path.basename(p))[0] for p in found]
    return sorted(set(ids))



def detect_cars_yolo(image, model):
    """Detect cars using YOLO and return masks, boxes, and scores."""
    results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
    masks, boxes, scores = [], [], []

    if results.masks is None:
        return masks, boxes, scores

    for i, cls_id in enumerate(results.boxes.cls):
        if int(cls_id) != CAR_CLASS_ID:
            continue

        mask = results.masks.data[i].cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = (mask > MASK_THRESHOLD).astype(np.uint8)
        if MASK_DILATION > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=MASK_DILATION)

        box   = results.boxes.xyxy[i].cpu().numpy().tolist()
        score = float(results.boxes.conf[i].cpu().numpy())

        masks.append(mask)
        boxes.append(box)
        scores.append(score)

    return masks, boxes, scores




def get_valid_projections(points, P2, Tr, R0, img_shape):
    """Project LiDAR points to image plane and filter to valid region."""
    pixels, valid_idx, depths = project_lidar_to_image(points, P2, Tr, R0)
    pixels, valid_idx, depths = filter_to_image(pixels, valid_idx, depths, img_shape)
    return pixels, valid_idx, depths




def assign_points_to_masks(pixels, valid_idx, points, masks):
    """Assign LiDAR points to detection masks."""
    sub_pcs = [[] for _ in masks]
    px_int  = pixels.astype(int)

    for j in range(len(px_int)):
        u = px_int[j, 0]
        v = px_int[j, 1]
        orig_idx = valid_idx[j]
        for m_idx in range(len(masks)):
            if masks[m_idx][v, u] == 1:
                sub_pcs[m_idx].append(orig_idx)
                break

    result = []
    for idxs in sub_pcs:
        if len(idxs) > 0:
            result.append(points[np.array(idxs)])
        else:
            result.append(np.empty((0, 4)))
    return result




def save_annotated_image(image, yolo_boxes, yolo_scores, car_labels, scene_id):
    """Save YOLO detections and GT boxes overlaid on image."""
    vis = image.copy()

    for lbl in car_labels:
        l, t, r, b = [int(v) for v in lbl['bbox_2d']]
        cv2.rectangle(vis, (l, t), (r, b), (220, 30, 30), 2)

    for box, score in zip(yolo_boxes, yolo_scores):
        x1, y1, x2, y2 = [int(v) for v in box]

        best_iou = 0.0
        for lbl in car_labels:
            iou = compute_iou_2d([x1, y1, x2, y2], lbl['bbox_2d'])
            if iou > best_iou:
                best_iou = iou

        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 210, 30), 2)
        label_txt = "Conf:{:.2f}  IoU:{:.2f}".format(score, best_iou)
        cv2.putText(
            vis, label_txt,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 220, 220), 2, cv2.LINE_AA
        )

    out_path = os.path.join(OUT_DIR, scene_id + "_annotated.png")
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("    -> Annotated : " + out_path)




def save_projection_images(image, pixels, depths, sub_pcs, P2, Tr, R0, scene_id):
    """Save depth map and car-only LiDAR projection."""
    H, W = image.shape[:2]

    colors_bgr   = depth_to_color_rainbow(depths)
    depth_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for k in range(len(pixels)):
        u = int(pixels[k, 0])
        v = int(pixels[k, 1])
        c = colors_bgr[k].tolist()
        cv2.circle(depth_canvas, (u, v), 1, c, -1)

    car_canvas = (image * 0.45).astype(np.uint8)
    for sub_pc in sub_pcs:
        if len(sub_pc) == 0:
            continue
        px_s, vi_s, dp_s = project_lidar_to_image(sub_pc, P2, Tr, R0)
        px_s, vi_s, dp_s = filter_to_image(px_s, vi_s, dp_s, image.shape)
        if len(dp_s) == 0:
            continue
        col_s = depth_to_color_rainbow(dp_s)
        for k in range(len(px_s)):
            u = int(px_s[k, 0])
            v = int(px_s[k, 1])
            c = col_s[k].tolist()
            cv2.circle(car_canvas, (u, v), 2, c, -1)

    cv2.putText(depth_canvas, "Projected Depth Map (Rainbow)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(car_canvas, "Car-only LiDAR Projection on RGB",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    combined = np.vstack([depth_canvas, car_canvas])
    out_path = os.path.join(PROJ_DIR, scene_id + "_projection.png")
    cv2.imwrite(out_path, combined)
    print("    -> Projection: " + out_path)




def save_bev_plot(points, sub_pcs, filtered_pcs, car_labels, Tr, scene_id):
    """Save Bird's Eye View plot of LiDAR and GT boxes."""
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # All points
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.25, label='All LiDAR')

    
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(sub_pcs))))
    for i, sub_pc in enumerate(sub_pcs):
        if len(sub_pc) == 0:
            continue
        ax.scatter(sub_pc[:, 0], sub_pc[:, 1], c=[cluster_colors[i]], s=8, alpha=0.8, label=f'Det {i}')

        
        min_x, max_x = sub_pc[:, 0].min(), sub_pc[:, 0].max()
        min_y, max_y = sub_pc[:, 1].min(), sub_pc[:, 1].max()
        rect_x = [min_x, max_x, max_x, min_x, min_x]
        rect_y = [min_y, min_y, max_y, max_y, min_y]
        ax.plot(rect_x, rect_y, color=cluster_colors[i], linewidth=2, linestyle='--')

    
    for lbl in car_labels:
        corners = get_3d_box_corners_cam(lbl)
        corners_lidar = cam_to_lidar(corners, Tr)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
        ]
        for edge in edges:
            p1, p2 = corners_lidar[edge[0]], corners_lidar[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)

    ax.scatter([0], [0], c='black', s=20, marker='x', label='LiDAR origin')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'BEV - Scene {scene_id}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    out_path = os.path.join(DET_DIR, scene_id + "_bev.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("    -> BEV Plot: " + out_path)


def compute_average_precision_at_iou(gt_boxes, pred_boxes, iou_threshold):
    """Compute AP for a single IoU threshold across all scenes."""
    total_gts = len(gt_boxes)
    if total_gts == 0:
        return 0.0

    preds = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
    matched_gt = set()
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)

    for idx, pred in enumerate(preds):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if gt['scene_id'] != pred['scene_id']:
                continue
            iou = compute_iou_2d(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[idx] = 1.0
            matched_gt.add(best_gt_idx)
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls = cum_tp / (total_gts + 1e-6)

    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def save_ap_curve(thresholds, ap_values):
    """Save the AP vs IoU threshold curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, ap_values, marker='o', linestyle='-', color='navy')
    ax.set_xlabel('IoU threshold', fontsize=12)
    ax.set_ylabel('Average Precision', fontsize=12)
    ax.set_title('AP vs IoU Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(thresholds)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.fill_between(thresholds, ap_values, alpha=0.15, color='navy')
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(AP_CURVE_PATH, dpi=120)
    plt.close(fig)
    print(f"    -> AP curve saved: {AP_CURVE_PATH}")



def process_scene(scene_id, model):
    """Process one scene: YOLO detection, LiDAR projection, association."""
    global ALL_SCORES, TOTAL_GT_CARS

    scene_num = scene_id.lstrip('0') or '0'
    image_path = os.path.join(IMAGE_DIR, scene_id + ".png")
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGE_DIR, scene_id + ".jpg")
    lidar_path = os.path.join(LIDAR_DIR, scene_id + ".bin")
    calib_path = os.path.join(CALIB_DIR, scene_id + ".txt")
    label_path = os.path.join(LABEL_DIR, scene_id + ".txt")

    if not os.path.exists(image_path):
        print(f"  [SKIP] Image not found: {image_path}")
        return

    
    image = load_image(image_path)
    points = load_lidar(lidar_path)
    P2, Tr, R0 = load_calibration(calib_path)
    car_labels = load_labels(label_path)

    
    yolo_masks, yolo_boxes, yolo_scores = detect_cars_yolo(image, model)

    
    pixels, valid_idx, depths = get_valid_projections(points, P2, Tr, R0, image.shape)

    
    sub_pcs = assign_points_to_masks(pixels, valid_idx, points, yolo_masks)

    
    for box, score in zip(yolo_boxes, yolo_scores):
        ALL_PREDICTIONS.append({
            'scene_id': scene_id,
            'box': box,
            'score': score,
        })
    for lbl in car_labels:
        ALL_GT_BOXES.append({
            'scene_id': scene_id,
            'box': lbl['bbox_2d'],
        })

    
    filtered_pcs = []
    for sub_pc in sub_pcs:
        if len(sub_pc) == 0:
            filtered_pcs.append(sub_pc)
        else:
            precision, filt = evaluate_sub_pointcloud(sub_pc, car_labels, Tr, R0)
            filtered_pcs.append(filt)

    
    save_annotated_image(image, yolo_boxes, yolo_scores, car_labels, scene_id)
    if len(pixels) > 0:
        save_projection_images(image, pixels, depths, sub_pcs, P2, Tr, R0, scene_id)
    save_bev_plot(points, sub_pcs, filtered_pcs, car_labels, Tr, scene_id)

    TOTAL_GT_CARS += len(car_labels)
    ALL_SCORES.extend(yolo_scores)




if __name__ == "__main__":
    startup_checks()

    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    print(f"Model loaded: {YOLO_MODEL_PATH}\n")

    scene_ids = get_scene_ids()
    if not scene_ids:
        print("ERROR: No scenes found in IMAGE_DIR")
        exit(1)

    print(f"Processing {len(scene_ids)} scenes...\n")
    for i, scene_id in enumerate(scene_ids, 1):
        print(f"[{i}/{len(scene_ids)}] Scene {scene_id}")
        try:
            process_scene(scene_id, model)
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    if len(ALL_GT_BOXES) > 0:
        ap_values = [
            compute_average_precision_at_iou(ALL_GT_BOXES, ALL_PREDICTIONS, thr)
            for thr in AP_THRESHOLDS
        ]
        save_ap_curve(AP_THRESHOLDS, ap_values)
        print("AP vs IoU threshold:")
        for thr, ap in zip(AP_THRESHOLDS, ap_values):
            print(f"  IoU {thr:.2f}: AP={ap:.3f}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if ALL_SCORES:
        print(f"Total detections: {len(ALL_SCORES)}")
        print(f"Mean confidence: {np.mean(ALL_SCORES):.3f}")
    print(f"Total GT cars: {TOTAL_GT_CARS}")
    print("\nOutput saved to:")
    print(f"  {OUT_DIR}")
    print(f"  {PROJ_DIR}")
    print(f"  {DET_DIR}")
