
import numpy as np
import cv2
import os


def load_image(image_path):
    """
    Load camera image and convert BGR to RGB.
    Tries both .png and .jpg extensions automatically.
    Returns: numpy array [H, W, 3]
    """
    img = cv2.imread(image_path)


    if img is None:
        if image_path.endswith('.png'):
            alt = image_path.replace('.png', '.jpg')
        else:
            alt = image_path.replace('.jpg', '.png')
        img = cv2.imread(alt)

    if img is None:
        raise FileNotFoundError(
            f"\n[ERROR] Image not found:\n  {image_path}\n"
            f"Check BASE_PATH in main.py\n"
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_lidar(bin_path):
    """
    Load LiDAR binary file.
    Each point = [x, y, z, intensity]
    Returns: numpy array [N, 4]
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"LiDAR file not found: {bin_path}")
    pts = np.fromfile(bin_path, dtype=np.float32)
    pts = pts.reshape(-1, 4)
    return pts


def load_calibration(calib_path):
    """
    Parse KITTI calibration text file.
    Returns:
        P2     : 3x4  camera projection matrix
        Tr_4x4 : 4x4  lidar-to-camera transform
        R0_4x4 : 4x4  rectification matrix
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calib file not found: {calib_path}")

    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            vals = val.strip().split()
            if vals:
                calib[key.strip()] = np.array([float(x) for x in vals])


    P2 = calib['P2'].reshape(3, 4)

    
    Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
    Tr_4x4 = np.vstack([Tr, [0, 0, 0, 1]])


    R0_4x4 = np.eye(4)
    if 'R0_rect' in calib:
        R0_4x4[:3, :3] = calib['R0_rect'].reshape(3, 3)

    return P2, Tr_4x4, R0_4x4


def load_labels(label_path):
    """
    Load KITTI label file.
    Returns list of dicts (one per object).
    Returns empty list if file does not exist.
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            obj = {
                'class'    : parts[0],
                'truncated': float(parts[1]),
                'occluded' : int(parts[2]),
                'alpha'    : float(parts[3]),
                'bbox_2d'  : [float(x) for x in parts[4:8]],  # l,t,r,b
                'height'   : float(parts[8]),
                'width'    : float(parts[9]),
                'length'   : float(parts[10]),
                'x'        : float(parts[11]),
                'y'        : float(parts[12]),
                'z'        : float(parts[13]),
                'rotation' : float(parts[14]),
            }
            labels.append(obj)
    return labels



def project_lidar_to_image(points, P2, Tr, R0):
    """
    Project 3D LiDAR points onto 2D image plane.

    Full pipeline:
        LiDAR [x,y,z,1]
          --x Tr--> camera 3D
          --x R0--> rectified camera 3D
          --x P2--> image pixel [u, v]

    Returns:
        pixels        : [M, 2]  float32 (u, v) pixel coordinates
        valid_indices : [M]     indices into the original points array
        depths        : [M]     z-depth of each valid point
    """
    N    = len(points)
    ones = np.ones((N, 1))
    pts_hom = np.hstack([points[:, :3], ones])   

    
    cam = (Tr @ pts_hom.T).T                     

    
    cam_rect = (R0 @ cam.T).T                    

    front         = cam_rect[:, 2] > 0
    cam_rect      = cam_rect[front]
    valid_indices = np.where(front)[0]
    depths        = cam_rect[:, 2].copy()

    img_hom = (P2 @ cam_rect.T).T               
    img_hom[:, 0] /= img_hom[:, 2]              
    img_hom[:, 1] /= img_hom[:, 2]              

    pixels = img_hom[:, :2].astype(np.float32)

    return pixels, valid_indices, depths


def filter_to_image(pixels, valid_indices, depths, img_shape):
    """
    Remove points that project outside the image boundary.
    Returns filtered (pixels, valid_indices, depths).
    """
    H, W = img_shape[:2]
    mask = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < W) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
    )
    return pixels[mask], valid_indices[mask], depths[mask]


def depth_to_color_rainbow(depths):
    """
    Map depth values to rainbow BGR colors.
    Red = close,  Blue = far  (matches Image 2 in the project).
    Returns: [N, 3] uint8 BGR array
    """
    d_min = depths.min()
    d_max = depths.max()
    norm  = (depths - d_min) / (d_max - d_min + 1e-6)   # 0 .. 1

    # Hue: 240 (blue) for far, 0 (red) for close
    hue = ((1.0 - norm) * 240).astype(np.uint8)

    hsv = np.zeros((len(depths), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hue
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr[:, 0, :]   # [N, 3]




def euclidean_distance(x, y, z):
    """Simple 3D Euclidean distance."""
    return float(np.sqrt(x**2 + y**2 + z**2))


def distance_from_pointcloud(sub_pc):
    """
    Minimum Euclidean distance of all points in sub-pointcloud.
    Minimum is used because it represents the closest surface of the car.
    Returns None if sub_pc is empty.
    """
    if sub_pc is None or len(sub_pc) == 0:
        return None
    dists = np.sqrt(sub_pc[:, 0]**2 + sub_pc[:, 1]**2 + sub_pc[:, 2]**2)
    return float(np.min(dists))


def gt_distance(label):
    """Distance to GT object centre."""
    return euclidean_distance(label['x'], label['y'], label['z'])




def get_3d_box_corners_cam(label):
    """
    Compute 8 corners of a 3D GT bounding box in camera coords.
    KITTI stores: (x,y,z) = bottom-centre of box in camera frame.
    """
    h, w, l = label['height'], label['width'], label['length']
    ry       = label['rotation']
    x_c = l / 2
    y_c = 0.0       
    z_c = w / 2

    corners = np.array([
        [ x_c, 0,  z_c],
        [-x_c, 0,  z_c],
        [-x_c, 0, -z_c],
        [ x_c, 0, -z_c],
        [ x_c,-h,  z_c],
        [-x_c,-h,  z_c],
        [-x_c,-h, -z_c],
        [ x_c,-h, -z_c],
    ])

   
    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,           0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    corners = (R @ corners.T).T

   
    corners[:, 0] += label['x']
    corners[:, 1] += label['y']
    corners[:, 2] += label['z']

    return corners   # [8, 3]


def cam_to_lidar(corners_cam, Tr):
    """
    Convert camera-frame 3D corners to LiDAR frame.
    Tr is the 4x4 lidar→cam matrix; we invert it.
    """
    Tr_inv  = np.linalg.inv(Tr)
    N       = corners_cam.shape[0]
    ones    = np.ones((N, 1))
    hom     = np.hstack([corners_cam, ones])   # [N, 4]
    lidar   = (Tr_inv @ hom.T).T              # [N, 4]
    return lidar[:, :3]


def is_point_in_gt_box(pt_cam, label):
    """
    Check whether a camera-frame 3D point is inside the GT box.
    Returns True / False.
    """
    ry  = label['rotation']
    cx, cy, cz = label['x'], label['y'], label['z']
    h, w, l    = label['height'], label['width'], label['length']

    dx = pt_cam[0] - cx
    dy = pt_cam[1] - cy
    dz = pt_cam[2] - cz

    cos_ry, sin_ry = np.cos(-ry), np.sin(-ry)
    lx =  cos_ry * dx + sin_ry * dz
    ly =  dy
    lz = -sin_ry * dx + cos_ry * dz

    return (
        abs(lx) <= l / 2 + 0.1 and
        abs(ly) <= h / 2 + 0.1 and
        abs(lz) <= w / 2 + 0.1
    )


def evaluate_sub_pointcloud(sub_pc, car_labels, Tr, R0):
    """
    For a sub-pointcloud (LiDAR frame), transform to camera frame
    and count how many points fall inside any GT car box.

    Returns:
        precision          : float 0..1
        filtered_sub_pc    : sub_pc with only inside points
    """
    if sub_pc is None or len(sub_pc) == 0:
        return 0.0, np.empty((0, 4))

    N    = len(sub_pc)
    ones = np.ones((N, 1))
    hom  = np.hstack([sub_pc[:, :3], ones])

    cam      = (Tr  @ hom.T).T
    cam_rect = (R0  @ cam.T).T

    inside = np.zeros(N, dtype=bool)
    for i, pt in enumerate(cam_rect):
        for lbl in car_labels:
            if is_point_in_gt_box(pt[:3], lbl):
                inside[i] = True
                break

    precision = float(inside.sum() / N)
    filtered_sub_pc = sub_pc[inside]
    return precision, filtered_sub_pc


def compute_iou_2d(boxA, boxB):
    """
    Compute 2D Intersection over Union between two boxes.
    Boxes: [left, top, right, bottom]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return float(inter / (areaA + areaB - inter + 1e-6))
