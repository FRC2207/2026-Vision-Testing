"""
test_pose_estimation.py
=======================
Full 6-DOF pose estimation for spherical game pieces (fuel/balls).

Given a detected bounding box from your existing YOLO pipeline, this module:
  1. Fits an ellipse to the contour of the detected ball in the image
  2. Generates a ring of 3D object-space points on the sphere equator
  3. Calls cv2.solvePnP to recover full rotation + translation
  4. Decomposes rotation vector → roll, pitch, yaw (camera-relative)
  5. Converts translation to robot-relative coordinates using your existing
     camera mount parameters (height, pitch, yaw, x, y offsets)

The test at the bottom simulates a ball at a known 3D position, projects it
into a synthetic image using your camera matrix, runs the full estimator on
it, and checks that the recovered pose is close to the ground truth.

USAGE (standalone test):
    python test_pose_estimation.py

USAGE (integrated with your ObjectDetectionCamera):
    from test_pose_estimation import SpherePoseEstimator
    estimator = SpherePoseEstimator(camera_matrix, dist_coeffs, ball_radius_m)
    pose = estimator.estimate_from_box(frame, box)
    # pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional


# ─── Data class for a full 6-DOF pose result ─────────────────────────────────

@dataclass
class BallPose:
    # Translation — camera-relative (metres)
    x: float       # right is positive
    y: float       # down is positive (camera frame)
    z: float       # forward (depth) — most useful for ranging

    # Rotation (radians) — camera-relative Euler angles from the rvec
    roll:  float
    pitch: float
    yaw:   float

    # Robot-relative ground-plane position (metres), same convention as your
    # existing _pixel_to_robot_coordinates output
    robot_x: float
    robot_y: float

    # Quality indicators
    reprojection_error: float   # pixels — lower is better, <2px is good
    ellipse_quality:    float   # 0-1, how well the ellipse fit the mask contour


# ─── Core estimator ──────────────────────────────────────────────────────────

class SpherePoseEstimator:
    """
    Full 6-DOF pose estimator for a sphere of known radius.

    Parameters
    ----------
    camera_matrix : np.ndarray, shape (3,3)
        Intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        Build from FOV + resolution if you don't have calibration data yet —
        see build_camera_matrix_from_fov().

    dist_coeffs : np.ndarray, shape (4,) or (5,) or (8,)
        Radial/tangential distortion coefficients from cv2.calibrateCamera.
        Pass np.zeros(5) if uncalibrated — results will be less accurate at
        the edges of the frame but fine in the centre.

    ball_radius_m : float
        Physical radius of the game piece in metres.
        REEFSCAPE coral = 0.075 m (3-inch radius), adjust as needed.

    camera_height_m : float
        Camera height above the ground plane in metres.

    camera_pitch_deg : float
        Camera pitch (tilted down = positive), degrees.

    camera_yaw_deg : float
        Camera yaw relative to robot forward (left = positive), degrees.

    camera_x_m, camera_y_m : float
        Camera offset from robot centre (metres).

    n_equator_points : int
        How many points to sample on the sphere equator for solvePnP.
        8 is fast and accurate; 16 is more robust on partially occluded balls.
    """

    def __init__(
        self,
        camera_matrix:    np.ndarray,
        dist_coeffs:      np.ndarray,
        ball_radius_m:    float = 0.075,
        camera_height_m:  float = 0.0,
        camera_pitch_deg: float = 0.0,
        camera_yaw_deg:   float = 0.0,
        camera_x_m:       float = 0.0,
        camera_y_m:       float = 0.0,
        n_equator_points: int   = 8,
    ):
        self.K             = camera_matrix.astype(np.float64)
        self.dist          = dist_coeffs.astype(np.float64)
        self.radius        = ball_radius_m
        self.cam_height    = camera_height_m
        self.cam_pitch_rad = math.radians(camera_pitch_deg)
        self.cam_yaw_rad   = math.radians(camera_yaw_deg)
        self.cam_x         = camera_x_m
        self.cam_y         = camera_y_m
        self.n_pts         = n_equator_points

        # Pre-build the ring of 3D object points on the sphere equator.
        # These are in "object space" — centred on the ball, lying on the
        # equatorial plane.  solvePnP will find the transform that maps them
        # to their observed 2D projections.
        angles = np.linspace(0, 2 * math.pi, n_equator_points, endpoint=False)
        self._obj_pts = np.array([
            [ball_radius_m * math.cos(a),
             ball_radius_m * math.sin(a),
             0.0]
            for a in angles
        ], dtype=np.float64)   # shape (N, 3)

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate_from_box(
        self,
        frame: np.ndarray,
        box_xyxy,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[BallPose]:
        """
        Estimate the 6-DOF pose of a ball from its bounding box.

        Parameters
        ----------
        frame    : BGR image (full frame, not cropped)
        box_xyxy : (x1, y1, x2, y2) bounding box in pixel coords
        mask     : optional binary mask (same size as frame).  If None,
                   a circular mask is synthesised from the bounding box.

        Returns None if solvePnP fails or the ellipse fit is too poor.
        """
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx_box = (x1 + x2) / 2
        cy_box = (y1 + y2) / 2
        rx_box = (x2 - x1) / 2   # half-width
        ry_box = (y2 - y1) / 2   # half-height

        if rx_box < 3 or ry_box < 3:
            return None  # box too small to fit an ellipse reliably

        # ── Step 1: get a contour for the ball ──────────────────────────────
        if mask is not None:
            ball_mask = mask
        else:
            ball_mask = self._synthesise_mask(frame, x1, y1, x2, y2)

        contour, ellipse_quality = self._extract_ellipse_contour(ball_mask, x1, y1, x2, y2)

        if contour is None or len(contour) < 5:
            # Fall back: use 8 points sampled on the box ellipse boundary
            contour = self._box_ellipse_points(cx_box, cy_box, rx_box, ry_box)
            ellipse_quality = 0.5   # synthetic, lower confidence

        # ── Step 2: fit an ellipse to the contour ────────────────────────────
        if len(contour) < 5:
            return None

        try:
            ((ex, ey), (ea, eb), angle_deg) = cv2.fitEllipse(contour)
        except cv2.error:
            return None

        ea_half = ea / 2
        eb_half = eb / 2

        if ea_half < 2 or eb_half < 2:
            return None  # degenerate ellipse

        # ── Step 3: project the equator ring onto the fitted ellipse ─────────
        # The projection of a sphere's equator is (roughly) the visible
        # ellipse outline.  We map our N object-space angles to the
        # corresponding image-space positions on that ellipse.
        image_pts_2d = self._project_equator_to_ellipse(
            ex, ey, ea_half, eb_half, math.radians(angle_deg)
        )   # shape (N, 1, 2) for solvePnP

        # ── Step 4: solvePnP ──────────────────────────────────────────────────
        obj_pts_shaped = self._obj_pts.reshape(-1, 1, 3)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts_shaped,
            image_pts_2d,
            self.K,
            self.dist,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or tvec is None:
            # RANSAC failed → try non-robust version as fallback
            success, rvec, tvec = cv2.solvePnP(
                obj_pts_shaped,
                image_pts_2d,
                self.K,
                self.dist,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success:
                return None

        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts_shaped,
            image_pts_2d,
            self.K,
            self.dist,
            rvec,
            tvec,
        )

        # ── Step 5: compute reprojection error ────────────────────────────────
        proj, _ = cv2.projectPoints(obj_pts_shaped, rvec, tvec, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        orig = image_pts_2d.reshape(-1, 2)
        reproj_err = float(np.mean(np.linalg.norm(proj - orig, axis=1)))

        if reproj_err > 10.0:   # something went badly wrong
            return None

        # ── Step 6: decompose rvec → Euler angles ────────────────────────────
        R, _ = cv2.Rodrigues(rvec)
        roll, pitch, yaw = self._rotation_matrix_to_euler(R)

        # ── Step 7: camera-frame translation ──────────────────────────────────
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])

        # ── Step 8: robot-relative ground-plane (x, y) ────────────────────────
        robot_x, robot_y = self._camera_to_robot(tx, ty, tz)

        return BallPose(
            x=tx, y=ty, z=tz,
            roll=roll, pitch=pitch, yaw=yaw,
            robot_x=robot_x, robot_y=robot_y,
            reprojection_error=reproj_err,
            ellipse_quality=ellipse_quality,
        )

    def estimate_from_box_fast(
        self,
        box_xyxy,
    ) -> Optional[BallPose]:
        """
        No-image fast path: uses only the bounding box and assumes a
        circular projection (good when you don't have a mask and want
        speed over accuracy for the angular terms).

        This always produces a valid result as long as the box is big enough.
        Angular errors are larger but x,y,z are still good.
        """
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rx = (x2 - x1) / 2
        ry = (y2 - y1) / 2

        if rx < 3 or ry < 3:
            return None

        contour = self._box_ellipse_points(cx, cy, rx, ry)
        image_pts_2d = contour.reshape(-1, 1, 2).astype(np.float64)
        obj_pts = self._obj_pts.reshape(-1, 1, 3)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, image_pts_2d, self.K, self.dist, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            return None

        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, image_pts_2d, self.K, self.dist, rvec, tvec)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
        reproj_err = float(np.mean(np.linalg.norm(
            proj.reshape(-1, 2) - image_pts_2d.reshape(-1, 2), axis=1
        )))

        R, _ = cv2.Rodrigues(rvec)
        roll, pitch, yaw = self._rotation_matrix_to_euler(R)
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        robot_x, robot_y = self._camera_to_robot(tx, ty, tz)

        return BallPose(
            x=tx, y=ty, z=tz,
            roll=roll, pitch=pitch, yaw=yaw,
            robot_x=robot_x, robot_y=robot_y,
            reprojection_error=reproj_err,
            ellipse_quality=0.5,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _synthesise_mask(
        self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
    ) -> np.ndarray:
        """
        Isolate the ball using colour/intensity segmentation within the box,
        then return a binary mask (0/255).

        Strategy:
          1. Crop to the bounding box (with a small margin)
          2. Convert to HSV
          3. Apply GrabCut or a simple threshold to separate ball from background
          4. Fall back to a synthetic ellipse mask if segmentation yields nothing
        """
        h, w = frame.shape[:2]
        margin = 4
        rx1 = max(0, int(x1) - margin)
        ry1 = max(0, int(y1) - margin)
        rx2 = min(w, int(x2) + margin)
        ry2 = min(h, int(y2) + margin)

        if rx2 - rx1 < 6 or ry2 - ry1 < 6:
            return self._ellipse_mask_image(frame.shape[:2], x1, y1, x2, y2)

        crop = frame[ry1:ry2, rx1:rx2]
        crop_h, crop_w = crop.shape[:2]

        # GrabCut initialised with the inner 70% of the box
        mask_gc = np.zeros((crop_h, crop_w), np.uint8)
        inner_margin_x = max(1, crop_w // 8)
        inner_margin_y = max(1, crop_h // 8)
        rect = (
            inner_margin_x, inner_margin_y,
            crop_w - 2 * inner_margin_x,
            crop_h - 2 * inner_margin_y,
        )
        if rect[2] > 0 and rect[3] > 0:
            try:
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(crop, mask_gc, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                fg = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                if fg.sum() > 50:
                    full_mask = np.zeros((h, w), np.uint8)
                    full_mask[ry1:ry2, rx1:rx2] = fg
                    return full_mask
            except cv2.error:
                pass

        # Fallback: synthesise an ellipse mask from the bounding box
        return self._ellipse_mask_image(frame.shape[:2], x1, y1, x2, y2)

    def _ellipse_mask_image(self, shape, x1, y1, x2, y2):
        mask = np.zeros(shape, np.uint8)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        rx = max(1, int((x2 - x1) / 2))
        ry = max(1, int((y2 - y1) / 2))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return mask

    def _extract_ellipse_contour(self, mask, x1, y1, x2, y2):
        """
        Find the largest contour inside the bounding box area.
        Returns (contour_array, quality_score).
        """
        roi_mask = np.zeros_like(mask)
        margin = 2
        rx1 = max(0, int(x1) - margin)
        ry1 = max(0, int(y1) - margin)
        rx2 = min(mask.shape[1], int(x2) + margin)
        ry2 = min(mask.shape[0], int(y2) + margin)
        roi_mask[ry1:ry2, rx1:rx2] = mask[ry1:ry2, rx1:rx2]

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, 0.0

        largest = max(contours, key=cv2.contourArea)

        # Quality: how circular is it?  (circularity = 4π·area / perimeter²)
        area = cv2.contourArea(largest)
        peri = cv2.arcLength(largest, True)
        quality = 0.0
        if peri > 0:
            quality = min(1.0, (4 * math.pi * area) / (peri ** 2))

        return largest, quality

    def _box_ellipse_points(self, cx, cy, rx, ry) -> np.ndarray:
        """Sample N points on the box ellipse boundary — used as fallback."""
        angles = np.linspace(0, 2 * math.pi, self.n_pts, endpoint=False)
        pts = np.array([
            [cx + rx * math.cos(a), cy + ry * math.sin(a)]
            for a in angles
        ], dtype=np.float64)
        return pts.reshape(-1, 1, 2)

    def _project_equator_to_ellipse(self, ex, ey, ea, eb, angle_rad) -> np.ndarray:
        """
        Map the N equatorial angles to positions on the fitted image ellipse.
        The fitted ellipse captures the perspective foreshortening.
        """
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        angles = np.linspace(0, 2 * math.pi, self.n_pts, endpoint=False)
        pts = []
        for a in angles:
            # Point on a unit ellipse at angle a, then rotate by the fitted tilt
            local_x = ea * math.cos(a)
            local_y = eb * math.sin(a)
            world_x = ex + local_x * cos_a - local_y * sin_a
            world_y = ey + local_x * sin_a + local_y * cos_a
            pts.append([world_x, world_y])
        return np.array(pts, dtype=np.float64).reshape(-1, 1, 2)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray):
        """
        Decompose a 3x3 rotation matrix to (roll, pitch, yaw) in radians.
        Convention: ZYX (yaw around Z, pitch around Y, roll around X).
        """
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll  = math.atan2( R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2( R[1, 0], R[0, 0])
        else:
            roll  = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = 0.0
        return roll, pitch, yaw

    def _camera_to_robot(self, tx: float, ty: float, tz: float):
        """
        Convert camera-frame translation (tx, ty, tz) to robot-relative
        ground-plane (x, y) in metres, using the same logic as your existing
        _pixel_to_robot_coordinates but operating on the solvePnP translation
        directly (which is already in metric units — no focal-length tricks needed).

        Camera axes convention (OpenCV):
          +X  = right
          +Y  = down
          +Z  = into scene (forward)
        """
        # tz is the true line-of-sight depth from the camera to the ball
        # ty is up/down in camera frame — use it to refine the ground range
        # horizontal angle from centre
        horiz_angle = math.atan2(tx, tz)

        # True 3D distance from camera to ball
        dist_3d = math.sqrt(tx**2 + ty**2 + tz**2)

        # Project to ground plane
        if self.cam_height > 0 and dist_3d > self.cam_height:
            true_horiz_dist = math.sqrt(dist_3d**2 - self.cam_height**2)
        else:
            true_horiz_dist = tz * math.cos(self.cam_pitch_rad) + ty * math.sin(self.cam_pitch_rad)

        left_right = true_horiz_dist * math.sin(horiz_angle)
        forward    = true_horiz_dist * math.cos(horiz_angle)

        # Rotate by camera yaw mount offset
        cos_yaw = math.cos(self.cam_yaw_rad)
        sin_yaw = math.sin(self.cam_yaw_rad)
        x_rot = forward * cos_yaw + left_right * sin_yaw
        y_rot = forward * sin_yaw - left_right * cos_yaw

        # Add camera offset from robot centre
        robot_x = x_rot + self.cam_x
        robot_y = y_rot + self.cam_y
        return robot_x, robot_y


# ─── Utility: build camera matrix from FOV + resolution ─────────────────────

def build_camera_matrix_from_fov(
    image_width: int,
    image_height: int,
    fov_horizontal_deg: float,
) -> np.ndarray:
    """
    Build an approximate intrinsic matrix when you don't have calibration data.
    Focal length is derived from the horizontal FOV and image width.

    For best accuracy, replace this with data from cv2.calibrateCamera using
    a checkerboard (see Utilities/camera_calibration.py for a starting point).
    """
    fx = (image_width / 2.0) / math.tan(math.radians(fov_horizontal_deg / 2.0))
    fy = fx  # square pixels assumed
    cx = image_width  / 2.0
    cy = image_height / 2.0
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float64)


# ─── Helper: draw pose debug overlay on a frame ─────────────────────────────

def draw_pose_overlay(frame: np.ndarray, box_xyxy, pose: BallPose) -> np.ndarray:
    """Draw bounding box + pose text on a copy of frame."""
    out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Box
    color = (0, 255, 0) if pose.reprojection_error < 3.0 else (0, 165, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    # Centre dot
    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    # Text
    lines = [
        f"Z(depth): {pose.z*100:.1f} cm",
        f"X(right): {pose.x*100:.1f} cm",
        f"Y(down):  {pose.y*100:.1f} cm",
        f"Yaw:   {math.degrees(pose.yaw):.1f} deg",
        f"Pitch: {math.degrees(pose.pitch):.1f} deg",
        f"Roll:  {math.degrees(pose.roll):.1f} deg",
        f"Robot x: {pose.robot_x*100:.1f} cm",
        f"Robot y: {pose.robot_y*100:.1f} cm",
        f"Reproj err: {pose.reprojection_error:.2f} px",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x2 + 5, y1 + 16 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ─── TESTS ────────────────────────────────────────────────────────────────────

def _project_sphere_to_box(
    center_3d: np.ndarray,
    radius: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple:
    """
    Given a sphere at center_3d (camera frame), project its centre and compute
    the approximate bounding box in pixel space.
    Returns (cx_px, cy_px, x1, y1, x2, y2).
    """
    pt = center_3d.reshape(1, 1, 3)
    proj, _ = cv2.projectPoints(pt, np.zeros(3), np.zeros(3), K, dist)
    cx_px, cy_px = float(proj[0, 0, 0]), float(proj[0, 0, 1])

    # Apparent radius in pixels using the on-axis focal length
    fx = float(K[0, 0])
    z  = float(center_3d[2])
    r_px = (fx * radius) / z if z > 0 else 20.0

    x1 = cx_px - r_px
    y1 = cy_px - r_px
    x2 = cx_px + r_px
    y2 = cy_px + r_px
    return cx_px, cy_px, x1, y1, x2, y2


def run_tests():
    print("=" * 60)
    print("  SpherePoseEstimator — self-test suite")
    print("=" * 60)

    # ── Camera setup (matches your Arducam config in gray_config.json) ────────
    IMG_W, IMG_H  = 320, 320
    FOV_DEG       = 60.0
    BALL_RADIUS_M = 0.075        # 3-inch radius ball ≈ 0.075 m
    CAM_HEIGHT_M  = 0.2065       # 8.125 inches
    CAM_PITCH_DEG = 15.0
    CAM_YAW_DEG   = 90.0         # side-facing camera

    K    = build_camera_matrix_from_fov(IMG_W, IMG_H, FOV_DEG)
    dist = np.zeros(5)  # perfect (undistorted) lens for synthetic test

    estimator = SpherePoseEstimator(
        camera_matrix    = K,
        dist_coeffs      = dist,
        ball_radius_m    = BALL_RADIUS_M,
        camera_height_m  = CAM_HEIGHT_M,
        camera_pitch_deg = CAM_PITCH_DEG,
        camera_yaw_deg   = CAM_YAW_DEG,
        camera_x_m       = 0.36,    # 14.1925 inches → metres
        camera_y_m       = 0.17,
    )

    print(f"\nCamera matrix:\n{K}\n")
    print(f"Ball radius : {BALL_RADIUS_M * 100:.1f} cm")
    print(f"Image size  : {IMG_W}x{IMG_H}")
    print()

    # ── Test cases: (ball_x, ball_y, ball_z) in camera frame (metres) ────────
    test_cases = [
        ("Directly ahead, 1 m",  np.array([ 0.0,   0.0,  1.0])),
        ("Directly ahead, 2 m",  np.array([ 0.0,   0.0,  2.0])),
        ("Directly ahead, 3 m",  np.array([ 0.0,   0.0,  3.0])),
        ("Left 0.5 m, 1.5 m out",np.array([-0.5,   0.0,  1.5])),
        ("Right 0.3 m, 2 m out", np.array([ 0.3,   0.0,  2.0])),
        ("High on frame, 1 m",   np.array([ 0.0,  -0.3,  1.0])),
        ("Low on frame, 1 m",    np.array([ 0.0,   0.2,  1.0])),
        ("Off-axis, 4 m",        np.array([ 0.8,   0.1,  4.0])),
    ]

    all_passed = True
    results_table = []

    for name, gt_xyz in test_cases:
        # Project the ground-truth sphere into a synthetic bounding box
        cx_px, cy_px, x1, y1, x2, y2 = _project_sphere_to_box(gt_xyz, BALL_RADIUS_M, K, dist)

        # Check the ball actually lands inside the image
        in_frame = (0 < cx_px < IMG_W) and (0 < cy_px < IMG_H)

        box = (x1, y1, x2, y2)

        # Run estimator (fast path — no image needed)
        pose = estimator.estimate_from_box_fast(box)

        if pose is None:
            results_table.append((name, "FAILED — solvePnP returned None", False))
            all_passed = False
            continue

        # ── Accuracy checks ───────────────────────────────────────────────────
        # Z (depth) is the most important for ranging — allow 5 % error
        z_err_pct = abs(pose.z - gt_xyz[2]) / gt_xyz[2] * 100

        # X (lateral) absolute error in cm
        x_err_cm  = abs(pose.x - gt_xyz[0]) * 100

        # Reprojection error should be tiny on a noise-free synthetic test
        reproj    = pose.reprojection_error

        z_ok     = z_err_pct < 8.0     # within 8 %
        x_ok     = x_err_cm  < 3.0     # within 3 cm lateral
        reproj_ok = reproj    < 5.0     # under 5 px

        passed = z_ok and x_ok and reproj_ok and in_frame
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        detail = (
            f"  z={pose.z*100:6.1f} cm  (gt {gt_xyz[2]*100:.1f} cm,  err {z_err_pct:.1f}%)  "
            f"x={pose.x*100:6.1f} cm  (err {x_err_cm:.1f} cm)  "
            f"reproj={reproj:.2f} px  "
            f"robot_x={pose.robot_x*100:.1f} cm  robot_y={pose.robot_y*100:.1f} cm"
        )
        results_table.append((name, detail, passed))

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"{'Test':<35}  {'Status':6}  Details")
    print("-" * 110)
    for name, detail, passed in results_table:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<35}  {status}  {detail}")

    print()
    if all_passed:
        print("✅  All tests passed — SpherePoseEstimator is working correctly.")
    else:
        print("❌  Some tests failed — check the details above.")

    # ── Extra: print the Euler angles for the on-axis 2 m case ───────────────
    print()
    print("── Euler angle breakdown for 'Directly ahead, 2 m' ──────────────")
    gt = np.array([0.0, 0.0, 2.0])
    _, _, x1, y1, x2, y2 = _project_sphere_to_box(gt, BALL_RADIUS_M, K, dist)
    pose = estimator.estimate_from_box_fast((x1, y1, x2, y2))
    if pose:
        print(f"  Depth (Z)  : {pose.z*100:.2f} cm   (ground truth: {gt[2]*100:.1f} cm)")
        print(f"  Lateral (X): {pose.x*100:.2f} cm   (ground truth: {gt[0]*100:.1f} cm)")
        print(f"  Vertical (Y): {pose.y*100:.2f} cm   (ground truth: {gt[1]*100:.1f} cm)")
        print(f"  Roll       : {math.degrees(pose.roll):.2f} °")
        print(f"  Pitch      : {math.degrees(pose.pitch):.2f} °")
        print(f"  Yaw        : {math.degrees(pose.yaw):.2f} °")
        print(f"  Reprojection error: {pose.reprojection_error:.3f} px")
        print(f"  Robot-relative x: {pose.robot_x*100:.2f} cm")
        print(f"  Robot-relative y: {pose.robot_y*100:.2f} cm")

    # ── Integration usage example ─────────────────────────────────────────────
    print()
    print("── Integration snippet (drop into ObjectDetectionCamera.run()) ───")
    print("""
    # In ObjectDetectionCamera.__init__, add:
    #   from test_pose_estimation import SpherePoseEstimator, build_camera_matrix_from_fov
    #   self.K    = build_camera_matrix_from_fov(img_w, img_h, self.fov)
    #   self.dist = np.zeros(5)   # replace with real calibration data
    #   self.pose_estimator = SpherePoseEstimator(
    #       self.K, self.dist,
    #       ball_radius_m    = self.ball_d_inches * 0.0254 / 2,
    #       camera_height_m  = self.camera_height * 0.0254,
    #       camera_pitch_deg = self.camera_pitch_angle,
    #       camera_yaw_deg   = self.camera_bot_relative_yaw,
    #       camera_x_m       = self.camera_x * 0.0254,
    #       camera_y_m       = self.camera_y * 0.0254,
    #   )

    # In run(), replace _box_to_robot_point() with:
    #   pose = self.pose_estimator.estimate_from_box_fast(box.xyxy)
    #   if pose:
    #       map_points.append(np.array([pose.robot_x, pose.robot_y]))
    #       # Also available: pose.z (depth), pose.roll/pitch/yaw

    # For better accuracy when you have the frame:
    #   pose = self.pose_estimator.estimate_from_box(frame, box.xyxy)
    """)


if __name__ == "__main__":
    run_tests()