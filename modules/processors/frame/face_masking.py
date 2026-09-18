import cv2
import numpy as np
from modules.typing import Face, Frame
import modules.globals
from modules.gpu_processing import gpu_gaussian_blur, gpu_resize

def apply_color_transfer(source, target):
    """Transfer the target's LAB tone to ``source`` (unmasked compatibility API)."""
    return match_color_lab(source, target)


def match_color_lab(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match colour statistics in LAB using only the supplied face mask.

    The masked path avoids sampling background, hair or clothing when a swap
    is blended.  All channel operations are vectorized NumPy expressions; the
    tiny face ROI is the only region converted back to BGR.
    """
    if source is None or target is None or source.size == 0 or target.size == 0:
        return source
    if source.shape[:2] != target.shape[:2]:
        source = cv2.resize(source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
    src = np.asarray(source, dtype=np.uint8)
    ref = np.asarray(target, dtype=np.uint8)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None:
        valid = np.ones(src_lab.shape[:2], dtype=bool)
    else:
        valid = np.asarray(mask) > 16
        if valid.shape != src_lab.shape[:2] or int(np.count_nonzero(valid)) < 16:
            return src
    if reference_mask is None:
        reference_valid = valid
    else:
        reference_valid = np.asarray(reference_mask) > 16
        if reference_valid.shape != valid.shape or int(np.count_nonzero(reference_valid)) < 16:
            reference_valid = valid
    src_values = src_lab[valid]
    ref_values = ref_lab[reference_valid]
    src_mean = src_values.mean(axis=0, dtype=np.float32)
    ref_mean = ref_values.mean(axis=0, dtype=np.float32)
    src_std = np.maximum(src_values.std(axis=0), 1.0)
    ref_std = ref_values.std(axis=0)
    # Keep luminance transfer conservative so Poisson still handles local
    # gradients instead of receiving a globally over-contrasted crop.
    ref_std = np.maximum(ref_std, 1.0)
    gain = np.clip(ref_std / src_std, 0.85, 1.15)
    offset = np.clip(ref_mean - src_mean, -12.0, 12.0)
    corrected = (src_lab - src_mean) * gain + src_mean + offset
    return cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
        face_outline = landmarks[0:33]
        hull = cv2.convexHull(face_outline)
        # Vectorized hull padding — expand each point outward from center
        center = np.mean(face_outline, axis=0, dtype=np.float32)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        directions = hull_pts - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)  # avoid division by zero
        directions /= norms
        hull_padded = (hull_pts + directions * padding).astype(np.int32)

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges (GPU-accelerated when available)
        mask = gpu_gaussian_blur(mask, (5, 5), 3)

    return mask


def create_hairline_safe_mask(
    face: Face,
    frame: Frame,
    blur_sigma: float | None = None,
    erosion_px: int | None = None,
) -> np.ndarray:
    """Return a tight facial-skin mask that excludes the hairline.

    InsightFace's optional 106-point landmarks are used when available.  The
    face outline is eroded and its upper edge is clipped to a smooth curve
    derived from the eyebrows and chin.  This keeps the real hair and sideburn
    pixels from being replaced, while leaving the forehead skin below the
    natural hairline available to the swap.  The fast five-point live path has
    a separate aligned-space guard in ``face_swapper``; this function provides
    a more precise mask whenever 106 landmarks are present.
    """
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        return mask
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape[0] < 106 or not np.all(np.isfinite(landmarks)):
        return mask

    outline = landmarks[:33].astype(np.int32)
    hull = cv2.convexHull(outline)
    if hull is None or len(hull) < 3:
        return mask
    cv2.fillConvexPoly(mask, hull, 255)

    # Approximate the natural hairline from the brow/chin geometry.  Keeping
    # the curve below the brow by 42% of the brow-to-chin distance avoids the
    # overly aggressive forehead extension used by the old mask.
    brow_points = np.concatenate((landmarks[33:43], landmarks[43:52]))
    brow_y = float(np.min(brow_points[:, 1]))
    chin_y = float(landmarks[16, 1])
    face_width = max(1.0, float(np.ptp(outline[:, 0])))
    guard_y = brow_y - 0.42 * max(0.0, chin_y - brow_y)
    yy = np.arange(mask.shape[0], dtype=np.float32)[:, None]
    xx = np.arange(mask.shape[1], dtype=np.float32)[None, :]
    center_x = float(np.mean(outline[:, 0]))
    half_width = max(1.0, face_width * 0.5)
    curve = guard_y + 0.035 * face_width * np.square((xx - center_x) / half_width)
    sigma = float(modules.globals.mask_blur if blur_sigma is None else blur_sigma)
    feather = max(1.0, sigma * 1.5)
    gate = np.clip((yy - curve + feather) / (2.0 * feather), 0.0, 1.0)
    mask = np.rint(mask.astype(np.float32) * gate).astype(np.uint8)

    erosion = int(modules.globals.mask_erosion if erosion_px is None else erosion_px)
    # Scale the aligned-space setting to the current frame's face width.
    scaled_erosion = max(0, int(round(erosion * face_width / 128.0)))
    if scaled_erosion:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (scaled_erosion * 2 + 1, scaled_erosion * 2 + 1),
        )
        mask = cv2.erode(mask, kernel)
    if sigma > 0:
        kernel_size = max(3, int(np.ceil(sigma * 3)) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return mask

def _feature_landmarks(face: Face):
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        landmarks = getattr(face, "landmark_2d_68", None)
    if landmarks is None:
        return None, None
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        return None, None
    if points.shape[0] >= 106:
        # InsightFace 106 convention. The 68-point fallback below uses the
        # conventional 48:68 mouth and 36:48 eye ranges.
        return points[52:64], (points[33:43], points[87:97])
    if points.shape[0] >= 68:
        return points[48:68], (points[36:42], points[42:48])
    return None, None


def _mask_blur(mask: np.ndarray) -> np.ndarray:
    sigma = float(getattr(modules.globals, "mask_blur", 1.5))
    if sigma <= 0:
        return mask
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)


def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    """Create a mouth-only mask whose slider cannot reach the eye region."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    mouth_polygon = None
    mouth_box = (0, 0, 0, 0)
    mouth_points, eye_sets = _feature_landmarks(face)
    if mouth_points is None:
        return mask, mouth_cutout, mouth_box, mouth_polygon

    slider = float(np.clip(getattr(modules.globals, "mouth_mask_size", 0.0) / 100.0, 0.0, 1.0))
    center = mouth_points.mean(axis=0)
    offsets = mouth_points - center
    x_scale = 1.0 + 0.65 * slider
    y_scale = np.where(offsets[:, 1] >= 0.0, 1.0 + 1.8 * slider, 1.0 + 0.2 * slider)
    expanded = center + offsets * np.column_stack((np.full(len(offsets), x_scale), y_scale))

    # Hard separation from both eyes, including at the maximum slider value.
    if eye_sets:
        eye_bottom = max(float(np.max(eye[:, 1])) for eye in eye_sets if len(eye))
        mouth_height = max(1.0, float(np.ptp(mouth_points[:, 1])))
        safe_top = eye_bottom + max(2.0, 0.12 * mouth_height)
        expanded[:, 1] = np.maximum(expanded[:, 1], safe_top)

    expanded[:, 0] = np.clip(expanded[:, 0], 0, frame.shape[1] - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, frame.shape[0] - 1)
    expanded = np.rint(expanded).astype(np.int32)
    # The 68-point layout includes inner lip points; its index order is not a
    # single simple outline. A hull makes the paint area stable for both layouts.
    expanded = cv2.convexHull(expanded).reshape(-1, 2)
    span = np.ptp(expanded, axis=0)
    pad_x = max(1, int(round(span[0] * 0.10)))
    pad_y = max(1, int(round(span[1] * 0.10)))
    min_x = max(0, int(np.min(expanded[:, 0])) - pad_x)
    max_x = min(frame.shape[1], int(np.max(expanded[:, 0])) + pad_x + 1)
    min_y = max(0, int(np.min(expanded[:, 1])) - pad_y)
    if eye_sets:
        min_y = max(min_y, int(np.ceil(safe_top)))
    max_y = min(frame.shape[0], int(np.max(expanded[:, 1])) + pad_y + 1)
    if max_x <= min_x or max_y <= min_y:
        return mask, mouth_cutout, mouth_box, mouth_polygon

    roi_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [expanded - np.array([min_x, min_y])], 255)
    mask[min_y:max_y, min_x:max_x] = _mask_blur(roi_mask)
    mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
    mouth_polygon = expanded
    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), mouth_polygon


def create_eyes_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    """Create independent eye ellipses; the mouth slider is never consulted."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyes_cutout = None
    eye_polygon = np.empty((0, 2), dtype=np.int32)
    _, eye_sets = _feature_landmarks(face)
    if not eye_sets or any(len(eye) == 0 for eye in eye_sets):
        return mask, eyes_cutout, (0, 0, 0, 0), eye_polygon

    scale = 1.0 + float(getattr(modules.globals, "mask_down_size", 0.1)) * float(
        getattr(modules.globals, "eyes_mask_size", 0.0)
    )
    all_points = np.vstack(eye_sets)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    padding = max(1, int(round(max(maxs - mins) * 0.20 * scale)))
    min_x = max(0, int(np.floor(mins[0] - padding)))
    min_y = max(0, int(np.floor(mins[1] - padding)))
    max_x = min(frame.shape[1], int(np.ceil(maxs[0] + padding + 1)))
    max_y = min(frame.shape[0], int(np.ceil(maxs[1] + padding + 1)))
    if max_x <= min_x or max_y <= min_y:
        return mask, eyes_cutout, (0, 0, 0, 0), eye_polygon

    yy, xx = np.ogrid[min_y:max_y, min_x:max_x]
    roi_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    polygons = []
    for eye in eye_sets:
        center = np.mean(eye, axis=0)
        axes = np.maximum(1.0, (np.ptp(eye, axis=0) * 0.5) * scale)
        inside = (((xx - center[0]) / axes[0]) ** 2 + ((yy - center[1]) / axes[1]) ** 2) <= 1.0
        roi_mask[inside] = 255
        t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        polygons.append(np.column_stack((center[0] + axes[0] * np.cos(t), center[1] + axes[1] * np.sin(t))))
    mask[min_y:max_y, min_x:max_x] = _mask_blur(roi_mask)
    eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()
    eye_polygon = np.vstack(polygons).astype(np.int32)
    return mask, eyes_cutout, (min_x, min_y, max_x, max_y), eye_polygon


def create_curved_eyebrow(points):
    if len(points) >= 5:
        # Sort points by x-coordinate
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        # Calculate dimensions
        x_min, y_min = np.min(sorted_points, axis=0)
        x_max, y_max = np.max(sorted_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        # Create more points for smoother curve
        num_points = 50
        x = np.linspace(x_min, x_max, num_points)
        
        # Fit quadratic curve through points for more natural arch
        coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
        y = np.polyval(coeffs, x)
        
        # Increased offsets to create more separation
        top_offset = height * 0.5  # Increased from 0.3 to shift up more
        bottom_offset = height * 0.2  # Increased from 0.1 to shift down more
        
        # Create smooth curves
        top_curve = y - top_offset
        bottom_curve = y + bottom_offset
        
        # Create curved endpoints with more pronounced taper
        end_points = 5
        start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Increased taper
        end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Increased taper
        
        # Create tapered ends
        start_curve = np.column_stack((
            start_x,
            np.linspace(bottom_curve[0], top_curve[0], end_points)
        ))
        end_curve = np.column_stack((
            end_x,
            np.linspace(bottom_curve[-1], top_curve[-1], end_points)
        ))
        
        # Combine all points to form a smooth contour
        contour_points = np.vstack([
            start_curve,
            np.column_stack((x, top_curve)),
            end_curve,
            np.column_stack((x[::-1], bottom_curve[::-1]))
        ])
        
        # Add slight padding for better coverage
        center = np.mean(contour_points, axis=0)
        vectors = contour_points - center
        padded_points = center + vectors * 1.2  # Increased padding slightly
        
        return padded_points
    return points

def create_eyebrows_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    eyebrows_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Left eyebrow landmarks (97-105) and right eyebrow landmarks (43-51)
        left_eyebrow = landmarks[97:105].astype(np.float32)
        right_eyebrow = landmarks[43:51].astype(np.float32)
        
        # Calculate centers and dimensions for each eyebrow
        left_center = np.mean(left_eyebrow, axis=0)
        right_center = np.mean(right_eyebrow, axis=0)
        
        # Calculate bounding box with padding adjusted by size
        all_points = np.vstack([left_eyebrow, right_eyebrow])
        padding_factor = modules.globals.eyebrows_mask_size
        min_x = np.min(all_points[:, 0]) - 25 * padding_factor
        max_x = np.max(all_points[:, 0]) + 25 * padding_factor
        min_y = np.min(all_points[:, 1]) - 20 * padding_factor
        max_y = np.max(all_points[:, 1]) + 15 * padding_factor
        
        # Ensure coordinates are within frame bounds
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(frame.shape[1], int(max_x))
        max_y = min(frame.shape[0], int(max_y))
        
        # Create mask for the eyebrows region
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        try:
            # Convert points to local coordinates
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            
            def create_curved_eyebrow(points):
                if len(points) >= 5:
                    # Sort points by x-coordinate
                    sorted_idx = np.argsort(points[:, 0])
                    sorted_points = points[sorted_idx]
                    
                    # Calculate dimensions
                    x_min, y_min = np.min(sorted_points, axis=0)
                    x_max, y_max = np.max(sorted_points, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Create more points for smoother curve
                    num_points = 50
                    x = np.linspace(x_min, x_max, num_points)
                    
                    # Fit quadratic curve through points for more natural arch
                    coeffs = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 2)
                    y = np.polyval(coeffs, x)
                    
                    # Increased offsets to create more separation
                    top_offset = height * 0.5  # Increased from 0.3 to shift up more
                    bottom_offset = height * 0.2  # Increased from 0.1 to shift down more
                    
                    # Create smooth curves
                    top_curve = y - top_offset
                    bottom_curve = y + bottom_offset
                    
                    # Create curved endpoints with more pronounced taper
                    end_points = 5
                    start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)  # Increased taper
                    end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)  # Increased taper
                    
                    # Create tapered ends
                    start_curve = np.column_stack((
                        start_x,
                        np.linspace(bottom_curve[0], top_curve[0], end_points)
                    ))
                    end_curve = np.column_stack((
                        end_x,
                        np.linspace(bottom_curve[-1], top_curve[-1], end_points)
                    ))
                    
                    # Combine all points to form a smooth contour
                    contour_points = np.vstack([
                        start_curve,
                        np.column_stack((x, top_curve)),
                        end_curve,
                        np.column_stack((x[::-1], bottom_curve[::-1]))
                    ])
                    
                    # Add slight padding for better coverage
                    center = np.mean(contour_points, axis=0)
                    vectors = contour_points - center
                    padded_points = center + vectors * 1.2  # Increased padding slightly
                    
                    return padded_points
                return points
            
            # Generate and draw eyebrow shapes
            left_shape = create_curved_eyebrow(left_local)
            right_shape = create_curved_eyebrow(right_local)
            
            # Apply multi-stage blurring for natural feathering (GPU-accelerated when available)
            # First, strong Gaussian blur for initial softening
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            
            # Second, medium blur for transition areas
            mask_roi = gpu_gaussian_blur(mask_roi, (11, 11), 3)
            
            # Finally, light blur for fine details
            mask_roi = gpu_gaussian_blur(mask_roi, (5, 5), 1)
            
            # Normalize mask values
            mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)
            
            # Place the mask ROI in the full-sized mask
            mask[min_y:max_y, min_x:max_x] = mask_roi
            
            # Extract the masked area from the frame
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            
            # Combine points for visualization
            eyebrows_polygon = np.vstack([
                left_shape + [min_x, min_y],
                right_shape + [min_x, min_y]
            ]).astype(np.int32)
            
        except Exception as e:
            # Fallback to simple polygons if curve fitting fails
            left_local = left_eyebrow - [min_x, min_y]
            right_local = right_eyebrow - [min_x, min_y]
            cv2.fillPoly(mask_roi, [left_local.astype(np.int32)], 255)
            cv2.fillPoly(mask_roi, [right_local.astype(np.int32)], 255)
            mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
            eyebrows_polygon = np.vstack([left_eyebrow, right_eyebrow]).astype(np.int32)
        
    return mask, eyebrows_cutout, (min_x, min_y, max_x, max_y), eyebrows_polygon

def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or polygon is None
    ):
        return frame

    try:
        resized_cutout = gpu_resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = gpu_resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        # Create mask for the area
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        
        # Split points for left and right parts if needed
        if len(polygon) > 50:  # Arbitrary threshold to detect if we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point] - [min_x, min_y]
            right_points = polygon[mid_point:] - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [left_points], 255)
            cv2.fillPoly(polygon_mask, [right_points], 255)
        else:
            adjusted_polygon = polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply strong initial feathering (GPU-accelerated when available)
        polygon_mask = gpu_gaussian_blur(polygon_mask, (21, 21), 7)

        # Apply additional feathering
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(np.float32), (0, 0), feather_amount
        )
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask *= np.float32(1.0 / max_val)

        # Apply additional smoothing to the mask edges
        feathered_mask = cv2.GaussianBlur(feathered_mask, (5, 5), 1)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi.astype(np.float32) * np.float32(1.0 / 255.0))

        combined_mask_3ch = combined_mask[:, :, np.newaxis]
        inv_mask = np.float32(1.0) - combined_mask_3ch
        blended = (
            color_corrected_area * combined_mask_3ch + roi * inv_mask
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_f32 = face_mask_roi[:, :, np.newaxis].astype(np.float32) * np.float32(1.0 / 255.0)
        face_mask_3channel = np.broadcast_to(face_mask_f32, blended.shape)
        final_blend = blended * face_mask_3channel + roi * (np.float32(1.0) - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame

def draw_mask_visualization(
    frame: Frame,
    mask_data: tuple,
    label: str,
    draw_method: str = "polygon"
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data

    vis_frame = frame.copy()

    # Ensure coordinates are within frame bounds
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    if draw_method == "ellipse" and len(polygon) > 50:  # For eyes
        # Split points for left and right parts
        mid_point = len(polygon) // 2
        left_points = polygon[:mid_point]
        right_points = polygon[mid_point:]
        
        try:
            # Fit ellipses to points - need at least 5 points
            if len(left_points) >= 5 and len(right_points) >= 5:
                # Convert points to the correct format for ellipse fitting
                left_points = left_points.astype(np.float32)
                right_points = right_points.astype(np.float32)
                
                # Fit ellipses
                left_ellipse = cv2.fitEllipse(left_points)
                right_ellipse = cv2.fitEllipse(right_points)
                
                # Draw the ellipses
                cv2.ellipse(vis_frame, left_ellipse, (0, 255, 0), 2)
                cv2.ellipse(vis_frame, right_ellipse, (0, 255, 0), 2)
        except Exception as e:
            # If ellipse fitting fails, draw simple rectangles as fallback
            left_rect = cv2.boundingRect(left_points)
            right_rect = cv2.boundingRect(right_points)
            cv2.rectangle(vis_frame, 
                        (left_rect[0], left_rect[1]), 
                        (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                        (0, 255, 0), 2)
            cv2.rectangle(vis_frame,
                        (right_rect[0], right_rect[1]),
                        (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]),
                        (0, 255, 0), 2)
    else:  # For mouth and eyebrows
        # Draw the polygon
        if len(polygon) > 50:  # If we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point]
            right_points = polygon[mid_point:]
            cv2.polylines(vis_frame, [left_points], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(vis_frame, [right_points], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.polylines(vis_frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    # Add label
    cv2.putText(
        vis_frame,
        label,
        (min_x, min_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return vis_frame
