"""Small CPU passes for skin texture transfer and final eye/brow clarity.

Camera texture must not reintroduce the subject's brows or isolated skin marks.
Final clarity uses only the current processed frame, without another restorer
or a previous expression.
"""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=8)
def _aligned_skin_gate(height: int, width: int) -> np.ndarray:
    gate = np.full((height, width), 255, np.uint8)
    # INSwapper's ArcFace alignment puts the eyes at y ~= .46. Keep the
    # camera's eye/brow contrast out of the transferred skin residual.
    for x in (.342, .657):
        cv2.ellipse(
            gate, (round(width * x), round(height * .40)),
            (max(1, round(width * .16)), max(1, round(height * .15))),
            0, 0, 360, 0, -1,
        )
    gate = cv2.GaussianBlur(gate, (0, 0), max(.7, min(height, width) * .009))
    return gate.astype(np.float32) / 255.0


def skin_microtexture_gate(camera_crop: np.ndarray) -> np.ndarray:
    """Keep wrinkles/pores, exclude compact dark marks from camera transfer.

    This gates the additive camera residual; it never inpaints or smooths the
    generated face. Long wrinkles and broad illumination are left alone.
    """
    height, width = camera_crop.shape[:2]
    gate = _aligned_skin_gate(height, width).copy()
    gray = cv2.cvtColor(camera_crop, cv2.COLOR_BGR2GRAY)
    diameter = max(5, min(15, round(min(height, width) * .065) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    skin_only = np.full((height, width), 255, np.uint8)
    # Compact nostril shadows and tooth/lip edges are anatomy, not blemishes.
    # Keep their existing camera residual unchanged.
    for cy, ax, ay in ((.57, .13, .10), (.74, .20, .08)):
        cv2.ellipse(skin_only, (round(width * .5), round(height * cy)),
                    (max(1, round(width * ax)), max(1, round(height * ay))),
                    0, 0, 360, 0, -1)
    candidate = ((dark >= 12) & (gate > .85) & (skin_only > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    blocked = np.zeros((height, width), np.uint8)
    max_area = max(4, height * width * .0025)
    max_side = max(4, min(height, width) * .085)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        aspect = max(w, h) / max(1, min(w, h))
        if area <= max_area and max(w, h) <= max_side and aspect <= 2.2:
            roi = blocked[y:y + h, x:x + w]
            roi[labels[y:y + h, x:x + w] == label] = 255
    if np.any(blocked):
        radius = max(1, round(min(height, width) * .018))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        blocked = cv2.dilate(blocked, kernel)
        blocked = cv2.GaussianBlur(blocked, (0, 0), max(.6, radius * .4))
        blocked = np.minimum(blocked, skin_only)
        gate *= 1.0 - blocked.astype(np.float32) / 255.0
    return gate


def enhance_eye_clarity(frame: np.ndarray, face, strength: float) -> np.ndarray:
    """Sharpen current generated eye/brow edges after restoration, in place.

    The slider's 0..5 range controls this local pass. Pixels outside the two
    soft feature masks remain identical, including nose, mouth and face rim.
    """
    if face is None or strength <= 0.0 or frame.size == 0:
        return frame
    try:
        kps = np.asarray(face.kps, np.float32)
        bbox = np.asarray(face.bbox, np.float32)
        if (kps.shape != (5, 2) or bbox.shape != (4,)
                or not np.isfinite(kps).all() or not np.isfinite(bbox).all()):
            return frame
        face_width, face_height = bbox[2:] - bbox[:2]
        eye_span = float(np.linalg.norm(kps[1] - kps[0]))
        if face_width < 48 or face_height < 48 or eye_span < 16:
            return frame
        yaw = abs(float((kps[2, 0] - np.mean(kps[:2, 0])) / eye_span))
        pose_weight = float(np.clip((.38 - yaw) / .16, 0.0, 1.0))
        if pose_weight <= 0:
            return frame
        eye_line = kps[1] - kps[0]
        angle = float(np.degrees(np.arctan2(eye_line[1], eye_line[0])))
        up = np.array([np.sin(np.radians(angle)), -np.cos(np.radians(angle))])
        brows = kps[:2] + up * (face_height * .09)
        centers = np.concatenate((kps[:2], brows))
        padding = max(face_width * .18, face_height * .12) + 6
        x1 = max(0, int(np.floor(centers[:, 0].min() - padding)))
        y1 = max(0, int(np.floor(centers[:, 1].min() - padding)))
        x2 = min(frame.shape[1], int(np.ceil(centers[:, 0].max() + padding)))
        y2 = min(frame.shape[0], int(np.ceil(centers[:, 1].max() + padding)))
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        mask = np.zeros(roi.shape[:2], np.uint8)
        for index, point in enumerate(centers):
            axes = (round(face_width * (.135 if index < 2 else .15)),
                    round(face_height * (.043 if index < 2 else .045)))
            cv2.ellipse(mask, tuple(np.rint(point - [x1, y1]).astype(int)),
                        tuple(max(1, axis) for axis in axes), angle,
                        0, 360, 255, -1, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(mask, (0, 0), max(1., face_width * .007))
        alpha = alpha.astype(np.float32) / 255.
        alpha[alpha < .015] = 0.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        guide = cv2.bilateralFilter(gray, 3, 8., 1.)
        sigma = float(np.clip(face_width / 280., .8, 1.8))
        fine = guide - cv2.GaussianBlur(guide, (0, 0), sigma)
        medium = guide - cv2.GaussianBlur(guide, (0, 0), sigma * 2.)
        residual = fine * .65 + medium * .45
        residual = np.sign(residual) * np.maximum(np.abs(residual) - 2., 0.)
        gain = float(np.sqrt(np.clip(strength / 5., 0., 1.))) * pose_weight * .5
        correction = np.clip(residual * gain, -4., 4.)
        # Bound overshoot near high-contrast sclera/eyelids. There is no frame
        # cache and no spatial resampling, so blinks use this frame's geometry.
        kernel = np.ones((5, 5), np.uint8)
        proposed = np.clip(gray + correction,
                           cv2.erode(gray, kernel) - 3.,
                           cv2.dilate(gray, kernel) + 3.)
        correction = (proposed - gray) * alpha
        roi[:] = np.rint(np.clip(
            roi.astype(np.float32) + correction[:, :, None], 0, 255
        )).astype(np.uint8)
        return frame
    except (AttributeError, TypeError, ValueError, cv2.error):
        return frame


def preserve_eye_apertures(frame: np.ndarray, camera: np.ndarray, face, repair_eyelids=False) -> np.ndarray:
    """Keep current camera eyeballs strictly inside the tracked eyelid contour.

    By default, brows, eyelids and surrounding skin stay swapped. Optional
    eyelid repair includes a small current-camera rim to replace conflicting
    generated eyelid edges. No cached texture is used; invalid eyes are skipped.
    """
    points = getattr(face, 'landmark_2d_106', None)
    if points is None or frame.shape != camera.shape:
        return frame
    points = np.asarray(points, np.float32)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        return frame
    h, w = frame.shape[:2]
    for indices in (range(33, 43), range(87, 97)):
        eye = points[list(indices)]
        x, y, ew, eh = cv2.boundingRect(np.rint(eye).astype(np.int32))
        if ew < 8 or eh < 3 or eh > ew * .8:
            continue
        if repair_eyelids:
            _blend_contour(frame,camera,eye,max(2,round(ew*.10)),max(1.5,ew*.045))
            continue
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(w, x + ew + 2), min(h, y + eh + 2)
        if x2 <= x1 or y2 <= y1:
            continue
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        hull = cv2.convexHull(np.rint(eye - [x1, y1]).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        # Feather inward only: never put camera iris pixels above the eyelid.
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip((distance - .8) / max(1., ew * .035), 0., 1.)[:, :, None]
        roi = frame[y1:y2, x1:x2]
        roi[:] = np.rint(roi * (1. - alpha) + camera[y1:y2, x1:x2] * alpha).astype(np.uint8)
    return frame


def preserve_natural_nose(frame: np.ndarray, camera: np.ndarray, face) -> np.ndarray:
    """Restore camera tip/nostrils with an inward feather; leave bridge/skin swapped.

    Opt-in: this preserves the camera subject's lower nose anatomy. It does not
    claim to reconstruct the source portrait's nose at a higher resolution.
    """
    points = getattr(face, 'landmark_2d_106', None)
    if points is None or frame.shape != camera.shape:
        return frame
    points = np.asarray(points, np.float32)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        return frame
    nose = points[74:87]
    # End the blend below the nostril shadow, rather than directly through it.
    # The old contour put semi-transparent pixels across the lower nostril edge.
    nose_width = float(np.ptp(nose[:, 0]))
    apron = points[78:86] + [0., max(1., nose_width * .08)]
    nose = np.concatenate((nose, apron)).astype(np.float32)
    x, y, width, height = cv2.boundingRect(np.rint(nose).astype(np.int32))
    if width < 8 or height < 6 or width > frame.shape[1] * .35 or height > width * 1.8:
        return frame
    x1, y1 = max(0, x - 3), max(0, y - 3)
    x2, y2 = min(frame.shape[1], x + width + 3), min(frame.shape[0], y + height + 3)
    if x2 <= x1 or y2 <= y1:
        return frame
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(nose - [x1, y1]).astype(np.int32)), 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    radius = max(1.5, width * .035)
    alpha = np.clip((distance - .6) / radius, 0., 1.)[:, :, None]
    roi = frame[y1:y2, x1:x2]
    original = camera[y1:y2, x1:x2].astype(np.float32)
    rim = (distance > 0) & (distance <= radius)
    if np.count_nonzero(rim) < 6:
        return frame
    # Match only broad color; do not sharpen/smooth camera nostril detail.
    offset = np.clip(np.median(roi[rim], axis=0) - np.median(original[rim], axis=0), -24., 24.)
    matched = np.clip(original + offset, 0., 255.)
    roi[:] = np.rint(roi * (1. - alpha) + matched * alpha).astype(np.uint8)
    return frame


def _blend_contour(frame,camera,points,expansion,feather,lower_nose=False):
    """Use current reference feature contours; match only the outer skin transition.

    No cached texture, invented detail, or final-frame blur. Work in a bounded ROI.
    """
    x,y,width,height=cv2.boundingRect(np.rint(points).astype(np.int32))
    pad=expansion+3
    x1,y1=max(0,x-pad),max(0,y-pad)
    x2,y2=min(frame.shape[1],x+width+pad),min(frame.shape[0],y+height+pad)
    if x2<=x1 or y2<=y1:return frame
    core=np.zeros((y2-y1,x2-x1),np.uint8)
    cv2.fillConvexPoly(core,cv2.convexHull(np.rint(points-[x1,y1]).astype(np.int32)),255)
    expanded=cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*expansion+1,)*2))
    distance=cv2.distanceTransform(expanded,cv2.DIST_L2,3)
    alpha=np.clip(distance/feather,0.,1.)
    if lower_nose:
        yy,xx=np.mgrid[y1:y2,x1:x2].astype(np.float32)
        # Keep the bridge and central upper tip generated. Restore the alar
        # edges and lower nostrils where double contours are most conspicuous.
        down=np.clip((yy-(y+height*.32))/max(2.,height*.22),0.,1.)
        sides=np.clip((np.abs(xx-(x+width*.5))-width*.14)/max(2.,width*.12),0.,1.)
        base=np.clip((yy-(y+height*.58))/max(2.,height*.20),0.,1.)
        alpha*=down*np.maximum(sides,base)
    roi=frame[y1:y2,x1:x2]
    original=camera[y1:y2,x1:x2].astype(np.float32)
    ring=(distance>0)&(distance<feather)
    offset=np.zeros(3,np.float32)
    if np.count_nonzero(ring)>=6:
        offset=np.clip(np.median(roi[ring],axis=0)-np.median(original[ring],axis=0),-20.,20.)
    # Reference iris/nostril pixels in the contour core stay exact. Color matching
    # is restricted to the expanded skin band.
    matched=np.clip(original+offset*(1.-core[:,:,None]/255.),0.,255.)
    roi[:]=np.rint(roi*(1.-alpha[:,:,None])+matched*alpha[:,:,None]).clip(0,255).astype(np.uint8)
    return frame


def repair_nose_edges(frame,reference,face):
    """Restore model-generated nose contours before enhancement, not camera anatomy."""
    points=getattr(face,'landmark_2d_106',None)
    if points is None or frame.shape!=reference.shape:return frame
    points=np.asarray(points,np.float32)
    if points.shape!=(106,2) or not np.isfinite(points).all():return frame
    nose=points[74:87]
    width=float(np.ptp(nose[:,0]));height=float(np.ptp(nose[:,1]))
    if width<8 or height<6 or width>frame.shape[1]*.35 or height>width*1.8:return frame
    result=_blend_contour(frame,reference,nose,max(2,round(width*.10)),max(1.5,width*.06),lower_nose=True)
    # Restoration/paste-back may strengthen a residual ridge even after the
    # native cleanup. Check this frame's final contour once, without caching.
    kps=getattr(face,'kps',None)
    if kps is None:return result
    kps=np.asarray(kps,np.float32)
    if kps.shape!=(5,2) or not np.isfinite(kps).all():return result
    span=max(1.,float(np.linalg.norm(kps[1]-kps[0])))
    yaw=abs(float(kps[2,0]-np.mean(kps[:2,0])))/span
    return suppress_model_nose_ridges(result,points) if yaw<.24 else result


def suppress_model_nose_ridges(image,points):
    """Reduce narrow lateral highlights already present in a generated nose.

    Works on the native generated crop before restoration. Only detected bright
    side ridges are filled; bridge, dark nostril openings and other features are
    protected. This does not reconstruct nose geometry or copy camera anatomy.
    """
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:return image
    points=np.asarray(points,np.float32)
    if points.shape!=(106,2) or not np.isfinite(points).all():return image
    nose=points[74:87]
    x,y,width,height=cv2.boundingRect(np.rint(nose).astype(np.int32))
    if width<12 or height<10 or width>image.shape[1]*.5 or height>width*2.:return image
    x1=max(0,x-round(width*.25));x2=min(image.shape[1],x+round(width*1.25))
    y1=max(0,y);y2=min(image.shape[0],y+height+2)
    if x2<=x1 or y2<=y1:return image
    roi=image[y1:y2,x1:x2]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    yy,xx=np.mgrid[y1:y2,x1:x2].astype(np.float32)
    center=(float(nose[:,0].min())+float(nose[:,0].max()))*.5
    sides=((np.abs(xx-center)>width*.18)&(np.abs(xx-center)<width*.58)
           &(yy>y+height*.33)&(yy<y+height*.94))
    broad=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(1.,width*.10),sigmaY=.7)
    response=gray.astype(np.float32)-broad
    mask=np.zeros_like(gray)
    radius=max(1,int(np.ceil(width*.055)))
    for side in (-1,1):
        region=sides&((xx-center)*side>0)
        for row in range(len(gray)):
            columns=np.flatnonzero(region[row])
            if not len(columns):continue
            peak=int(columns[np.argmax(response[row,columns])])
            if response[row,peak]>2. and gray[row,peak]>95:
                cv2.ellipse(mask,(peak,row),(radius,1),0,0,360,255,-1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    nostrils=((gray<90)&(yy>y+height*.60)).astype(np.uint8)
    nostrils=cv2.dilate(nostrils,np.ones((3,3),np.uint8))
    mask[(nostrils>0)|~sides]=0
    if np.count_nonzero(mask)<4:return image
    filled=cv2.inpaint(roi,mask,max(1.,width*.035),cv2.INPAINT_TELEA)
    alpha=np.clip(cv2.distanceTransform(mask,cv2.DIST_L2,3)/1.4,0,1)*.9
    # Bound local changes; do not replace an entire wing with a flat patch.
    delta=np.clip(filled.astype(np.float32)-roi.astype(np.float32),-20.,20.)
    result=image.copy()
    result[y1:y2,x1:x2]=np.rint(roi+delta*alpha[:,:,None]).clip(0,255).astype(np.uint8)
    return result
