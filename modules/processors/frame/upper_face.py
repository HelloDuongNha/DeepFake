"""Current-camera upper-face fallback at oblique poses, without neural inference."""
import cv2
import numpy as np


def _smooth(value):
    value=np.clip(value,0.,1.)
    return value*value*(3.-2.*value)


def preserve_upper_face(frame,camera,face):
    """Restore current forehead, brows and glabella when the head turns.

    Frontal output is untouched. The support stops above the eyes/nose, uses
    current landmarks in the facial axis, and never reuses an older expression.
    """
    if frame.shape!=camera.shape or frame.ndim!=3:return frame
    kps=getattr(face,'kps',None);bbox=getattr(face,'bbox',None)
    points=getattr(face,'landmark_2d_106',None)
    if kps is None or bbox is None or points is None:return frame
    kps=np.asarray(kps,np.float32);bbox=np.asarray(bbox,np.float32);points=np.asarray(points,np.float32)
    if (kps.shape!=(5,2) or bbox.shape!=(4,) or points.shape!=(106,2)
            or not all(np.isfinite(v).all() for v in (kps,bbox,points))):return frame
    width,height=bbox[2:]-bbox[:2]
    direction=kps[1]-kps[0];span=float(np.linalg.norm(direction))
    if min(width,height)<48 or span<max(12.,width*.10):return frame
    origin=kps[:2].mean(0)
    u=direction/span
    if u[0]<0:u=-u
    v=np.array([-u[1],u[0]],np.float32)
    yaw=abs(float((kps[2]-origin)@u))/span
    strength=float(_smooth((yaw-.09)/.07))
    if strength<=0:return frame
    brows=np.concatenate((points[43:52],points[97:106]))
    if np.ptp(brows[:,0])<max(12.,span*.45):return frame
    if (np.any(brows[:,0]<bbox[0]-.25*width) or np.any(brows[:,0]>bbox[2]+.25*width)
            or np.any(brows[:,1]<bbox[1]-.25*height) or np.any(brows[:,1]>bbox[3])):return frame
    # Keep the transition in skin above the eye apertures. Dense eyebrows guide
    # protection, not the upper boundary (which otherwise jumps with a frown).
    cutoff=-height*.015
    feather=max(3.,height*.07)
    corners=np.array([[-width*.70,-height*.70],[width*.70,-height*.70],
                      [width*.70,cutoff],[-width*.70,cutoff]],np.float32)
    corners=origin+corners[:,0,None]*u+corners[:,1,None]*v
    x1,y1=np.maximum(np.floor(corners.min(0)).astype(int),[0,0])
    x2,y2=np.minimum(np.ceil(corners.max(0)).astype(int)+1,[frame.shape[1],frame.shape[0]])
    if x2<=x1 or y2<=y1:return frame
    yy,xx=np.mgrid[y1:y2,x1:x2].astype(np.float32)
    dx=xx-origin[0];dy=yy-origin[1]
    local_x=dx*u[0]+dy*u[1];local_y=dx*v[0]+dy*v[1]
    alpha=_smooth((cutoff-local_y)/feather)
    alpha*=_smooth((width*.65-np.abs(local_x))/max(3.,width*.10))
    alpha*=strength
    # Keep a brow mask only for skin-colour measurement. Cutting holes around
    # generated brows leaves visible islands when their geometry is distorted;
    # forehead and brows must share one continuous camera blend.
    brow_guard=np.zeros((y2-y1,x2-x1),np.uint8)
    for indices in (range(43,52),range(97,106)):
        brow=points[list(indices)]-[x1,y1]
        cv2.fillConvexPoly(brow_guard,cv2.convexHull(np.rint(brow).astype(np.int32)),255)
    radius=max(1,round(width*.006))
    brow_guard=cv2.dilate(brow_guard,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))
    brow_guard=cv2.GaussianBlur(brow_guard,(0,0),max(.7,width*.003)).astype(np.float32)/255.
    # Even tilted contours must never place forehead texture in the eyes.
    eye_guard=np.zeros_like(brow_guard,dtype=np.uint8)
    for indices in (range(33,43),range(87,97)):
        eye=points[list(indices)]-[x1,y1]
        cv2.fillConvexPoly(eye_guard,cv2.convexHull(np.rint(eye).astype(np.int32)),255)
    eye_guard=cv2.dilate(eye_guard,np.ones((3,3),np.uint8))
    alpha[eye_guard>0]=0.
    alpha[alpha<.002]=0.
    if not np.any(alpha):return frame
    roi=frame[y1:y2,x1:x2]
    original=camera[y1:y2,x1:x2].astype(np.float32)
    # Match broad local colour at the transition, fading the correction away
    # into exact camera pixels. Never sharpen or smooth the camera forehead.
    rim=(alpha>.04)&(alpha<.45)&(brow_guard<.1)
    offset=np.zeros(3,np.float32)
    if np.count_nonzero(rim)>=12:
        offset=np.clip(np.median(roi[rim],axis=0)-np.median(original[rim],axis=0),-18.,18.)
    matched=np.clip(original+offset*(1.-alpha[:,:,None]),0,255)
    roi[:]=np.rint(roi*(1.-alpha[:,:,None])+matched*alpha[:,:,None]).clip(0,255).astype(np.uint8)
    return frame
