"""Validated desktop preferences; atomic saves keep the previous file on error."""
import json
import math
import os
import tempfile

HISTORY_LIMIT=24
BOOL_FIELDS=('keep_fps','keep_audio','keep_frames','many_faces','poisson_blend',
             'color_correction','nsfw_filter','live_mirror','live_resizable','show_fps',
             'mask_enabled','live_portrait','natural_eyes','natural_nose','occlusion_protection',
             'preserve_inner_mouth','eyelid_repair','nose_edge_repair','forehead_guard')
NUMERIC_FIELDS={'opacity':(0.,1.),'sharpness':(0.,5.)}

def read():
    from modules.paths import SETTINGS_PATH
    try:
        with open(SETTINGS_PATH,encoding='utf-8') as f:data=json.load(f)
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError):return {}

def history(value):
    result=[]
    if not isinstance(value,list):return result
    for path in value:
        if isinstance(path,str) and path and path not in result:
            result.append(path)
        if len(result)>=HISTORY_LIMIT:break
    return result

def apply(state,settings):
    for key in BOOL_FIELDS:
        if type(state.get(key)) is bool:setattr(settings,key,state[key])
    for key,(low,high) in NUMERIC_FIELDS.items():
        value=state.get(key)
        if type(value) in (int,float) and math.isfinite(value):
            setattr(settings,key,max(low,min(high,float(value))))
    flags=state.get('fp_ui')
    if isinstance(flags,dict):
        settings.fp_ui={key:flags.get(key,False) is True for key in
            ('face_enhancer','face_enhancer_gpen256','face_enhancer_gpen512','face_enhancer_restoreformer')}
        if settings.fp_ui['face_enhancer_gpen512']:
            settings.fp_ui.update(face_enhancer=False,face_enhancer_gpen512=False,face_enhancer_gpen256=True)
    settings.source_history=history(state.get('source_history',[]))
    source=state.get('source_path')
    # Existing CLI source, if supplied, takes priority over desktop history.
    if not settings.source_path and isinstance(source,str) and os.path.isfile(source):
        settings.source_path=source
    for key in ('window_geometry','live_window_geometry'):
        value=state.get(key)
        if isinstance(value,str) and len(value)<8192:setattr(settings,key,value)
    value=state.get('preferred_camera_index')
    if type(value) is int and value>=0:settings.preferred_camera_index=value

def save(settings):
    from modules.paths import SETTINGS_PATH
    state={key:getattr(settings,key) for key in BOOL_FIELDS}
    state.update({key:getattr(settings,key) for key in NUMERIC_FIELDS})
    state.update(fp_ui=dict(settings.fp_ui),swapper_model=settings.swapper_model,
                 source_path=settings.source_path,source_history=history(settings.source_history),
                 preferred_camera_index=settings.preferred_camera_index,
                 window_geometry=settings.window_geometry,live_window_geometry=settings.live_window_geometry,
                 version=2)
    parent=os.path.dirname(SETTINGS_PATH);temp=None
    try:
        os.makedirs(parent,exist_ok=True)
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=parent,delete=False) as f:
            temp=f.name;json.dump(state,f,ensure_ascii=False,indent=2)
        os.replace(temp,SETTINGS_PATH)
        return True
    except OSError as error:
        print(f'[settings] Could not save preferences: {error}',flush=True)
        return False
    finally:
        if temp and os.path.exists(temp):
            try:os.unlink(temp)
            except OSError:pass
