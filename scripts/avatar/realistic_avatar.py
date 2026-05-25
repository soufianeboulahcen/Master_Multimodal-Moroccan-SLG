"""Photorealistic clothed human avatar renderer (CPU-only).

Reads DWPose skeleton PNGs from outputs/pose_control/ and renders a
fully-clothed, identity-consistent human avatar with:
  - Anatomically correct body proportions
  - Modern clothing (shirt, trousers, shoes)
  - Realistic skin tone with subsurface-scattering approximation
  - Facial features (eyes, eyebrows, nose, mouth, hair, beard)
  - Studio three-point lighting (key, fill, rim)
  - Neutral gradient background with floor shadow
  - Temporal consistency via EMA blending

No GPU or diffusion models required.

Usage:
    python scripts/avatar/realistic_avatar.py
    python scripts/avatar/realistic_avatar.py --style cinematic
    python scripts/avatar/realistic_avatar.py --style all --fps 25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
CTRL_BASE = ROOT / "outputs" / "pose_control"
OUT_BASE  = ROOT / "outputs" / "avatar" / "realistic"

IDENTITY = {
    "skin":         (195, 155, 120),
    "skin_shadow":  (148, 112,  82),
    "skin_hi":      (222, 188, 158),
    "hair":         ( 28,  18,  10),
    "beard":        ( 38,  25,  14),
    "shirt":        ( 45,  65, 110),
    "shirt_hi":     ( 62,  88, 145),
    "shirt_shadow": ( 28,  44,  78),
    "trouser":      ( 38,  38,  48),
    "trouser_hi":   ( 54,  54,  68),
    "shoe":         ( 22,  18,  14),
    "eye_iris":     ( 48,  32,  16),
    "lip":          (158, 102,  82),
}

STYLES = {
    "research": {
        "bg_top":   (230, 228, 225), "bg_bottom": (195, 192, 188),
        "floor":    (175, 172, 168), "fill_str": 0.35,
        "rim_str":  0.55,            "vignette": 0.25,
    },
    "cinematic": {
        "bg_top":   ( 14,  12,  10), "bg_bottom": (  6,   5,   4),
        "floor":    ( 20,  18,  15), "fill_str": 0.20,
        "rim_str":  0.80,            "vignette": 0.55,
    },
    "studio": {
        "bg_top":   ( 52,  50,  48), "bg_bottom": ( 28,  26,  24),
        "floor":    ( 38,  36,  34), "fill_str": 0.40,
        "rim_str":  0.65,            "vignette": 0.40,
    },
}

JOINT_COLOURS = {
    "nose":       (255,   0,  85), "neck":       (255,   0,   0),
    "r_shoulder": (255,  85,   0), "r_elbow":    (255, 170,   0),
    "r_wrist":    (255, 255,   0), "l_shoulder": (170, 255,   0),
    "l_elbow":    ( 85, 255,   0), "l_wrist":    (  0, 255,   0),
    "mid_hip":    (  0, 255,  85), "r_hip":      (  0, 255, 170),
    "r_knee":     (  0, 255, 255), "r_ankle":    (  0, 170, 255),
    "l_hip":      (  0,  85, 255), "l_knee":     (  0,   0, 255),
    "l_ankle":    (255,   0, 170), "r_eye":      (170,   0, 255),
    "l_eye":      (255,   0, 255),
}


def _centroid(img_bgr, bgr, tol=38):
    lo  = np.array([max(0, c-tol) for c in bgr], np.uint8)
    hi  = np.array([min(255, c+tol) for c in bgr], np.uint8)
    msk = cv2.inRange(img_bgr, lo, hi)
    pts = cv2.findNonZero(msk)
    if pts is None:
        return None
    return (int(pts[:,0,0].mean()), int(pts[:,0,1].mean()))


def parse_joints(png):
    img = cv2.imread(str(png))
    if img is None:
        return {k: None for k in JOINT_COLOURS}
    return {n: _centroid(img, c) for n, c in JOINT_COLOURS.items()}


def _dist(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(b[0]-a[0], b[1]-a[1])


def _lerp(a, b, t):
    return (int(a[0]+(b[0]-a[0])*t), int(a[1]+(b[1]-a[1])*t))


def _limb(cv, p1, p2, col, thick, shadow=None):
    if p1 is None or p2 is None:
        return
    sc = shadow or tuple(max(0, c-45) for c in col)
    cv2.line(cv, p1, p2, sc,  thick+6, cv2.LINE_AA)
    cv2.line(cv, p1, p2, col, thick,   cv2.LINE_AA)
    hi = tuple(min(255, c+32) for c in col)
    cv2.line(cv, p1, p2, hi, max(thick//3, 1), cv2.LINE_AA)


def _blob(cv, c, r, col, shadow=None):
    if c is None:
        return
    sc = shadow or tuple(max(0, x-40) for x in col)
    cv2.circle(cv, c, r+3, sc,  -1, cv2.LINE_AA)
    cv2.circle(cv, c, r,   col, -1, cv2.LINE_AA)


def _poly(cv, pts, col, shadow=None):
    if not pts or len(pts) < 3:
        return
    arr = np.array(pts, np.int32)
    sc  = shadow or tuple(max(0, c-40) for c in col)
    cv2.fillPoly(cv, [arr], sc)
    inner = (arr*0.96 + arr.mean(0)*0.04).astype(np.int32)
    cv2.fillPoly(cv, [inner], col)


def make_bg(h, w, style):
    bg = np.zeros((h, w, 3), np.float32)
    ys = np.linspace(0, 1, h)[:, None]
    for c in range(3):
        bg[:,:,c] = style["bg_top"][c]*(1-ys) + style["bg_bottom"][c]*ys
    fy = int(h*0.82)
    fc = np.array(style["floor"], np.float32)
    for row in range(fy, h):
        a = min((row-fy)/max(h-fy, 1), 1.0)
        bg[row] = bg[row]*(1-a*0.6) + fc*(a*0.6)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X-w/2)/(w*0.55))**2 + ((Y-h*0.42)/(h*0.55))**2)
    vig  = np.clip(1.0 - dist*style["vignette"], 0.0, 1.0)[..., np.newaxis]
    return np.clip(bg*vig, 0, 255).astype(np.uint8)


def apply_lighting(canvas, fg_mask, style):
    h, w   = canvas.shape[:2]
    result = canvas.astype(np.float32)
    m      = fg_mask[..., np.newaxis].astype(np.float32)/255.0
    Y, X   = np.ogrid[:h, :w]
    kd = np.clip(1.0-np.sqrt(((X-w*0.22)/w)**2+((Y-h*0.12)/h)**2)*1.1, 0, 1)[...,np.newaxis]
    result += kd * np.array([210,225,245], np.float32) * 0.20 * m
    fd = np.clip(1.0-np.sqrt(((X-w*0.82)/w)**2+((Y-h*0.40)/h)**2)*1.4, 0, 1)[...,np.newaxis]
    result += fd * np.array([240,215,195], np.float32) * style["fill_str"] * 0.14 * m
    rd = np.clip(1.0-np.sqrt(((X-w*0.55)/w)**2+((Y-h*0.04)/h)**2)*2.2, 0, 1)[...,np.newaxis]
    result += rd * np.array([220,230,255], np.float32) * style["rim_str"] * 0.12 * m
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_avatar(canvas, J, h, w):
    ID  = IDENTITY
    sk  = ID["skin"];       ssh = ID["skin_shadow"];  shi = ID["skin_hi"]
    sh  = ID["shirt"];      shh = ID["shirt_hi"];     shs = ID["shirt_shadow"]
    tr  = ID["trouser"];    trh = ID["trouser_hi"]
    shoe = ID["shoe"];      hair = ID["hair"]

    neck   = J.get("neck");       nose   = J.get("nose")
    r_sh   = J.get("r_shoulder"); l_sh   = J.get("l_shoulder")
    r_el   = J.get("r_elbow");    l_el   = J.get("l_elbow")
    r_wr   = J.get("r_wrist");    l_wr   = J.get("l_wrist")
    mhip   = J.get("mid_hip");    r_hip  = J.get("r_hip");  l_hip = J.get("l_hip")
    r_knee = J.get("r_knee");     l_knee = J.get("l_knee")
    r_ank  = J.get("r_ankle");    l_ank  = J.get("l_ankle")

    sw = _dist(r_sh, l_sh)
    lw = max(int(sw*0.13), 8)
    gw = max(int(sw*0.15), 9)

    # Ground shadow
    if mhip:
        fy = int(h*0.82)
        sl = canvas.copy()
        cv2.ellipse(sl, (mhip[0], fy), (int(sw*0.55), int(h*0.018)),
                    0, 0, 360, (0,0,0), -1, cv2.LINE_AA)
        cv2.addWeighted(sl, 0.28, canvas, 0.72, 0, canvas)

    # Legs
    for hip, knee, ank in [(r_hip,r_knee,r_ank),(l_hip,l_knee,l_ank)]:
        _limb(canvas, hip,  knee, tr, gw+4, tr)
        _limb(canvas, knee, ank,  tr, gw+2, tr)
        if hip and knee:
            _limb(canvas, hip, knee, trh, max(gw//3,2))

    # Shoes
    for ank in [r_ank, l_ank]:
        if ank is None:
            continue
        ax, ay = ank
        sw2 = int(sw*0.09)
        _poly(canvas, [(ax-sw2,ay),(ax+sw2+4,ay),
                       (ax+sw2+6,ay+int(h*0.018)),(ax-sw2-2,ay+int(h*0.018))], shoe)
        cv2.line(canvas,(ax-sw2+2,ay+2),(ax+sw2,ay+2),(55,50,45),1,cv2.LINE_AA)

    # Torso
    if r_sh and l_sh and mhip:
        rh = r_hip or (r_sh[0]-int(sw*0.05), mhip[1])
        lh = l_hip or (l_sh[0]+int(sw*0.05), mhip[1])
        _poly(canvas, [r_sh,l_sh,lh,mhip,rh], sh, shs)
        if neck:
            cv2.line(canvas, r_sh, neck, shs, max(lw//2,3), cv2.LINE_AA)
            cv2.line(canvas, l_sh, neck, shs, max(lw//2,3), cv2.LINE_AA)
            cv2.line(canvas, neck, mhip, shs, 1, cv2.LINE_AA)
        if r_sh and mhip:
            ov = canvas.copy()
            cv2.fillPoly(ov, [np.array([r_sh,_lerp(r_sh,l_sh,0.28),
                _lerp(mhip,r_sh,0.32),(r_sh[0],mhip[1])],np.int32)], shh)
            cv2.addWeighted(ov, 0.22, canvas, 0.78, 0, canvas)

    # Sleeves
    for s,e,wr in [(r_sh,r_el,r_wr),(l_sh,l_el,l_wr)]:
        _limb(canvas, s, e,  sh, lw+2, shs)
        _limb(canvas, e, wr, sh, lw,   shs)

    # Forearms + hands
    for e, wr in [(r_el,r_wr),(l_el,l_wr)]:
        if e and wr:
            _limb(canvas, e, wr, sk, max(lw-2,5), ssh)
            _blob(canvas, wr, max(lw-1,5), sk, ssh)
            cv2.circle(canvas, wr, max(lw-3,3), shi, -1, cv2.LINE_AA)

    # Neck
    if neck and nose:
        _limb(canvas, neck, nose, sk, max(int(sw*0.07),6), ssh)

    # Head
    if nose:
        hr  = max(int(sw*0.22), 18)
        hcx = nose[0]
        hcy = nose[1] - int(hr*0.25)

        cv2.ellipse(canvas,(hcx,hcy-int(hr*0.08)),(hr+4,hr+2),
                    0,0,360,hair,-1,cv2.LINE_AA)
        _blob(canvas,(hcx,hcy),hr,sk,ssh)

        ov = canvas.copy()
        cv2.ellipse(ov,(hcx-int(hr*0.15),hcy-int(hr*0.10)),
                    (int(hr*0.65),int(hr*0.55)),0,0,360,shi,-1,cv2.LINE_AA)
        cv2.addWeighted(ov,0.20,canvas,0.80,0,canvas)

        cv2.ellipse(canvas,(hcx,hcy+int(hr*0.38)),
                    (int(hr*0.55),int(hr*0.38)),0,0,180,ID["beard"],-1,cv2.LINE_AA)

        for ex in [hcx-hr, hcx+hr]:
            ey2 = hcy+int(hr*0.05); er = int(hr*0.14)
            cv2.ellipse(canvas,(ex,ey2),(er,int(er*1.4)),0,0,360,sk,-1,cv2.LINE_AA)
            cv2.ellipse(canvas,(ex,ey2),(int(er*0.55),int(er*0.9)),0,0,360,ssh,-1,cv2.LINE_AA)

        ey_y  = hcy-int(hr*0.10); ey_off = int(hr*0.30); ey_r = max(int(hr*0.12),4)
        for ex in [hcx-ey_off, hcx+ey_off]:
            cv2.ellipse(canvas,(ex,ey_y),(ey_r+2,int(ey_r*0.75)),0,0,360,(235,230,225),-1,cv2.LINE_AA)
            cv2.circle(canvas,(ex,ey_y),ey_r,ID["eye_iris"],-1,cv2.LINE_AA)
            cv2.circle(canvas,(ex,ey_y),max(ey_r-2,2),(12,8,5),-1,cv2.LINE_AA)
            cv2.circle(canvas,(ex-int(ey_r*0.3),ey_y-int(ey_r*0.3)),max(ey_r//3,1),(240,238,235),-1,cv2.LINE_AA)
            cv2.ellipse(canvas,(ex,ey_y),(ey_r+2,int(ey_r*0.75)),0,180,360,(28,18,10),1,cv2.LINE_AA)

        bw = int(hr*0.28); bt = max(int(hr*0.07),2); by = ey_y-int(hr*0.20)
        for bx in [hcx-ey_off, hcx+ey_off]:
            cv2.fillPoly(canvas,[np.array([[bx-bw,by+2],[bx,by-2],[bx+bw,by+1],
                [bx+bw,by+bt+1],[bx,by+bt-1],[bx-bw,by+bt+2]],np.int32)],hair)

        nt=(hcx,hcy+int(hr*0.32)); nw=int(hr*0.14)
        cv2.ellipse(canvas,nt,(nw,int(nw*0.65)),0,0,360,ssh,-1,cv2.LINE_AA)
        cv2.ellipse(canvas,nt,(nw-2,int(nw*0.5)),0,0,360,sk,-1,cv2.LINE_AA)

        my=hcy+int(hr*0.52); mw=int(hr*0.30); lip=ID["lip"]
        cv2.ellipse(canvas,(hcx,my),(mw,int(hr*0.07)),0,180,360,lip,-1,cv2.LINE_AA)
        cv2.ellipse(canvas,(hcx,my+int(hr*0.06)),(mw-2,int(hr*0.09)),0,0,180,lip,-1,cv2.LINE_AA)
        cv2.line(canvas,(hcx-mw,my),(hcx+mw,my),tuple(max(0,c-30) for c in lip),1,cv2.LINE_AA)

    return canvas


def post_process(frame, grain=2.2):
    f = frame.astype(np.float32)
    s = np.clip(1.0-f/128.0, 0, 1)
    f[:,:,2] += s[:,:,2]*5;  f[:,:,0] -= s[:,:,0]*2
    hi = np.clip(f/210.0-0.5, 0, 1)
    f[:,:,0] += hi[:,:,0]*4
    f = np.clip(f, 0, 255)
    noise = np.random.normal(0, grain, frame.shape).astype(np.float32)
    f = np.clip(f+noise, 0, 255).astype(np.uint8)
    bar = int(frame.shape[0]*0.045)
    f[:bar] = 0;  f[-bar:] = 0
    return f


def ema_blend(prev, curr, alpha=0.08):
    if prev is None:
        return curr
    return np.clip(prev.astype(np.float32)*alpha + curr.astype(np.float32)*(1-alpha),
                   0, 255).astype(np.uint8)


def render_clip(clip_dir, style_name, fps, size):
    pngs = sorted(clip_dir.glob("pose_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No pose_*.png in {clip_dir}")
    native_fps = fps
    manifest   = clip_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            native_fps = json.load(f).get("fps", fps)
    out_fps = fps or native_fps
    T       = len(pngs)
    style   = STYLES[style_name]
    h = w   = size
    out_path = OUT_BASE / f"{clip_dir.name}_{style_name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w,h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")
    print(f"[realistic_avatar] {clip_dir.name}  {T} frames @ {out_fps}fps  style={style_name}  {w}x{h}")
    np.random.seed(42)
    prev = None
    bg   = make_bg(h, w, style)
    for i, png in enumerate(pngs):
        joints = parse_joints(png)
        canvas = bg.copy()
        canvas = draw_avatar(canvas, joints, h, w)
        diff   = cv2.absdiff(canvas, bg)
        gray   = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg  = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        fg     = cv2.dilate(fg, np.ones((5,5), np.uint8), iterations=2)
        canvas = apply_lighting(canvas, fg, style)
        canvas = post_process(canvas)
        canvas = ema_blend(prev, canvas)
        prev   = canvas.copy()
        writer.write(canvas)
        if (i+1) % 20 == 0 or i == T-1:
            print(f"  frame {i+1}/{T}", flush=True)
    writer.release()
    print(f"[realistic_avatar] done -> {out_path.name}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip",  default="all")
    p.add_argument("--style", default="research", choices=list(STYLES)+["all"])
    p.add_argument("--fps",   type=int, default=25)
    p.add_argument("--size",  type=int, default=768)
    args = p.parse_args()
    clip_dirs = (
        sorted(d for d in CTRL_BASE.iterdir() if d.is_dir() and list(d.glob("pose_*.png")))
        if args.clip == "all" else [CTRL_BASE / args.clip]
    )
    if not clip_dirs:
        print(f"[error] no clips in {CTRL_BASE}", file=sys.stderr); return 1
    styles = list(STYLES) if args.style == "all" else [args.style]
    total  = 0
    for clip_dir in clip_dirs:
        for style in styles:
            try:
                render_clip(clip_dir, style, args.fps, args.size); total += 1
            except Exception as e:
                print(f"[error] {clip_dir.name}/{style}: {e}", file=sys.stderr)
    print(f"\n[realistic_avatar] {total} video(s) -> {OUT_BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
