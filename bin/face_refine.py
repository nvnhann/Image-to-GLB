"""Chiếu khuôn mặt từ ảnh gốc (độ phân giải cao) lên texture của Hunyuan3D-2.1.

Bước vẽ texture của Hunyuan sinh mỗi góc nhìn cả người ở 512-768px nên khuôn mặt chỉ còn ~40px.
Ảnh gốc là ảnh chính diện, nên ta căn ảnh gốc khớp với bóng của mesh ở góc nhìn trước (azim 0)
rồi chiếu vùng mặt (phát hiện bằng OpenCV) thẳng vào UV, làm mềm biên để không lộ đường nối.

Dùng (chạy từ thư mục Hunyuan3D-2.1): python face_refine.py <thư mục out> [--strength 1.0] [--grow 1.0]
Ghi đè textured.jpg + textured.glb (bản gốc lưu thành textured_raw.jpg), xuất face_debug.png để so sánh.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, "./hy3dpaint")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from DifferentiableRenderer.MeshRender import MeshRender
from DifferentiableRenderer.camera_utils import get_mv_matrix, transform_pos
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb

RES = 2048  # độ phân giải render ở góc chính diện

p = argparse.ArgumentParser()
p.add_argument("outdir")
p.add_argument("--strength", type=float, default=1.0, help="mức thay bằng ảnh gốc ở tâm mặt (0-1)")
p.add_argument("--grow", type=float, default=1.0, help="phóng to vùng mặt được chiếu")
args = p.parse_args()
d = args.outdir
obj_path, tex_path, raw_path = (os.path.join(d, f) for f in ("textured.obj", "textured.jpg", "textured_raw.jpg"))
if not os.path.exists(raw_path):
    shutil.copy(tex_path, raw_path)  # chạy lại nhiều lần vẫn bắt đầu từ texture gốc

render = MeshRender(default_resolution=RES, texture_size=4096, bake_mode="back_sample", raster_mode="cr")
render.load_mesh(trimesh.load(obj_path, force="mesh", process=False))
tex = torch.from_numpy(np.asarray(Image.open(raw_path).convert("RGB")) / 255.0).float().cuda()


def render_view(texture, elev=0, azim=0):
    """Render màu (texture, không chiếu sáng) và alpha ở góc nhìn cho trước."""
    mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=render.camera_distance)
    clip = transform_pos(render.camera_proj_mat, transform_pos(mv, render.vtx_pos, keepdim=True))
    rast, _ = render.raster_rasterize(clip, render.pos_idx, resolution=(RES, RES))
    uv, _ = render.raster_interpolate(render.vtx_uv[None], rast, render.uv_idx)
    grid = uv[0] * 2 - 1
    col = F.grid_sample(texture.permute(2, 0, 1)[None], grid[None], align_corners=False)[0].permute(1, 2, 0)
    alpha = (rast[0, ..., -1:] > 0).float()
    cam = transform_pos(mv, render.vtx_pos, keepdim=True)
    depth, _ = render.raster_interpolate(cam[None, :, 2:3].contiguous(), rast, render.pos_idx)
    return col * alpha, alpha[..., 0], depth[0, ..., 0], cam[:, 0]


# 1. Căn ảnh gốc khớp với bóng mesh ở góc chính diện
src = np.asarray(Image.open(os.path.join(d, "input_rgba.png")).convert("RGBA")).astype(np.float32) / 255
_, mesh_alpha, depth, cam_x = render_view(tex)
mesh_alpha, depth = mesh_alpha.cpu().numpy(), depth.cpu().numpy()


def bbox(mask):
    ys, xs = np.nonzero(mask > 0.5)
    return xs.min(), ys.min(), xs.max(), ys.max()


ix0, iy0, ix1, iy1 = bbox(src[..., 3])
rx0, ry0, rx1, ry1 = bbox(mesh_alpha)
s0 = (ry1 - ry0) / (iy1 - iy0)
best = (-1, None)
small = cv2.resize(mesh_alpha, (512, 512)) > 0.5
for ds in np.linspace(0.97, 1.03, 7):  # tinh chỉnh scale/dịch chuyển theo IoU của bóng
    s = s0 * ds
    for dx in range(-24, 25, 4):
        for dy in range(-24, 25, 4):
            tx = (rx0 + rx1) / 2 - s * (ix0 + ix1) / 2 + dx
            ty = (ry0 + ry1) / 2 - s * (iy0 + iy1) / 2 + dy
            M = np.float32([[s, 0, tx], [0, s, ty]])
            a = cv2.resize(cv2.warpAffine(src[..., 3], M, (RES, RES)), (512, 512)) > 0.5
            iou = (a & small).sum() / max((a | small).sum(), 1)
            if iou > best[0]:
                best = (iou, M)
iou, M = best
print(f"căn ảnh: IoU bóng = {iou:.3f}")
if iou < 0.8:
    print("Cảnh báo: ảnh gốc và mesh lệch nhiều, kết quả chiếu mặt có thể sai lệch")
warped = cv2.warpAffine(src, M, (RES, RES), flags=cv2.INTER_CUBIC)

# 2. Phát hiện khuôn mặt trong ảnh gốc, tạo mặt nạ elip làm mềm biên
gray = cv2.cvtColor((src[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
faces = []
for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
    faces = [f for f in cascade.detectMultiScale(gray, 1.05, 4, minSize=(40, 40)) if f[1] < iy0 + 0.35 * (iy1 - iy0)]
    if len(faces):
        break
if not len(faces):
    sys.exit("Không phát hiện được khuôn mặt trong ảnh gốc")
fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
print(f"khuôn mặt trong ảnh gốc: x={fx} y={fy} w={fw} h={fh}")
mask = np.zeros(gray.shape, np.float32)
center = (int(fx + fw / 2), int(fy + fh * 0.52))
axes = (int(fw * 0.62 * args.grow), int(fh * 0.78 * args.grow))
cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
mask = cv2.GaussianBlur(mask, (0, 0), fw * 0.12) * src[..., 3] * args.strength
mask_w = cv2.warpAffine(mask, M, (RES, RES), flags=cv2.INTER_LINEAR)

# Bỏ vật thể chắn trước mặt (vd. vũ khí): chỉ giữ bề mặt có độ sâu gần với khuôn mặt
px_per_unit = (rx1 - rx0) / float(cam_x.max() - cam_x.min())
face_depth = np.median(depth[(mask_w > 0.5) & (mesh_alpha > 0.5)])
near = np.abs(depth - face_depth) < 0.6 * fw * M[0, 0] / px_per_unit
mask_w *= cv2.GaussianBlur(near.astype(np.float32), (0, 0), 3)

# 3. Chiếu (RGB + mặt nạ) vào UV và trộn với texture gốc
proj, cos_map, _ = render.back_project(np.dstack([warped[..., :3], mask_w]), 0, 0)
facing = torch.clamp((cos_map - 0.35) / 0.35, 0, 1)  # bỏ vùng nghiêng (má bên, tai) để tránh vệt kéo dài
w = torch.clamp(proj[..., 3:4], 0, 1) * facing
new_tex = tex * (1 - w) + proj[..., :3] * w
cv2.imwrite(tex_path, (new_tex.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
print(f"đã thay {int((w > 0.01).sum())} texel")

# 4. Ảnh so sánh vùng mặt trước/sau (góc chính diện) + xuất lại GLB
before = render_view(tex)[0]
after = render_view(new_tex)[0]
x0, y0, x1, y1 = cv2.transform(np.float32([[[fx - fw, fy - fh], [fx + 2 * fw, fy + 2 * fh]]]), M)[0].astype(int).ravel()
x0, y0 = max(x0, 0), max(y0, 0)
crop = lambda t: (t[y0:y1, x0:x1].cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(np.hstack([crop(before), crop(after), (warped[y0:y1, x0:x1, :3] * 255).astype(np.uint8)])).save(
    os.path.join(d, "face_debug.png")
)
convert_obj_to_glb(obj_path, obj_path.replace(".obj", ".glb"))
print("xong:", obj_path.replace(".obj", ".glb"))
sys.stdout.flush()
os._exit(0)  # bpy hay segfault khi thoát
