"""Ảnh -> GLB bằng Hunyuan3D-2.1 (shape + texture PBR). Chạy từ thư mục Hunyuan3D-2.1."""
import argparse
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
import textureGenPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from utils.simplify_mesh_utils import mesh_simplify_trimesh
from torchvision_fix import apply_fix

apply_fix()

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("outdir")
p.add_argument("--steps", type=int, default=50, help="số bước diffusion cho shape")
p.add_argument("--guidance", type=float, default=5.0, help="độ bám theo ảnh của shape")
p.add_argument("--octree", type=int, default=384, help="độ phân giải lưới (256/384/512)")
p.add_argument("--views", type=int, default=8, help="số góc nhìn khi vẽ texture (6-9)")
p.add_argument("--tex-res", type=int, default=768, help="độ phân giải mỗi góc nhìn texture (512/768)")
p.add_argument("--faces", type=int, default=150000, help="số mặt của mesh trước khi vẽ texture (gốc: 40000)")
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--no-texture", action="store_true", help="chỉ sinh hình khối")
args = p.parse_args()
os.makedirs(args.outdir, exist_ok=True)

image = Image.open(args.image).convert("RGBA")
# Ảnh không có nền trong suốt thật (alpha toàn 255) -> tách nền
if image.getextrema()[3][0] == 255:
    image = BackgroundRemover()(image.convert("RGB"))
# Cắt sát nhân vật rồi pad thành ảnh vuông: bước texture resize thẳng ảnh về 512x512,
# ảnh dọc 1:2 sẽ bị bóp méo và khuôn mặt mất chi tiết
x0, y0, x1, y1 = image.getchannel("A").getbbox()
side = int(max(x1 - x0, y1 - y0) * 1.1)
square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
square.paste(image.crop((x0, y0, x1, y1)), ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
image = square.resize((1024, 1024), Image.LANCZOS) if side < 1024 else square
image_path = os.path.join(args.outdir, "input_rgba.png")
image.save(image_path)

shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
mesh = shape(
    image=image,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
    octree_resolution=args.octree,
    generator=torch.manual_seed(args.seed),
)[0]
shape_path = os.path.join(args.outdir, "shape.glb")
mesh.export(shape_path)
print("shape:", shape_path)

if not args.no_texture:
    del shape
    torch.cuda.empty_cache()
    conf = Hunyuan3DPaintConfig(args.views, args.tex_res)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    # Giữ nhiều mặt hơn (gốc giảm còn 40k) và lưu texture 4096 đầy đủ (gốc thu nhỏ còn 2048)
    textureGenPipeline.remesh_mesh = lambda src, dst: mesh_simplify_trimesh(src, dst, target_count=args.faces)
    paint = Hunyuan3DPaintPipeline(conf)
    save_mesh = paint.render.save_mesh
    paint.render.save_mesh = lambda path, downsample=False: save_mesh(path, downsample=False)
    obj = paint(
        mesh_path=shape_path,
        image_path=image_path,
        output_mesh_path=os.path.join(args.outdir, "textured.obj"),
    )
    print("textured:", obj.replace(".obj", ".glb"))

# bpy hay segfault khi Python dọn dẹp lúc thoát -> thoát thẳng sau khi đã ghi xong file
sys.stdout.flush()
os._exit(0)
