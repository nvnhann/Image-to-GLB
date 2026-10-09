# Ghi chú Image-to-GLB

Chuyển một ảnh nhân vật thành model 3D `.glb`. Máy thử: A100 40GB, CUDA 13.0 (nvcc 13.0), Python mặc định 3.13.

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `bin/setup-crm.sh`, `bin/run-crm.sh` | Cài và chạy CRM (thu-ml/CRM), venv `.venv` (Python 3.10) |
| `bin/setup-hunyuan.sh`, `bin/run-hunyuan.sh`, `bin/hunyuan_gen.py` | Cài và chạy Hunyuan3D-2.1, venv `.venv-hy` (Python 3.11) |
| `bin/face_refine.py` | Chiếu khuôn mặt từ ảnh gốc lên texture của Hunyuan |
| `viewer/index.html` | Viewer three.js, có ô chọn model, kéo-thả `.glb` |
| `IMG.webp`, `image/img_1.webp` | Ảnh thử (1120×2240) |
| `out/` | Kết quả (đã gitignore) |

`CRM/` và `Hunyuan3D-2.1/` là mã nguồn clone về, pin commit trong script setup. `.venv*` và `out/` không commit.

## Cách dùng

```bash
bash bin/setup-hunyuan.sh                    # cài một lần (tải trọng số vài GB)
bash bin/run-hunyuan.sh image/img_1.webp out/img_1 --octree 512 --seed 1
cd Hunyuan3D-2.1 && ../.venv-hy/bin/python ../bin/face_refine.py ../out/img_1   # sửa mặt (tuỳ chọn)

python -m http.server 8000                   # xem: http://localhost:8000/viewer/
```

Tuỳ chọn của `run-hunyuan.sh`: `--steps 50 --guidance 5.0 --octree 384 --views 8 --tex-res 768 --faces 150000 --seed 1234 --no-texture`.
Kết quả: `out/<tên>/textured.glb` (bản `shape.glb` là hình khối chưa có texture).

Viewer phải chạy qua HTTP, mở bằng `file://` sẽ bị trình duyệt chặn.

## So sánh model (cùng ảnh `IMG.webp`)

| | CRM | Hunyuan3D-2.1 |
|---|---|---|
| Mesh cuối | 15,7k mặt | 150k mặt |
| Texture | 1024², mờ | 4096², có metallic và roughness |
| Thời gian | khoảng 2 phút | khoảng 5 phút |

CRM sinh 6 góc nhìn ở 256² nên mặt vỡ và đầu bị nhân đôi. Hunyuan tốt hơn hẳn. Giấy phép Hunyuan là Tencent Non-Commercial, dùng thương mại phải xin phép.

## Lỗi đã gặp khi cài và cách xử lý (đã nằm trong script)

**CRM**
- Thư mục `CRM` trong git là gitlink rỗng, không có `.gitmodules`: script kiểm tra `CRM/.git` rồi clone và checkout `4964e36`.
- `requirements.txt` thiếu: `absl-py`, `accelerate`, `wandb`, `onnxruntime`.
- `huggingface_hub<0.26` (diffusers 0.24 cần `cached_download`), `kiui==0.2.14` (0.3.5 lỗi `Union`).
- ImageDream bắt buộc có `xformers`: cài `xformers==0.0.33.post2` từ index cu130, khớp torch 2.9.1.
- Repo `stabilityai/stable-diffusion-2-1-base` đã bị gỡ khỏi HuggingFace: patch `model/crm/model.py` để dựng `DDIMScheduler` trực tiếp bằng config gốc.
- `onnxruntime-gpu` cần CUDA 12, đổi sang `onnxruntime` bản CPU (chỉ dùng tách nền).

**Hunyuan3D-2.1**
- `bpy==4.0` đã bị gỡ khỏi PyPI: dùng Python 3.11 và `bpy==4.2.0`.
- `compile_mesh_painter.sh` ra file thiếu đuôi `.so` vì venv của uv không có `python3-config`: biên dịch bằng `sysconfig`.
- `bpy` hay segfault lúc Python thoát: script kết thúc bằng `os._exit(0)` sau khi ghi xong file.
- Bỏ `cupy`, `deepspeed`, `gradio`, `fastapi`, `uvicorn` và các mirror pip Trung Quốc khỏi requirements.

## Bài học chất lượng

- **Bước vẽ texture resize ảnh tham chiếu về 512×512**, nên ảnh dọc 1:2 bị bóp méo và mặt vỡ. `hunyuan_gen.py` đã cắt sát nhân vật và pad thành ảnh vuông.
- Mặc định của Hunyuan giảm mesh còn 40k mặt và thu texture còn 2048². Script giữ 150k mặt và texture 4096².
- **Vật dài, mảnh chắn người** (giáo, bút lông) làm model dựng sai độ sâu, cắt ngang mặt. Đổi seed không sửa được (đã thử 4 seed). Cách duy nhất: bỏ vật đó khỏi ảnh đầu vào, hoặc sinh riêng rồi ghép trong Blender.
- **Tấm sàn sinh nhầm:** đôi khi model sinh thêm một tấm sàn 2×2 dính dưới chân, làm nhân vật co còn 45% chiều cao. `remove_ground_slab` trong `hunyuan_gen.py` tự phát hiện và xoá. Gặp ở `img_1.webp` với seed 1234; seed 1, 2, 3 không bị.
- **`face_refine.py`** chiếu mặt từ ảnh gốc (độ phân giải cao) lên texture ở góc chính diện, bỏ qua vật chắn phía trước mặt dựa trên độ sâu. Nó làm mặt rõ hẳn khi nhìn chính diện và ±35°. File `face_debug.png` trong thư mục kết quả để kiểm tra.
- Căn ảnh gốc khớp mesh bằng IoU bóng. Dưới 0,8 thì script cảnh báo kết quả có thể lệch.

## Giới hạn còn lại

Ở góc nghiêng 60–90°, mặt vẫn xấu: hình khối do Hunyuan sinh gần như phẳng và trơn (không có hốc mắt, sống mũi), và ảnh chính diện không chứa thông tin về mặt nghiêng. Mắt anime là chi tiết vẽ, không có cấu trúc 3D tương ứng. Đổi seed hay chỉnh texture không khắc phục được.

## Yêu cầu ảnh đầu vào tốt

- Chính diện, tư thế chữ A, tay không chạm người, không vật dài chắn người.
- Khuôn mặt đủ lớn: ảnh toàn thân nên cao từ 2048px, hoặc dùng ảnh nửa người.
- Nền trơn hoặc PNG trong suốt, ánh sáng đều, không bóng gắt, không glow hay khói.
- Prompt gợi ý khi tạo ảnh bằng AI: *"full body, front view, A-pose, arms away from body, plain white background, even soft lighting, no weapon, character turnaround reference"*.

## Việc có thể làm tiếp

1. **Hunyuan3D-2mv (đa góc nhìn)**, nằm ở repo Hunyuan3D-2: đưa vào ảnh chính diện, nghiêng 90° và sau lưng của cùng nhân vật để dựng đúng hình khối đầu và mặt. Mở rộng `face_refine` để chiếu thêm ảnh mặt nghiêng. Ảnh phải thống nhất: cùng nhân vật, trang phục, tư thế A, kích thước, góc nhìn thẳng không phối cảnh. Tạo bằng AI với prompt "character turnaround sheet, front/side/back view, same character, orthographic".
2. Thử dịch vụ thương mại (Tripo3D, Meshy, Rodin) với cùng ảnh để so sánh. Viewer mở được GLB tải về bằng cách kéo-thả.
3. Sửa thủ công mặt trong Blender nếu cần chất lượng cao.
4. Cho `run-hunyuan.sh` tự gọi `face_refine.py` sau khi sinh.
5. Retopology mesh trước khi dùng cho rigging hoặc animation (mesh hiện tại là lưới tự sinh).

## Lưu ý bảo mật

Remote `origin` từng chứa GitHub token dạng plaintext trong URL. Cần revoke token và dùng URL không có token (`git remote set-url origin https://github.com/nvnhann/Image-to-GLB.git`) kèm credential helper hoặc `gh auth login`.
