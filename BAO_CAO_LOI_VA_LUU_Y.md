# Báo cáo lỗi và lưu ý kỹ thuật — Lab Day 10

Học viên: **Nguyễn Đăng Thực** · MSHV **2A202603014**

Các bẫy dưới đây là những lỗi em **gặp thật** khi làm bài (hoặc phát hiện bằng kiểm tra trước khi tốn GPU), kèm triệu chứng, nguyên nhân gốc và cách sửa. Môi trường chạy: Kaggle GPU T4 x2 (15 GB/GPU), Python 3.13, torch 2.11.0+cu128, transformers 5.19.0, peft 0.21.2, trl 1.9.2, bitsandbytes 0.50.2.

## 1. `llm_int8_skip_modules=["visual", "lm_head"]` không còn giữ ViT ở fp16 (transformers 5.19)

- **Triệu chứng.** Smoke test E2/E3: số tham số projector huấn luyện được chỉ **54,579,200** thay vì **109,105,152** (đúng một nửa), và GradScaler báo `ValueError: Attempting to unscale FP16 gradients.`
- **Nguyên nhân gốc.** transformers 5.x quyết định có lượng tử hoá một module hay không bằng `should_convert_module(full_name, patterns)`, so khớp `re.match(pattern, full_name)` **neo ở đầu** tên đầy đủ, hoặc `full_name.endswith(pattern)`. Tên thật của ViT là `model.visual.blocks.N...` và `model.visual.merger...`, nên `"visual"` không khớp. Kết quả: **104 Linear của ViT (96 trong các block + 8 của merger/deepstack) bị NF4 mà không có cảnh báo nào**. `lm_head` vẫn được bỏ qua vì khớp chính xác tên. `Params4bit` lưu 2 trọng số/byte nên `numel()` chỉ bằng một nửa, và `.to(float32)` trên tensor `uint8` đã lượng tử chỉ cho ra rác.
- **Cách sửa.** `llm_int8_skip_modules=["visual", "model.visual", "lm_head"]`, thêm `assert_vit_not_quantized()` ngay sau khi nạp model. `python -m lab.checks` in ra số Linear của ViT bị 4-bit với từng danh sách skip (104 → 0), xem `report/checks.json`.
- **Bài học.** Phải đếm tham số huấn luyện và so với quy mô kỳ vọng (merger 27.3M + deepstack 81.8M = 109.1M). Một con số "lệch đúng một nửa" là dấu hiệu lượng tử hoá.

## 2. PEFT lỗi vì torchao cũ có sẵn trên Kaggle

- **Triệu chứng.** `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`, xuất hiện khi gắn LoRA lên `nn.Linear` thường (ViT fp16 ở E3).
- **Nguyên nhân.** PEFT duyệt qua các dispatcher; với Linear không lượng tử, nó gọi `is_torchao_available()`, và hàm này **ném lỗi** (chứ không trả `False`) khi torchao quá cũ.
- **Cách sửa.** `pip uninstall -y torchao` trong kernel (bài không dùng torchao).

## 3. TRL `GRPOTrainer` ép adapter QLoRA sang bf16

- **Quan sát khi đọc mã TRL 1.9.2.** Nếu model đã lượng tử hoá, trainer chạy `param.data = param.data.to(torch.bfloat16)` cho mọi tham số huấn luyện.
- **Vấn đề trên T4.** T4 (sm75) không có bf16 gốc. Quan trọng hơn: với `learning_rate=5e-6`, bước cập nhật AdamW (~lr) nhỏ hơn độ phân giải tương đối của bf16 (2⁻⁸ ≈ 3.9e-3 so với độ lớn trọng số LoRA ~1e-2), nên **phần lớn cập nhật bị làm tròn mất**.
- **Cách sửa.** Sau khi khởi tạo trainer, đưa tham số `requires_grad` về fp32 và dùng `fp16=True, bf16=False` (forward fp16, master weight fp32).

## 4. Lưu projector riêng (E2/E3)

`model.save_pretrained()` của PEFT chỉ lưu adapter LoRA. Projector được mở full nên phải `torch.save` riêng thành `projector.pt` (ở mỗi lần lưu, 100 bước một lần), rồi khi suy luận nạp lại bằng `copy_` vào đúng tên tham số (`load_trained()` assert không có tên lạ). Nếu quên, E2/E3 sẽ suy luận bằng projector gốc và kết quả trông giống E1.

## 5. Box vượt mép ảnh trong nhãn YOLO

63/4.737 box có `xc ± w/2` hoặc `yc ± h/2` nằm ngoài [0, 1] (tới 1015.6 trên thang 0–1000). `yolo_to_box` kẹp về [0, 1000], còn `score.py` thì không kẹp GT. Vì vậy box dự đoán đúng hoàn toàn vẫn có IoU < 1 với các GT này (ảnh hưởng nhỏ, IoU vẫn ≫ 0.5). Sai số quy đổi ngược 0–1000 → pixel với các box còn lại ≤ 0.32 px (`python -m lab.data_check`).

## 6. Token kết thúc trong nhãn

Chat template của Qwen3-VL thêm `<|im_end|>\n` sau câu trả lời. Em giữ `<|im_end|>` trong labels (để model học dừng, tránh lặp tới `max_new_tokens`) nhưng mask token `\n` phía sau. Kiểm tra: giải mã các token có `labels != -100` ra đúng `JSON + <|im_end|>`.

## 7. Lưu ý khác

- Qwen3-VL dùng patch **16** (không phải 14) và merge 2×2: ảnh 640×360 được `smart_resize` về 640×352 → lưới 40×22 patch → 880 patch → **220 token ảnh**.
- Mỗi chữ số trong toạ độ là **một token** riêng (tokenizer Qwen tách số), nên một box tốn khoảng 27.7 token. Đáp án test dài nhất là 196 token, nên `max_new_tokens=256` là đủ. Với GRPO (`max_completion_length=128`), em chỉ dùng ảnh train có ≤ 4 box để đáp án đúng không bị cắt cụt.
- Windows: đặt `PYTHONIOENCODING=utf-8` trước khi chạy `scripts/*.py` (in tiếng Việt).
- Không commit `data/`, `work/`, `.env`, `*.safetensors`, `*.pt`. Adapter và projector chỉ nằm trong output kernel Kaggle và `work/` (gitignore).
