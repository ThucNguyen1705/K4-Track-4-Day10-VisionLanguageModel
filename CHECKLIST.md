# Checklist tiến độ — Lab Day 10

Học viên: **Nguyễn Đăng Thực** · MSHV **2A202603014**

## Chuẩn bị
- [x] Fork repo, clone về máy, `python -m pytest -q` → **68 passed, 1 skipped**
- [x] Khoá Kaggle chỉ nằm trong `.env` (gitignore), không có trong notebook/mã
- [x] GPU: Kaggle T4 x2 qua `tools/kaggle_job.py` (kernel private, mỗi thí nghiệm 1 T4)

## Chặng 1 — Dữ liệu
- [x] `prepare_data.py --kaggle` → `dataset_version = d10-21e4070301`
- [x] train/val/test = 1.422 / 276 / 311 ảnh, số box từng lớp khớp bảng của đề
- [x] Không phiên quay nào nằm ở hai split (13 / 2 / 3 phiên)
- [x] 6 ảnh có box gốc: `report/figures/gt_samples_train.jpg`
- [x] Commit #1

## Chặng 2 — Nhãn JSON và E0
- [x] `yolo_to_box`, `objects_to_answer` (sắp theo y1, x1), quy đổi ngược sai số ≤ 0.32 px
- [x] E0 zero-shot trên 311 ảnh test → `report/evidence/e0.json`
- [x] Commit #2

## Chặng 3 — Chat template và mask loss
- [x] 640×360 → `image_grid_thw=[1,22,40]` → 220 token ảnh
- [x] Giải mã `labels != -100` = JSON đáp án + `<|im_end|>`
- [x] Collate: pad token, nối `pixel_values`/`image_grid_thw` theo chiều 0
- [x] Commit #3

## Chặng 4 — E1/E2/E3
- [x] 252 module LoRA (E1/E2), 348 (E3); projector 109.1M tham số ở float32
- [x] Sửa `llm_int8_skip_modules` cho transformers 5.19 (ViT giữ fp16)
- [x] Lưu riêng `projector.pt`, nạp lại khi suy luận
- [x] `e1.json`–`e3.json` có `meta` (trainable_params, peak_vram_gb, train_minutes)
- [x] E1: `parse_valid_rate ≥ 0.95`, F1(E1) − F1(E0) ≥ 0.10
- [x] Commit #4

## Chặng 5 — E4
- [x] Chỉ đổi một biến (`data_size`: 0.1 và 1.0), chạy cả hai giá trị
- [x] `report/evidence/e4_data_0.1.json`, `e4_data_1.0.json`
- [x] Commit #5

## Chặng 6 — E5 (tuỳ chọn)
- [x] Reward IoU + 11 unit test (`python -m pytest lab/tests -q`)
- [x] GRPO 60 bước bằng `trl.GRPOTrainer`, log reward và dấu hiệu hacking
- [x] `report/evidence/e5.json`, so với SFT kèm CI95

## Nộp bài
- [x] `REPORT.md` điền đủ, không còn `<điền`
- [x] ≥ 6 ảnh minh hoạ đúng/sai có box dự đoán và box đúng (`report/figures/`)
- [x] `python scripts/check_submission.py` → 0 FAIL
- [x] Không track `data/`, `work/`, `.env`, `*.pt`, `*.safetensors`
- [x] ≥ 5 commit, push lên GitHub (public)
