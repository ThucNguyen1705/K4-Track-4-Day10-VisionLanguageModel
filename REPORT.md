Họ tên: Nguyễn Đăng Thực — MSHV 2A202603014
`dataset_version`: d10-21e4070301

> Môi trường: Kaggle GPU **T4 x2** (15 GB/GPU; mỗi thí nghiệm chạy trên **một** T4, hai thí nghiệm song song), Python 3.13, torch 2.11.0+cu128, transformers 5.19.0, peft 0.21.2, trl 1.9.2, bitsandbytes 0.50.2. Mã: `lab/` (gọi từ `lab_day10.ipynb`), kernel chạy bằng `tools/kaggle_job.py`. VRAM đỉnh = `torch.cuda.max_memory_allocated()` từ sau khi nạp model đến hết huấn luyện; "Phút" = thời gian vòng huấn luyện 300 bước (không gồm nạp model và suy luận). Mọi số dưới đây lấy từ `report/evidence/*.json` (bảng sinh bằng `python -m lab.summarize`), log ở `report/logs/`.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – (chỉ suy luận) | – | 0.088 | [0.062, 0.115] | 0.91 | 0.109 | 0.036 | 0.179 |
| E1 | LoRA LLM | 33,030,144 | 6.23 | 43.6 | 0.282 | [0.244, 0.322] | 1.11 | 0.207 | 0.204 | 0.515 |
| E2 | + projector | 142,135,296 | 7.83 | 39.9 | 0.369 | [0.325, 0.412] | 0.94 | 0.232 | 0.296 | 0.618 |
| E3 | + LoRA ViT | 148,426,752 | 8.01 | 48.1 | **0.420** | [0.379, 0.463] | 0.92 | 0.333 | 0.325 | 0.683 |
| E4 | data_size = 0.1 (E1 với 142 ảnh) | 33,030,144 | 5.80 | 43.1 | 0.183 | [0.152, 0.216] | 1.96 | 0.123 | 0.106 | 0.433 |
| E4 | data_size = 1.0 (= run E1) | 33,030,144 | 6.23 | 43.6 | 0.282 | [0.244, 0.322] | 1.11 | 0.207 | 0.204 | 0.515 |
| E5 | GRPO reward IoU (từ E1r) | 33,030,144 | 5.31 | 24.0 | 0.303 | [0.262, 0.346] | 0.90 | 0.247 | 0.216 | 0.533 |

| Mã | parse_valid_rate | Precision | Recall | TP | FP | FN | mean IoU (cặp khớp) |
|---|---|---|---|---|---|---|---|
| E0 | 0.839 | 0.124 | 0.069 | 40 | 283 | 543 | 0.600 |
| E1 | 0.990 | 0.306 | 0.261 | 152 | 344 | 431 | 0.679 |
| E2 | 0.997 | 0.403 | 0.340 | 198 | 293 | 385 | 0.680 |
| E3 | 1.000 | 0.447 | 0.396 | 231 | 286 | 352 | 0.700 |
| E4 (0.1) | 1.000 | 0.164 | 0.206 | 120 | 611 | 463 | 0.663 |
| E5 | 1.000 | 0.355 | 0.264 | 154 | 280 | 429 | 0.684 |

Bootstrap **ghép cặp** hiệu F1 (2.000 lần, cùng tập ảnh resample cho cả hai setup; `python -m lab.paired_bootstrap` → `report/analysis/paired_bootstrap.json`):

| So sánh | ΔF1 | CI95 của Δ | P(Δ ≤ 0) |
|---|---|---|---|
| E2 − E1 | +0.087 | [+0.048, +0.126] | 0.000 |
| E3 − E2 | +0.051 | [+0.008, +0.097] | 0.011 |
| E3 − E1 | +0.138 | [+0.094, +0.181] | 0.000 |
| E4: 1.0 − 0.1 | +0.099 | [+0.064, +0.136] | 0.000 |
| E5 − E1r (GRPO so với chính adapter khởi tạo) | +0.010 | [−0.005, +0.025] | 0.106 |
| E1r − E1 (chạy lại E1 y hệt cấu hình, kernel khác) | +0.012 | [−0.014, +0.036] | 0.199 |

Ảnh minh hoạ (khung màu theo lớp = GT: pothole đỏ, crack vàng, manhole xanh dương; khung xanh lá "P:" = dự đoán):
- `report/figures/gt_samples_train.jpg`: 6 ảnh train với box gốc (kiểm tra nhãn).
- `report/figures/e0_loi.jpg`: 2 lỗi điển hình của E0 (box dải ngang toàn khung; JSON bị cắt do lặp).
- `report/figures/e3_dung.jpg`: E3 đúng hoàn toàn (2 ảnh).
- `report/figures/e3_bo_sot.jpg`: E3 bỏ sót (2 ảnh, FN ≥ 2, TP = 0).
- `report/figures/e3_du_thua.jpg`: E3 dư box / sai lớp (2 ảnh).
- `report/figures/loss_sft.png`: loss E1–E4; `report/figures/grpo_reward.png`: reward GRPO.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Em đếm trên cây model thật (`python -m lab.checks` → `report/checks.json`):

| Khối | Tham số | Ghi chú |
|---|---|---|
| ViT `model.visual` (24 block, patch 16, hidden 1024) | 415,347,712 | gồm cả projector bên dưới |
| ↳ projector `visual.merger` | 27,271,680 | gộp 2×2 patch rồi MLP sang 2560 chiều của LLM |
| ↳ projector `visual.deepstack_merger_list` (3 merger) | 81,833,472 | DeepStack lấy đặc trưng ở block 5, 11, 17 (`deepstack_visual_indexes`) |
| ↳ phần còn lại của ViT (patch embed + 24 block) | 306,242,560 | |
| LLM `model.language_model` (36 lớp, gồm embedding) | 4,022,468,096 | `lm_head` buộc chung với embedding (`tie_word_embeddings=True`) |
| **Tổng** | **≈ 4.44B** | 4,022.5M + 415.3M |

Luồng: ảnh 640×360 → `smart_resize` 640×352 → 40×22 = 880 patch 16×16 → 24 block ViT → merger gộp 2×2 → **220 token** 2560 chiều chèn vào vị trí `<|image_pad|>`. Đồng thời 3 deepstack-merger đưa đặc trưng của block 5/11/17 cộng vào hidden state của 3 lớp đầu LLM. Sau đó LLM sinh JSON.

- **E1** chỉ cập nhật LoRA r=16 trên 252 Linear của LLM (q/k/v/o/gate/up/down × 36 lớp) = **33,030,144** tham số. ViT và projector đóng băng ở fp16, trọng số LLM gốc đóng băng ở NF4.
- **E2** = E1 + mở full projector (merger + deepstack = 109,105,152 tham số, ở float32) = **142,135,296**.
- **E3** = E2 + LoRA r=16 trên 96 Linear của ViT (`attn.qkv`, `attn.proj`, `mlp.linear_fc1/fc2` × 24 block, +6,291,456) = **148,426,752**.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate` của E0 = **0.839**: 50/311 output hỏng. Model gốc thực ra đã khá nghe lời về định dạng: không có output nào bọc ```json hay kèm câu dẫn, và 16 ảnh trả `[]`. Nhưng F1 chỉ 0.088 (TP 40, FP 283, FN 543). Phân tích bằng `python -m lab.analyze` (`report/analysis/e0.json`) cho thấy hai lỗi chính:

1. **Sinh lặp vô hạn đến hết `max_new_tokens=256` nên JSON bị cắt (hỏng).** Model rải một chuỗi box crack sát nhau theo một nhịp đều cho tới khi hết token. Ví dụ thô (ảnh `vlcsnap-2025-02-18-23h07m53s515`, 413 ký tự, phần cuối):
   `... {"label": "crack", "bbox_2d": [600, 350, 620, 360]}, {"label": "crack", "bbox_2d": [620, 350, 640, 360]}, {"label": "crack", "bbox_2d": [640, 350, `
   Toàn bộ 50 output hỏng đều thuộc kiểu này.
2. **Bịa box / định vị sai, nhất là "box dải ngang toàn khung".** Trong 283 FP có 135 box bịa (không chồng GT nào), 124 box lệch vị trí (0.1 ≤ IoU < 0.5 với GT cùng lớp), 17 box khổng lồ ≥ 25% khung, 7 box sai lớp. Có 48 box rộng ≥ 900/1000, ví dụ `vlcsnap-00207`: `[{"label": "crack", "bbox_2d": [0, 650, 1000, 700]}]`, tức một vạch ngang cả mặt đường thay vì từng vết nứt. Hệ quả là crack chỉ đạt F1 0.036 (recall crack lớn 8/148, vừa 1/136).

Ngoài ra model rất thiên về crack (203 box crack, 94 manhole, 26 pothole) và bỏ sót gần hết vật thể nhỏ (pothole < 0.5% khung: recall 1/68). Ảnh: `report/figures/e0_loi.jpg`.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Lệch thang làm F1 rơi về **0**. Em kiểm chứng bằng chính trọng tài `scripts/score.py`:

| Thí nghiệm (file trong `report/analysis/`) | F1 |
|---|---|
| E3 (model xuất 0–1000), chấm đúng `--coord norm1000` | 0.420 |
| Cùng file E3, chấm nhầm `--coord pixel` (`e3_scored_as_pixel.json`) | **0.000** (P = R = 0, FP/ảnh 1.66) |
| Đổi output E3 sang pixel (x·0.64, y·0.36) rồi chấm như 0–1000 (`e3_pixel_values_scored_norm1000.json`) | **0.000** |
| Cùng file pixel đó, chấm đúng `--coord pixel` (`e3_pixel_values_scored_pixel.json`) | 0.420 (mean IoU 0.701) |
| E0 chấm nhầm `--coord pixel` (`e0_scored_as_pixel.json`) | 0.000 |

Lý do: `score.py` đưa mọi thứ về 0–1000. Nếu số 0–1000 bị hiểu là pixel, x bị nhân 1000/640 = 1.5625 và y nhân 1000/360 = 2.78. Box bị phóng to và dời xuống dưới không đều theo hai trục (box ở y = 500 dời tới y ≈ 1389, ra ngoài khung). Với box có diện tích trung vị chỉ 0.4–2% khung, dịch vài chục đơn vị đã đưa IoU về 0, nên không còn cặp nào đạt ngưỡng 0.5. Chiều ngược lại cũng vậy: box co về góc trên-trái.

Vì sao Qwen3-VL dùng 0–1000: toạ độ tương đối không phụ thuộc độ phân giải (ảnh bị `smart_resize` 640×360 → 640×352 trước khi vào ViT, nên "pixel" model nhìn thấy khác pixel gốc), và miền giá trị cố định dùng chung cho mọi ảnh khi pretrain grounding. Em cũng kiểm tra phía nhãn: quy đổi YOLO → 0–1000 → pixel lệch tối đa **0.32 px** so với box gốc (`python -m lab.data_check`), trừ 63/4.737 box vượt mép bị kẹp về [0, 1000].

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Số đo trên tập train của em (`lab.checks` và tokenizer Qwen): một mẫu gồm **220 token ảnh** + **93 token template/prompt** (cố định, tổng 313 token bị gán -100) + trung bình **70.2 token đáp án** (median 57, max 366; mỗi chữ số là một token nên ~27.7 token/box). Như vậy chỉ **17.4%** số token của chuỗi có nhãn; 82.6% là -100 và riêng token ảnh đã chiếm 58% chuỗi. Ví dụ in ra: chuỗi 562 token, 313 bị mask, 249 tính loss; giải mã phần `labels != -100` ra đúng `[{"label":"crack","bbox_2d":[430,467,583,506]},...]<|im_end|>`.

Vì sao chỉ học phần trả lời:
- Bài toán là p(JSON | ảnh, prompt). Ảnh và prompt là điều kiện; lúc suy luận model không bao giờ phải sinh chúng.
- Vị trí token ảnh chỉ là placeholder `<|image_pad|>` giống hệt nhau, nên "đoán token tiếp theo là `<|image_pad|>`" là việc tầm thường. Prompt thì giống nhau ở mọi mẫu, nên model học thuộc nó trong vài bước.

Nếu tính loss trên cả chuỗi:
- (a) Hơn 80% số hạng của loss trung bình đến từ token dễ, có loss ≈ 0 sau vài chục bước. Gradient của 17% token toạ độ bị pha loãng khoảng 5–6 lần, tương đương giảm learning rate hiệu dụng cho đúng phần cần học. Loss báo cáo trông rất thấp nhưng không phản ánh khả năng định vị (đường loss của em ~0.5–0.6 chỉ đo trên token trả lời).
- (b) Model tốn dung lượng LoRA để tái tạo template.
- (c) Loss giữa các setup không còn so sánh được.

Em vẫn giữ `<|im_end|>` trong nhãn để model học dừng: E0 không được dạy điều này và 50/311 output lặp đến hết 256 token, còn sau SFT thì E3 hỏng 0/311. Em chỉ mask token `\n` sau `<|im_end|>`.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

**Vì sao không thể huấn luyện full khi 4-bit.** bitsandbytes thay `nn.Linear` bằng `Linear4bit` với trọng số `Params4bit`: các byte uint8 đóng gói 2 giá trị NF4 kèm hằng số scale theo khối 64 (double-quant). Đây không phải tham số số thực liên tục nên không có gradient cho trọng số gốc, optimizer không cập nhật được, và `.to(float32)` chỉ ép kiểu các byte đã đóng gói thành rác. Chỉ có thể gắn adapter (LoRA) lên nó. Muốn mở full merger thì merger phải còn là tensor fp16 thật, nghĩa là phải loại ViT (gồm merger và deepstack) khỏi lượng tử hoá bằng `llm_int8_skip_modules`.

**Em đã gặp đúng lỗi này.** Với transformers 5.19, cấu hình `["visual", "lm_head"]` của đề **không** khớp tên `model.visual.*`, vì `should_convert_module` so khớp `re.match` neo ở đầu tên đầy đủ. Kết quả là 104 Linear của ViT, trong đó 8 Linear của projector, bị NF4 âm thầm (`lab.checks` in ra 104 → 0 sau khi sửa). Triệu chứng ở smoke test:
- E2 chỉ báo 54,579,200 tham số projector "huấn luyện được", đúng một nửa 109,105,152 vì `Params4bit` đóng 2 giá trị/byte;
- GradScaler báo `ValueError: Attempting to unscale FP16 gradients.`

Cách xử lý: dùng `llm_int8_skip_modules=["visual", "model.visual", "lm_head"]` và thêm `assert_vit_not_quantized()` ngay sau khi nạp model. Sau khi sửa, E2/E3 huấn luyện đủ 109.1M tham số projector; projector được lưu riêng `projector.pt` và nạp lại bằng `copy_` lúc suy luận. Chi tiết trong `BAO_CAO_LOI_VA_LUU_Y.md`.

**Vì sao float32.**
- (a) GradScaler fp16 không cho unscale gradient fp16, nên master weight của tham số huấn luyện phải ở fp32. Forward vẫn chạy fp16 qua autocast.
- (b) Cập nhật AdamW với lr 2e-4 trên 109M trọng số dày đặc: fp16 chỉ có 10 bit mantissa (ε ≈ 9.8e-4 tương đối), nên nhiều bước cập nhật nhỏ bị làm tròn mất, và moment bậc hai của Adam dễ underflow.
- (c) Gradient đi qua 4 merger (1 chính + 3 deepstack) rồi cộng vào hidden state ở nhiều lớp, nên độ lớn biến thiên rộng và fp16 dễ tràn số dẫn tới loss nan.

Với cấu hình này, loss E2/E3 của em hữu hạn suốt 300 bước. E3 có một đợt loss tăng tạm thời lên ~1.0 quanh bước 210–225 rồi tự hồi phục (`report/figures/loss_sft.png`), nhưng không có nan/inf.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Ảnh là frame cắt từ video dashcam: hai frame cách nhau 0.5–2 s gần như trùng nhau (cùng đoạn đường, cùng ổ gà, cùng ánh sáng, cùng xe phía trước). Chia ngẫu nhiên theo ảnh thì gần như mọi ảnh test đều có frame "anh em" trong train. Model chỉ cần nhớ cảnh, tức nhớ rằng ở đúng con đường đó ổ gà nằm ở toạ độ nào, là đạt IoU cao. F1 khi đó đo khả năng nhớ chứ không đo khả năng tổng quát sang con đường mới, nên cao giả; khi triển khai ở đường mới, hiệu năng sẽ tụt.

`prepare_data.py` gom ảnh theo phiên rồi gán **cả phiên** vào một split bằng thứ tự SHA256 (seed `day10`; salt = 0 là lần đầu thoả ràng buộc ≥ 20 box/lớp ở val và test). Phiên được xác định bằng ngày + giờ trong tên `vlcsnap-…`, ngày với ảnh điện thoại, và khối 50 số với `vlcsnap-N`. Em kiểm tra lại bằng `lab/data_check.py`: không phiên nào nằm ở hai split.

| Split | Số phiên | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 13 | 1.422 | 1.074 | 1.750 | 691 |
| val | 2 | 276 | 66 | 470 | 103 |
| test | 3 | 311 | 121 | 299 | 163 |

Test chỉ gồm 3 phiên (`vlc-2025-02-18-23`: 213 ảnh, `vlc-2025-02-19-17`: 83, `num-4`: 15), nên phân bố lớp lệch: test có 1.87 box/ảnh so với 2.47 của train, và val chỉ có 66 pothole. Đây là lý do em đọc F1 theo từng lớp và luôn kèm CI.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

Có, và mức tăng **vượt nhiễu**:

| | F1 | ΔF1 ghép cặp (CI95) | Tham số | VRAM | Phút |
|---|---|---|---|---|---|
| E1 | 0.282 | | 33.0M | 6.23 GB | 43.6 |
| E2 | 0.369 | +0.087 [+0.048, +0.126] so với E1 | 142.1M | 7.83 GB (+1.6) | 39.9 |
| E3 | 0.420 | +0.051 [+0.008, +0.097] so với E2 | 148.4M | 8.01 GB (+0.2) | 48.1 (+20%) |

- **Mở projector (E2)** là bước đáng tiền nhất: +0.087 F1 với chi phí +1.6 GB và gần như không thêm thời gian. 39.9 phút so với 43.6 của E1 là nhiễu phần cứng: E1r cùng cấu hình E1 chạy ở kernel khác mất 40.4 phút. Với vật thể nhỏ (diện tích trung vị pothole 0.42%, manhole 0.67% khung), cho projector học lại cách nén 2×2 patch thành token và cách bơm đặc trưng DeepStack giúp LLM "thấy" chi tiết tốt hơn. Precision tăng 0.306 → 0.403, FP/ảnh giảm 1.11 → 0.94, manhole F1 0.515 → 0.618.
- **LoRA trên ViT (E3)** thêm +0.051 với chỉ 6.3M tham số và +0.2 GB, nhưng chậm hơn khoảng 20% vì gradient phải chảy ngược qua 24 block ViT (có gradient checkpointing nên VRAM tăng ít). Pothole hưởng lợi nhiều nhất (0.232 → 0.333). CI của hiệu sát 0 (cận dưới +0.008, P(Δ ≤ 0) = 0.011), nên em coi đây là cải thiện có thật nhưng biên độ chưa chắc chắn.
- VRAM: E3 cao nhất vì phải lưu activation và gradient qua ViT, cộng thêm 109M tham số projector ở fp32 kèm 2 moment Adam fp32 (~1.3 GB cho riêng projector + optimizer). Trên T4 16 GB cả ba đều dư sức.

Kết luận: nếu chỉ chọn một thì chọn E2 (tỉ lệ F1/chi phí tốt nhất); nếu có thêm ~5 phút/run thì chọn E3 (tốt nhất tuyệt đối).

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

CI95 bootstrap theo ảnh của `score.py` rộng khoảng **±0.04** quanh F1 (độ rộng 0.078–0.087: E1 [0.244, 0.322], E2 [0.325, 0.412], E3 [0.379, 0.463]). Hệ quả:

- **So CI rời nhau là phép thử quá bảo thủ.** CI của E2 và E3 chồng nhau (0.379–0.412), CI của E1 và E2 gần như chạm nhau (0.322 so với 0.325). Nếu chỉ nhìn CI riêng lẻ, em không kết luận được E3 > E2. Vì hai setup được chấm trên **cùng** 311 ảnh, phép đúng là bootstrap **ghép cặp** trên hiệu F1. Phép này loại phần nhiễu chung do độ khó từng ảnh, cho E3 − E2 = +0.051 [+0.008, +0.097] và E2 − E1 = +0.087 [+0.048, +0.126]: cả hai đều có ý nghĩa, nhưng E3 − E2 sát ngưỡng.
- **Nhiễu giữa các lần chạy là một nguồn khác mà CI test không đo.** Em vô tình có phép đo này: E1r (cùng cấu hình E1, seed 42, chạy ở kernel khác) đạt 0.293 so với 0.282 của E1, chênh +0.012 [−0.014, +0.036]. Vì vậy mọi chênh lệch ≲ 0.02, như E5 − E1r = +0.010, không thể gán cho phương pháp.
- **CI bootstrap theo ảnh vẫn quá lạc quan**, vì 311 ảnh chỉ đến từ 3 phiên quay (213 ảnh của cùng một phiên). Các frame liền kề tương quan mạnh nên số mẫu "độc lập" thực tế nhỏ hơn 311 nhiều. Bootstrap theo phiên sẽ cho CI rộng hơn, nhưng chỉ có 3 phiên nên không ước lượng được.
- **F1 theo lớp còn rộng hơn**, nhất là pothole với 121 box, phần lớn nằm trong một phiên.

Kết luận đứng vững: E3 > E1 (+0.138 [+0.094, +0.181]), E2 > E1, 100% dữ liệu > 10% dữ liệu. Kết luận còn yếu: E3 > E2. Kết luận không đứng vững: E5 > SFT. Mọi kết luận chỉ đúng cho các con đường giống 3 phiên test này.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Ở model tốt nhất E3: **crack kém nhất (F1 0.325)**, pothole 0.333, manhole tốt nhất 0.683. Recall theo kích thước GT (`report/analysis/e3.json`):

| Lớp (diện tích trung vị GT) | nhỏ < 0.5% | vừa 0.5–2% | lớn ≥ 2% |
|---|---|---|---|
| crack (1.98%) | 1/15 = 0.07 | 38/136 = 0.28 | 61/148 = 0.41 |
| pothole (0.42%) | 20/68 = 0.29 | 8/46 = 0.17 | 5/7 = 0.71 |
| manhole (0.67%) | 29/64 = 0.45 | 67/96 = 0.70 | 2/3 = 0.67 |

- **Crack kém vì ngữ nghĩa và cách gán nhãn, không phải vì nhỏ.** Crack có box lớn nhất (trung vị ~2% khung) nhưng là vật thể dạng đường mảnh, phân nhánh, không có biên rõ. Cùng một mạng nứt, người gán nhãn có lúc khoanh một box lớn, có lúc chia nhiều box nhỏ (xem `gt_samples_train.jpg`: một box crack lớn ôm cả vùng ổ gà). Vì vậy box dự đoán dù "đúng vùng" vẫn khó đạt IoU ≥ 0.5: 122/286 FP của E3 là box lệch vị trí cùng lớp (0.1 ≤ IoU < 0.5). Crack còn là lớp model dự đoán nhiều nhất (316/517 box) nên gánh phần lớn FP.
- **Pothole kém vì nhỏ và lẫn với crack.** Pothole là lớp nhỏ nhất (trung vị 0.42% khung, tức chỉ khoảng 1–2 token ảnh sau khi gộp 32×32 px). Pothole cũng thường nằm trong vùng nứt, nên 37 FP của E3 là sai lớp (IoU ≥ 0.5 với GT lớp khác, chủ yếu pothole ↔ crack); xem ảnh phải của `e3_du_thua.jpg`, nơi các ổ gà bị gán là crack. Số mẫu train không thiếu (1.074 box), nhưng test chỉ có 121 box nên F1 pothole dao động mạnh: E2 0.232 → E3 0.333.
- **Manhole tốt nhất vì có ngữ nghĩa rõ ràng**: hình tròn/chữ nhật kim loại, biên sắc, ít mơ hồ khi gán nhãn. Precision đạt 0.790 dù box nhỏ.
- Mở ViT/projector giúp nhiều nhất cho vật thể nhỏ: pothole F1 E1 0.207 → E3 0.333, manhole 0.515 → 0.683.

Ảnh: `report/figures/e3_dung.jpg` (đúng hoàn toàn), `e3_bo_sot.jpg` (bỏ sót: ảnh trái ngược sáng, mặt đường hư hại dày đặc, model gộp cả vùng thành **một** box crack khổng lồ thay vì 7 box pothole/crack riêng; ảnh phải có box đúng vùng nhưng lệch hoặc gán pothole thành crack, nên TP = 0 dù "nhìn đúng chỗ"), `e3_du_thua.jpg` (box thừa trên mặt đường bong tróc và nhầm pothole ↔ crack).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

Em đổi **đúng một biến `data_size`**: `train_fraction` 0.1 (142 ảnh lấy ngẫu nhiên với seed 42) so với 1.0 (1.422 ảnh). Mọi thứ khác giữ nguyên: setup E1, r=16, alpha=32, lr 2e-4, batch 2×4, **cùng 300 bước**, seed 42, cùng prompt và thang 0–1000. Giá trị 1.0 chính là run E1.

| | F1 | CI95 | P | R | FP/ảnh | loss cuối |
|---|---|---|---|---|---|---|
| data 0.1 | 0.183 | [0.152, 0.216] | 0.164 | 0.206 | 1.96 | 0.007 |
| data 1.0 | 0.282 | [0.244, 0.322] | 0.306 | 0.261 | 1.11 | ~0.57 |

- 10% dữ liệu giảm F1 −0.099 (ghép cặp [−0.136, −0.064]). Hai CI95 riêng lẻ cũng **không chồng nhau** (0.216 < 0.244), nên kết luận "nhiều dữ liệu hơn tốt hơn rõ rệt" là đáng tin trên test này.
- Cơ chế thấy rõ trong log: với 142 ảnh, 300 bước tương đương ~17 epoch nên loss train tụt về **0.007** (học thuộc, `loss_sft.png`). Model sinh nhiều box hơn (2.35 box/ảnh so với GT 1.87), FP/ảnh tăng gấp đôi (1.96), 396 box bịa. Recall chỉ giảm nhẹ (0.206 so với 0.261) nhưng precision giảm một nửa.
- Mặt khác, 142 ảnh vẫn đưa F1 từ 0.088 (E0) lên 0.183, tức khoảng 1/2 mức tăng của bộ đầy đủ, đủ để dạy định dạng (parse_valid 1.000) và thang toạ độ.

Giới hạn:
1. Em chạy E4 trên setup **E1** chứ không phải setup tốt nhất E3. Lý do là để kịp thời gian, E4 được xếp chạy song song với E2/E3 trước khi biết kết quả. Ablation vẫn sạch (chỉ một biến thay đổi so với E1), nhưng con số tuyệt đối có thể khác nếu chạy trên E3.
2. Chỉ có một seed và một mẫu 10%; nhiễu giữa các lần chạy (~0.012, xem E1r) nhỏ hơn nhiều so với hiệu 0.099 nên không đổi kết luận.
3. Như Câu 8, test chỉ có 3 phiên nên kết luận áp dụng cho các con đường giống vậy.

### Bonus E5 — GRPO với reward IoU

- **Reward** (`lab/common.py: compute_reward`): 0.0 nếu JSON hỏng, 1.0 nếu GT và dự đoán cùng rỗng; còn lại 0.2 + 0.8 · ΣIoU(cặp khớp cùng lớp, IoU ≥ 0.5, ghép tham lam) / max(n_gt, n_pred, 1). Có **11 unit test** (`python -m pytest lab/tests -q`): hoàn hảo 1.0, JSON hỏng 0.0, sai nhãn 0.2, thừa box 0.6, cùng rỗng 1.0, trả `[]` khi có GT 0.2, box khổng lồ 0.2, ngưỡng IoU 0.45/0.67, một dự đoán không khớp hai GT, chấp nhận ```json.
- **Huấn luyện**: `trl.GRPOTrainer` 1.9.2, khởi tạo từ E1r (bản chạy lại E1 trong kernel song song để không phải chờ kernel chính), 60 bước, 4 completion/ảnh, 2 ảnh/bước, lr 5e-6, beta 0, `max_completion_length=128`; chỉ dùng 1.280 ảnh train có ≤ 4 box để đáp án đúng không bị cắt. 24.0 phút, VRAM đỉnh 5.31 GB, ~23.7 s/bước. Lưu ý: TRL ép adapter QLoRA sang bf16; em đưa lại về fp32 vì T4 không có bf16 và cập nhật cỡ 5e-6 nhỏ hơn độ phân giải bf16. Log: `report/logs/e5_*.json*`, `report/figures/grpo_reward.png`.
- **Reward hacking: không thấy.** Tỉ lệ trả `[]` = 0 suốt 60 lượt sinh; box khổng lồ (> 25% khung) chỉ 0.017/lượt; số box dự đoán trung bình 1.77 so với GT 2.02 (không phình ra); JSON hỏng giảm 3.7% → 1.3%; độ dài completion ổn định (54.9 → 54.2 token, chỉ 1.5% bị cắt). Thiết kế reward chặn hai kiểu hack: mọi ảnh đều có GT nên `[]` chỉ được 0.2, và box khổng lồ có IoU < 0.5 nên cũng chỉ được 0.2. Reward trung bình dao động 0.27–0.37 và **không tăng rõ** (20 bước đầu 0.312, 20 bước cuối 0.285). 9/60 bước có gradient bằng 0, và trung bình 39% nhóm có cả 4 completion cùng reward (advantage = 0), vì reward thưa: phần lớn completion chỉ được 0.2 khi không box nào đạt IoU 0.5.
- **So với SFT**: E5 F1 0.303 [0.262, 0.346] so với E1r 0.293 [0.254, 0.335]; ghép cặp +0.010 [−0.005, +0.025], P(Δ ≤ 0) = 0.106, nên **không vượt CI95** và nhỏ hơn cả nhiễu giữa hai lần chạy E1 (0.012). Xu hướng thì nhất quán với mục tiêu reward: precision tăng 0.315 → 0.355, FP/ảnh giảm 1.12 → 0.90 (mẫu số max(n_gt, n_pred) phạt box thừa), recall giảm nhẹ 0.274 → 0.264. Với 60 bước × 8 completion và reward thưa thì chưa đủ tín hiệu; cần nhiều bước hơn hoặc reward dày hơn (ví dụ thêm IoU từng phần dưới 0.5).

## Nhìn lại

- **Lỗi công cụ chấm không bắt được, và cũng không bắt được ở phía học viên nếu không đếm tham số:** `llm_int8_skip_modules=["visual","lm_head"]` của đề không còn tác dụng với transformers 5.19, khiến cả ViT và projector bị NF4 âm thầm. Pipeline vẫn chạy và E1 vẫn ra số; chỉ E2/E3 lộ ra nhờ số tham số projector bằng đúng một nửa. Nếu bỏ qua, E0/E1 sẽ được đo với một ViT 4-bit khác hẳn mô tả của đề. Em đã sửa trước khi chạy thật và ghi lại trong `BAO_CAO_LOI_VA_LUU_Y.md`.
- **Nhãn:** 63 box YOLO vượt mép ảnh (tới 1015.6/1000), và `score.py` không kẹp GT nên box dự đoán đã kẹp không thể đạt IoU 1. Crack được khoanh không nhất quán (một box lớn hoặc nhiều box nhỏ), đây có lẽ là trần F1 lớn nhất của lớp crack. Một số ổ gà nằm trong box crack lớn khiến hai lớp chồng nhau.
- **Điều em chưa chắc:**
  - (1) Đợt loss tăng của E3 quanh bước 210–225 nhiều khả năng do một batch khó gặp lr còn cao khi ViT đang mở, vì không có nan và tự hồi phục; em chưa kiểm chứng bằng seed khác.
  - (2) CI của `score.py` bootstrap theo ảnh trong khi test chỉ có 3 phiên, nên mọi CI ở trên đều lạc quan.
  - (3) Chạy lại cùng cấu hình (E1 so với E1r) chênh 0.012 F1 dù cùng seed (bnb/cuBLAS không tất định), nên mọi hiệu dưới khoảng 0.02 là nhiễu.
- **Khác với đề:** em dùng Kaggle T4 thay Colab T4 (cùng loại GPU, mỗi thí nghiệm 1 GPU). E4 chạy trên setup E1 thay vì setup tốt nhất E3 để kịp thời gian. E5 khởi tạo từ E1r chứ không phải adapter tốt nhất E3, cũng để chạy song song.
