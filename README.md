# cntovn

Công cụ dịch phụ đề `.srt` tiếng Trung sang tiếng Việt với định dạng giữ nguyên, phù hợp để đưa vào các dịch vụ chuyển văn bản thành giọng nói như Vbee.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cách sử dụng

```bash
python translate_srt.py input_zh.srt output_vi.srt
```

Các tùy chọn hữu ích:

- `--zh-en-model`: đổi sang model Hugging Face khác để dịch từ tiếng Trung sang tiếng Anh.
- `--en-vi-model`: đổi sang model dịch tiếng Anh sang tiếng Việt.
- `--batch-size`: chỉnh kích thước lô dịch nếu cần tối ưu tốc độ/bộ nhớ.

Kết quả đầu ra là file `.srt` tiếng Việt đã được dịch theo từng phân cảnh tương ứng với file gốc.
