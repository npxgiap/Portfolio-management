"""
========================================================================================
PORTFOLIO MANAGEMENT & QUANTITATIVE ANALYTICS: MASTER WEB DASHBOARD
========================================================================================
File tổng hợp thiết kế dưới dạng Web Dashboard hoàn chỉnh.
- Tự động nạp dữ liệu định lượng từ portfolio_engine.py
- Nhúng toàn bộ 5 mô hình đồ thị từ charts.py dưới dạng vector SVG sắc nét
- Không tách thành từng file ảnh tĩnh (.png)
- Tích hợp bảng giá Adjusted Close, bảng Variance/Expected Return, và nút tải Excel/CSV
- Tự động mở trình duyệt web khi chạy: python dashboard.py
========================================================================================
"""

import sys
import os

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm đường dẫn thư mục hiện tại vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from web_dashboard import start_web_server

if __name__ == '__main__':
    # Khởi động Web Server cục bộ tại cổng 8501 và tự động mở trình duyệt web
    start_web_server(port=8501)
