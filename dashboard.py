import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yfinance as yf
import pandas as pd

# 1. Danh sách mã cổ phiếu và thời gian cần lấy dữ liệu
tickers = ['FPT.VN', 'VNM.VN']
start_date = '2023-01-01'
end_date = '2026-09-29'

print(f"[-] Đang tải dữ liệu Adjusted Closing Price của {tickers}...")

# 2. Tải dữ liệu: Trong yfinance phiên bản mới, auto_adjust=True thì cột 'Close' chính là giá đóng cửa điều chỉnh
try:
    # Cách 1 (Khuyên dùng và ổn định nhất): Dùng auto_adjust=True và lấy ['Close']
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close'][tickers].dropna()
except Exception:
    # Cách 2 (Dự phòng): Dùng auto_adjust=False và lấy ['Adj Close']
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    data = raw['Adj Close'][tickers].dropna()

# 3. Hiển thị thông tin bảng giá
print("\n" + "="*60)
print("     BẢNG GIÁ ĐÓNG CỬA ĐIỀU CHỈNH (ADJUSTED CLOSING PRICE)")
print("="*60)
print(f"• Số phiên giao dịch: {len(data)}")
print(f"• Thời gian: Từ {data.index[0].strftime('%d/%m/%Y')} đến {data.index[-1].strftime('%d/%m/%Y')}")

print("\n--- 5 Phiên đầu tiên (Head): ---")
print(data.head().apply(lambda col: col.map(lambda x: f"{x:,.2f} VND")))

print("\n--- 5 Phiên gần nhất (Tail): ---")
print(data.tail().apply(lambda col: col.map(lambda x: f"{x:,.2f} VND")))

# 4. Xuất ra file Excel (.xlsx) và CSV
excel_name = 'Adjusted_Close_FPT_VNM.xlsx'
daily_rets = data.pct_change().dropna()
stats_df = data.describe()

try:
    with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
        # Sheet 1: Giá đóng cửa điều chỉnh
        data.to_excel(writer, sheet_name='Gia_Dieu_Chinh')
        # Sheet 2: Tỷ suất sinh lời hàng ngày
        daily_rets.to_excel(writer, sheet_name='Loi_Nhuan_Ngay')
        # Sheet 3: Thống kê mô tả
        stats_df.to_excel(writer, sheet_name='Thong_Ke')
        
    print(f"\n[+] ĐÃ XUẤT THÀNH CÔNG FILE EXCEL: {excel_name}")
    print("    • Sheet 1: 'Gia_Dieu_Chinh' (Adjusted Close Price)")
    print("    • Sheet 2: 'Loi_Nhuan_Ngay' (Daily Returns)")
    print("    • Sheet 3: 'Thong_Ke' (Descriptive Statistics)")
except PermissionError:
    alt_excel = 'Adjusted_Close_FPT_VNM_latest.xlsx'
    with pd.ExcelWriter(alt_excel, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Gia_Dieu_Chinh')
        daily_rets.to_excel(writer, sheet_name='Loi_Nhuan_Ngay')
        stats_df.to_excel(writer, sheet_name='Thong_Ke')
    print(f"\n[!] File '{excel_name}' đang mở trong Excel, đã xuất dự phòng sang: {alt_excel}")
except Exception as e:
    print(f"\n[!] Lỗi khi xuất Excel: {e}")

# Xuất thêm file CSV dự phòng
try:
    data.to_csv('Adj_Close_FPT_VNM.csv')
    print(f"[+] Đã lưu bản sao CSV: Adj_Close_FPT_VNM.csv")
except Exception:
    pass

