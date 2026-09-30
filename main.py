"""
========================================================================================
PORTFOLIO MANAGEMENT & QUANTITATIVE ANALYSIS: FPT & VNM
Author: Quantitative Portfolio Analyst
Mô hình triển khai:
  1. Efficient Frontier & Capital Allocation Line (CAL) - Markowitz Portfolio Theory
  2. Return Distribution Analysis & Risk Metrics (Histogram, Normal Fit, VaR, CVaR)
  3. CAPM Model: Security Characteristic Line (SCL) & Portfolio Beta Sensitivity (w: 0 -> 1)
  4. Security Market Line (SML) & Portfolio Valuation / Jensen's Alpha (w: 0 -> 1)
  5. Covariance & Correlation Heatmap Analysis
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

from portfolio_engine import load_and_calculate_portfolio
from charts import (
    create_fig_efficient_frontier,
    create_fig_return_distribution,
    create_fig_capm_regression,
    create_fig_sml,
    create_fig_covariance_correlation
)
from web_dashboard import start_web_server

def main():
    print("="*75)
    print("      KHỞI CHẠY HỆ THỐNG ĐỊNH LƯỢNG DANH MỤC ĐẦU TƯ FPT & VNM")
    print("="*75)
    
    # 1. Thu thập & Tiền xử lý dữ liệu
    data = load_and_calculate_portfolio()
    
    # 2. In báo cáo tóm tắt cho Analyst
    print("\n" + "="*70)
    print("             EXECUTIVE PORTFOLIO ANALYST REPORT")
    print("="*70)
    print(f"1. OPTIMAL ALLOCATION (Tangency Portfolio):")
    print(f"   - Tỷ trọng khuyến nghị: FPT = {data['w_fpt_tan']*100:.1f}% | VNM = {data['w_vnm_tan']*100:.1f}%")
    print(f"   - Lợi nhuận kỳ vọng:    {data['ret_tan']*100:.2f}%/năm")
    print(f"   - Mức độ rủi ro (Vol):  {data['vol_tan']*100:.2f}%/năm")
    print(f"   - Hệ số Sharpe:         {data['sharpe_tan']:.2f}")

    print(f"\n2. MINIMUM RISK ALLOCATION (Minimum Variance Portfolio - MVP):")
    print(f"   - Tỷ trọng an toàn:     FPT = {data['w_fpt_mvp']*100:.1f}% | VNM = {data['w_vnm_mvp']*100:.1f}%")
    print(f"   - Mức độ rủi ro tối thiểu: {data['vol_mvp']*100:.2f}%/năm (Lợi nhuận: {data['ret_mvp']*100:.2f}%)")

    print(f"\n3. COVARIANCE & CORRELATION ANALYSIS (3-Year):")
    print(f"   - Annualized Covariance (FPT, VNM): {data['cov_fpt_vnm_annual']:.6f}")
    print(f"   - Correlation Coefficient (r):     {data['corr_fpt_vnm']:.4f}")
    print(f"   - Diversification Benefit:         Đạt hiệu ứng giảm rủi ro vượt trội (MVP Vol = {data['vol_mvp']*100:.2f}%)")

    print(f"\n4. CAPM & SYSTEMATIC RISK (BETA) ANALYSIS:")
    for ticker in data['stock_tickers']:
        st = data['capm_stats'][ticker]
        print(f"   - {ticker}:")
        print(f"       + Hệ số Beta (β):     {st['beta']:.2f} (Đo lường rủi ro hệ thống so với VN30)")
        print(f"       + Jensen's Alpha (α):  {st['alpha_annual']*100:+.2f}%/năm")
        print(f"       + Hệ số R²:           {st['r_squared']:.2f}")
    print(f"   - Danh mục Tangency (FPT = {data['w_fpt_tan']*100:.0f}%, VNM = {data['w_vnm_tan']*100:.0f}%):")
    print(f"       + Hệ số Beta (β_p):    {data['beta_tan']:.2f}")
    print(f"       + Jensen's Alpha (α_p): {data['alpha_tan']*100:+.2f}%/năm")
    print(f"   - Danh mục MVP (FPT = {data['w_fpt_mvp']*100:.0f}%, VNM = {data['w_vnm_mvp']*100:.0f}%):")
    print(f"       + Hệ số Beta (β_p):    {data['beta_mvp']:.2f}")
    print(f"       + Jensen's Alpha (α_p): {data['alpha_mvp']*100:+.2f}%/năm")
    print("="*70)

    # 3. Khởi động Web Dashboard (Nhúng trực tiếp vector SVG, không tách thành từng ảnh tĩnh)
    print("\n[+] Đang mở Web Dashboard tích hợp toàn bộ đồ thị trên trình duyệt...")
    start_web_server(port=8501)

if __name__ == '__main__':
    main()
