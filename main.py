"""
========================================================================================
PORTFOLIO MANAGEMENT & QUANTITATIVE ANALYSIS: FPT & VNM
Author: Quantitative Portfolio Analyst
Mô hình triển khai:
  1. Efficient Frontier & Capital Allocation Line (CAL) - Markowitz Portfolio Theory
  2. Asset Weights Allocation & Portfolio Transition Dynamics (VNM & FPT)
  3. Return Distribution Analysis & Risk Metrics (Histogram, Normal Fit, VaR, CVaR)
  4. CAPM Model: Security Characteristic Line (SCL) & Portfolio Beta Sensitivity (w: 0 -> 1)
  5. Security Market Line (SML) & Portfolio Valuation / Jensen's Alpha (w: 0 -> 1)
  6. Covariance & Correlation Heatmap Analysis
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
    create_fig_asset_weights,
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

    print(f"\n4. CAPM & STATSMODELS OLS ECONOMETRIC ANALYSIS:")
    for ticker in data['stock_tickers']:
        st = data['capm_stats'][ticker]
        print(f"   - {ticker}:")
        print(f"       + Hệ số Beta (β):     {st['beta']:.4f} (t-stat: {st['beta_tstat']:.2f}, p-value: {st['beta_pvalue']:.2e})")
        print(f"       + 95% CI Beta:        [{st['beta_ci_95'][0]:.3f}, {st['beta_ci_95'][1]:.3f}]")
        print(f"       + Jensen's Alpha (α):  {st['alpha_annual']*100:+.2f}%/năm (p-value: {st['alpha_pvalue']:.4f})")
        print(f"       + Hệ số R² / Adj R²:  {st['r_squared']:.4f} / {st['adj_r_squared']:.4f}")
        print(f"       + F-statistic:        {st['f_stat']:.2f} (p-value: {st['f_pvalue']:.2e})")
    print(f"   - Danh mục Tangency [Scipy SLSQP] (FPT = {data['w_fpt_tan']*100:.1f}%, VNM = {data['w_vnm_tan']*100:.1f}%):")
    print(f"       + Hệ số Beta (β_p):    {data['beta_tan']:.2f}")
    print(f"       + Jensen's Alpha (α_p): {data['alpha_tan']*100:+.2f}%/năm")
    print(f"   - Danh mục MVP [Scipy SLSQP] (FPT = {data['w_fpt_mvp']*100:.1f}%, VNM = {data['w_vnm_mvp']*100:.1f}%):")
    print(f"       + Hệ số Beta (β_p):    {data['beta_mvp']:.2f}")
    print(f"       + Jensen's Alpha (α_p): {data['alpha_mvp']*100:+.2f}%/năm")

    print(f"\n5. SCIPY STATS & RISK MODELING (Jarque-Bera & Modified VaR):")
    for ticker in data['stock_tickers']:
        m = data['dist_metrics'][ticker]
        jb_txt = "Non-Normal (p<0.01)" if m['jb_pvalue'] < 0.01 else "Normal"
        print(f"   - {ticker}:")
        print(f"       + Skewness:           {m['skew']:.4f} | Excess Kurtosis: {m['kurt']:.4f}")
        print(f"       + Jarque-Bera Test:   JB = {m['jb_stat']:.2f}, p-value = {m['jb_pvalue']:.2e} -> {jb_txt}")
        print(f"       + Historical VaR 95%: {m['var_95']:.2f}% | Cornish-Fisher VaR: {m['var_cf_95']:.2f}%")
        print(f"       + CVaR 95% (1-Day):   {m['cvar_95']:.2f}%")

    print(f"\n6. ASSET ALLOCATION STRATEGIES & WEIGHTS BREAKDOWN (VNM vs FPT):")
    print(f"   {'Chiến lược':<28} | {'FPT':>7} | {'VNM':>7} | {'E(R)':>8} | {'Vol':>8} | {'Sharpe':>7} | {'Beta':>6}")
    print("   " + "-"*80)
    for s in data.get('portfolio_strategies', []):
        print(f"   {s['name']:<28} | {s['w_fpt']*100:>6.1f}% | {s['w_vnm']*100:>6.1f}% | {s['ret']*100:>7.2f}% | {s['vol']*100:>7.2f}% | {s['sharpe']:>7.2f} | {s['beta']:>6.2f}")
    print("="*70)

    # 3. Khởi động Web Dashboard (Nhúng trực tiếp vector SVG, không tách thành từng ảnh tĩnh)
    print("\n[+] Đang mở Web Dashboard tích hợp toàn bộ đồ thị trên trình duyệt...")
    start_web_server(port=8501)

if __name__ == '__main__':
    main()
