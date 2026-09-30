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

# Đảm bảo in tiếng Việt có dấu không bị lỗi trên Windows console (cp1252)
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thiết lập phong cách hiển thị đồ họa chuyên nghiệp
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# ==============================================================================
# 1. THU THẬP & TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
stock_tickers = ['FPT.VN', 'VNM.VN']
# FUEKIV30.VN: KIM Growth VN30 ETF - Quỹ ETF mô phỏng chỉ số VN30 có đầy đủ dữ liệu 3 năm (2023-2026) trên yfinance
market_ticker = 'FUEKIV30.VN'
all_tickers = stock_tickers + [market_ticker]

# Khoảng thời gian phân tích: ĐỦ 3 NĂM từ 27-09-2023 đến 27-09-2026
start_date = '2023-09-27'
end_date = '2026-09-28'   # yfinance lấy cận end exclusive nên đặt 2026-09-28 để lấy trọn ngày 27-09-2026
rf_annual = 0.045         # Lãi suất phi rủi ro: 4.5%/năm (TPCP Việt Nam kỳ hạn 10 năm)
trading_days = 252        # Số ngày giao dịch chuẩn trong năm

print(f"[-] Đang tải dữ liệu trọn vẹn 3 năm ({start_date} -> 27-09-2026) từ Yahoo Finance...")
raw_data = yf.download(all_tickers, start=start_date, end=end_date)['Close'][all_tickers].dropna()
print(f"[+] Đã tải thành công: {len(raw_data)} phiên giao dịch (Từ {raw_data.index[0].strftime('%d/%m/%Y')} đến {raw_data.index[-1].strftime('%d/%m/%Y')})")

# Lợi nhuận hàng ngày (Daily Returns)
daily_returns = raw_data.pct_change().dropna()
stock_returns = daily_returns[stock_tickers]
market_returns = daily_returns[market_ticker]

# Lợi nhuận & Ma trận hiệp phương sai (Covariance Matrix)
daily_cov_matrix = stock_returns.cov()
cov_matrix = daily_cov_matrix * trading_days  # Ma trận hiệp phương sai năm hóa
corr_matrix = stock_returns.corr()            # Ma trận hệ số tương quan

annual_returns = stock_returns.mean() * trading_days
stock_vols = stock_returns.std() * np.sqrt(trading_days)
market_annual_return = market_returns.mean() * trading_days
market_vol = market_returns.std() * np.sqrt(trading_days)

corr_fpt_vnm = corr_matrix.loc['FPT.VN', 'VNM.VN']
cov_fpt_vnm_annual = cov_matrix.loc['FPT.VN', 'VNM.VN']
cov_fpt_vnm_daily = daily_cov_matrix.loc['FPT.VN', 'VNM.VN']

print("\n" + "="*65)
print(f"BẢNG MA TRẬN HIỆP PHƯƠNG SAI & HỆ SỐ TƯƠNG QUAN (FPT & VNM):")
print("="*65)
print(f"1. Ma trận hiệp phương sai ngày (Daily Covariance Matrix):")
print(daily_cov_matrix.apply(lambda col: col.map(lambda x: f"{x:.6f}")))
print(f"\n2. Ma trận hiệp phương sai năm hóa (Annualized Covariance Matrix - 252 days):")
print(cov_matrix.apply(lambda col: col.map(lambda x: f"{x:.6f}")))
print(f"\n3. Ma trận tương quan (Correlation Matrix):")
print(corr_matrix.apply(lambda col: col.map(lambda x: f"{x:.4f}")))
print(f"\n-> Chi tiết cặp FPT - VNM:")
print(f"   • Covariance (Daily):  {cov_fpt_vnm_daily:.6f}")
print(f"   • Covariance (Annual): {cov_fpt_vnm_annual:.6f}")
print(f"   • Correlation:         {corr_fpt_vnm:.4f}")
print("="*65)

# ==============================================================================
# TÍNH TOÁN CHI TIẾT EXPECTED RETURN & CÁC CHỈ SỐ VARIANCE CỦA CÁC CỔ PHIẾU
# ==============================================================================
# 1. Expected Return (Tỷ suất sinh lời kỳ vọng)
daily_mean_returns = daily_returns.mean()                      # Lợi nhuận trung bình ngày
annual_arithmetic_returns = daily_mean_returns * trading_days  # Lợi nhuận trung bình năm hóa (Số học)
total_days = len(raw_data)
# CAGR: Tỷ suất sinh lời kép / hình học năm hóa (Geometric Return)
cagr_returns = (raw_data.iloc[-1] / raw_data.iloc[0]) ** (trading_days / total_days) - 1

# 2. Variance & Volatility (Phương sai & Rủi ro biến động)
daily_variances = daily_returns.var()                          # Phương sai mẫu theo ngày (Daily Variance: σ²)
annual_variances = daily_variances * trading_days              # Phương sai năm hóa (Annualized Variance)
daily_stds = daily_returns.std()                               # Độ lệch chuẩn ngày (Daily Volatility: σ)
annual_stds = daily_stds * np.sqrt(trading_days)               # Độ lệch chuẩn năm hóa (Annualized Volatility)

# 3. Downside Variance & Semi-deviation (Phương sai giảm giá rủi ro)
downside_var_daily = daily_returns.apply(lambda col: np.mean(np.minimum(col, 0)**2))
downside_var_annual = downside_var_daily * trading_days
downside_std_annual = np.sqrt(downside_var_annual)

# Bảng tổng hợp các chỉ số Expected Return & Variance
metrics_df = pd.DataFrame({
    'Daily Mean Return (%)': daily_mean_returns * 100,
    'Annualized Return (%)': annual_arithmetic_returns * 100,
    'CAGR Geometric (%)': cagr_returns * 100,
    'Daily Variance (σ²)': daily_variances,
    'Annualized Var (σ²)': annual_variances,
    'Daily Std Dev (%)': daily_stds * 100,
    'Annualized Vol (%)': annual_stds * 100,
    'Downside Var (Annual)': downside_var_annual,
    'Downside Vol (%)': downside_std_annual * 100
})

print("\n" + "="*85)
print(f"BẢNG THỐNG KÊ CHI TIẾT EXPECTED RETURN & CÁC CHỈ SỐ VARIANCE (3 NĂM):")
print("="*85)
print(metrics_df.apply(lambda col: col.map(lambda x: f"{x:.4f}" if 'Var' in col.name or 'σ²' in col.name else f"{x:.2f}%")).to_string())
print("="*85)

# ==============================================================================
# 2. MÔ PHỎNG EFFICIENT FRONTIER & TỐI ƯU HÓA DANH MỤC
# ==============================================================================
weights_fpt = np.linspace(0.0, 1.0, 201)
port_returns = []
port_volatilities = []
port_sharpes = []

for w_fpt in weights_fpt:
    w = np.array([w_fpt, 1.0 - w_fpt])
    ret = np.dot(w, annual_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (ret - rf_annual) / vol
    
    port_returns.append(ret)
    port_volatilities.append(vol)
    port_sharpes.append(sharpe)

port_returns = np.array(port_returns)
port_volatilities = np.array(port_volatilities)
port_sharpes = np.array(port_sharpes)

# 1. Minimum Variance Portfolio (MVP)
min_vol_idx = np.argmin(port_volatilities)
w_fpt_mvp = weights_fpt[min_vol_idx]
w_vnm_mvp = 1.0 - w_fpt_mvp
ret_mvp = port_returns[min_vol_idx]
vol_mvp = port_volatilities[min_vol_idx]
sharpe_mvp = port_sharpes[min_vol_idx]

# 2. Maximum Sharpe Ratio Portfolio (Tangency Portfolio)
max_sharpe_idx = np.argmax(port_sharpes)
w_fpt_tan = weights_fpt[max_sharpe_idx]
w_vnm_tan = 1.0 - w_fpt_tan
ret_tan = port_returns[max_sharpe_idx]
vol_tan = port_volatilities[max_sharpe_idx]
sharpe_tan = port_sharpes[max_sharpe_idx]

# ==============================================================================
# 3. TÍNH TOÁN CAPM & HỆ SỐ BETA CHO FPT, VNM VÀ DANH MỤC (WEIGHT 0 -> 1)
# ==============================================================================
rf_daily = rf_annual / trading_days
excess_market = market_returns - rf_daily

capm_stats = {}
for ticker in stock_tickers:
    excess_stock = stock_returns[ticker] - rf_daily
    # Hồi quy tuyến tính: Excess Stock = Alpha + Beta * Excess Market
    beta, alpha_daily = np.polyfit(excess_market, excess_stock, 1)
    alpha_annual = alpha_daily * trading_days
    
    # R-squared
    y_pred = alpha_daily + beta * excess_market
    ss_res = np.sum((excess_stock - y_pred) ** 2)
    ss_tot = np.sum((excess_stock - np.mean(excess_stock)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Kỳ vọng lợi nhuận theo CAPM: E(R) = Rf + Beta * (E(Rm) - Rf)
    expected_capm = rf_annual + beta * (market_annual_return - rf_annual)
    
    capm_stats[ticker] = {
        'beta': beta,
        'alpha_annual': alpha_annual,
        'r_squared': r_squared,
        'expected_capm': expected_capm
    }

# Tính toán các chỉ số CAPM cho danh mục khi weight chạy từ 0 đến 1 (101 kịch bản)
weights_capm = np.linspace(0.0, 1.0, 101)  # w_fpt từ 0.0 (100% VNM) đến 1.0 (100% FPT)
beta_fpt = capm_stats['FPT.VN']['beta']
beta_vnm = capm_stats['VNM.VN']['beta']
alpha_fpt = capm_stats['FPT.VN']['alpha_annual']
alpha_vnm = capm_stats['VNM.VN']['alpha_annual']

# Beta danh mục: Beta_p(w) = w * Beta_fpt + (1 - w) * Beta_vnm
port_betas = weights_capm * beta_fpt + (1.0 - weights_capm) * beta_vnm

# Alpha danh mục: Alpha_p(w) = w * Alpha_fpt + (1 - w) * Alpha_vnm
port_alphas = weights_capm * alpha_fpt + (1.0 - weights_capm) * alpha_vnm

# Tỷ suất sinh lời kỳ vọng thực tế (Historical/Expected) của danh mục
port_actual_capm_rets = weights_capm * annual_returns['FPT.VN'] + (1.0 - weights_capm) * annual_returns['VNM.VN']

# Tỷ suất sinh lời kỳ vọng theo mô hình CAPM: E(Rp) = Rf + Beta_p * (E(Rm) - Rf)
port_capm_rets = rf_annual + port_betas * (market_annual_return - rf_annual)

# Các điểm danh mục đặc biệt (MVP & Tangency)
beta_mvp = w_fpt_mvp * beta_fpt + w_vnm_mvp * beta_vnm
alpha_mvp = w_fpt_mvp * alpha_fpt + w_vnm_mvp * alpha_vnm
capm_ret_mvp = rf_annual + beta_mvp * (market_annual_return - rf_annual)

beta_tan = w_fpt_tan * beta_fpt + w_vnm_tan * beta_vnm
alpha_tan = w_fpt_tan * alpha_fpt + w_vnm_tan * alpha_vnm
capm_ret_tan = rf_annual + beta_tan * (market_annual_return - rf_annual)

# ==============================================================================
# HÀM BỔ TRỢ: TÍNH THỐNG KÊ PHÂN PHỐI & RỦI RO
# ==============================================================================
def calc_dist_metrics(returns_series):
    n = len(returns_series)
    mean_val = np.mean(returns_series)
    std_val = np.std(returns_series, ddof=1)
    median_val = np.median(returns_series)
    
    # Skewness & Kurtosis
    skew_val = (np.sum((returns_series - mean_val)**3) / n) / (std_val**3)
    kurt_val = (np.sum((returns_series - mean_val)**4) / n) / (std_val**4) - 3  # Excess kurtosis
    
    # Value at Risk 95% & CVaR 95% (1 ngày)
    var_95 = np.percentile(returns_series, 5)
    cvar_95 = returns_series[returns_series <= var_95].mean()
    
    return {
        'mean': mean_val,
        'std': std_val,
        'median': median_val,
        'skew': skew_val,
        'kurt': kurt_val,
        'var_95': var_95,
        'cvar_95': cvar_95
    }

# ==============================================================================
# FIGURE 1: EFFICIENT FRONTIER & CAPITAL ALLOCATION LINE (CAL)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(11.5, 7.2), dpi=100)

# Tách 2 nhánh:
# 1. Nhánh trên (Efficient Frontier): Từ điểm MVP lên đến 100% FPT (weights_fpt >= w_fpt_mvp)
eff_vols = port_volatilities[min_vol_idx:] * 100
eff_rets = port_returns[min_vol_idx:] * 100
eff_sharpes = port_sharpes[min_vol_idx:]

# 2. Nhánh dưới (Inefficient / Dominated): Từ 100% VNM đến điểm MVP (weights_fpt <= w_fpt_mvp)
ineff_vols = port_volatilities[:min_vol_idx+1] * 100
ineff_rets = port_returns[:min_vol_idx+1] * 100

# Vẽ đường Inefficient Frontier (Nét đứt màu xám)
ax1.plot(ineff_vols, ineff_rets, color='#b2bec3', linestyle='--', linewidth=2.5, 
         label='Dominated / Inefficient Branch (VNM Dominated)', zorder=2)

# Vẽ đường Efficient Frontier (Nét liền đậm có dải màu Sharpe Ratio)
scatter = ax1.scatter(eff_vols, eff_rets, c=eff_sharpes, cmap='viridis', s=35, 
                      alpha=0.9, edgecolors='none', label='Efficient Frontier (Upper Branch)', zorder=3)
cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
cbar.set_label(r'Sharpe Ratio ($R_f = 4.5\%$)', fontsize=11, fontweight='bold')

# Đường Capital Allocation Line (CAL) xuất phát từ Rf qua Tangency Portfolio
cal_vols = np.linspace(0, max(port_volatilities) * 1.15, 100)
cal_returns = rf_annual + sharpe_tan * cal_vols
ax1.plot(cal_vols * 100, cal_returns * 100, color='#e67e22', linestyle='-', linewidth=2.0, 
         label=f'Capital Allocation Line (CAL, Slope={sharpe_tan:.2f})', zorder=2)

# Đánh dấu 100% FPT (Nằm trên Efficient Frontier)
ax1.scatter(stock_vols['FPT.VN'] * 100, annual_returns['FPT.VN'] * 100,
            color='#d63031', marker='o', s=180, edgecolor='black', linewidth=1.5, zorder=6,
            label=f"100% FPT [On Efficient Frontier]: Ret={annual_returns['FPT.VN']*100:.1f}%, Vol={stock_vols['FPT.VN']*100:.1f}%")

# Đánh dấu 100% VNM (Nằm trên Inefficient Frontier)
ax1.scatter(stock_vols['VNM.VN'] * 100, annual_returns['VNM.VN'] * 100,
            color='#0984e3', marker='o', s=180, edgecolor='black', linewidth=1.5, zorder=6,
            label=f"100% VNM [Dominated / Inefficient]: Ret={annual_returns['VNM.VN']*100:.1f}%, Vol={stock_vols['VNM.VN']*100:.1f}%")

# Đánh dấu Minimum Variance Portfolio (MVP)
ax1.scatter(vol_mvp * 100, ret_mvp * 100,
            color='#00b894', marker='^', s=220, edgecolor='black', linewidth=1.5, zorder=7,
            label=f"Min Variance (MVP): FPT={w_fpt_mvp*100:.0f}%, VNM={w_vnm_mvp*100:.0f}% (Vol={vol_mvp*100:.1f}%)")

# Đánh dấu Tangency Portfolio (Max Sharpe)
ax1.scatter(vol_tan * 100, ret_tan * 100,
            color='#fdcb6e', marker='*', s=380, edgecolor='black', linewidth=1.5, zorder=7,
            label=f"Tangency (Max Sharpe={sharpe_tan:.2f}): Ret={ret_tan*100:.1f}%, Vol={vol_tan*100:.1f}%")

# Điểm Lãi suất phi rủi ro Rf
ax1.scatter(0, rf_annual * 100, color='#636e72', marker='s', s=100, zorder=5, label=f'Risk-free Rate $R_f$ ({rf_annual*100:.1f}%)')

# Annotations chi tiết giải thích rõ ràng
ax1.annotate(f"★ 100% FPT NẰM TRÊN EFFICIENT FRONTIER\n"
             f"Ret: {annual_returns['FPT.VN']*100:.1f}% | Vol: {stock_vols['FPT.VN']*100:.1f}%\n"
             f"Lợi nhuận cao nhất trong tập danh mục",
             xy=(stock_vols['FPT.VN'] * 100, annual_returns['FPT.VN'] * 100),
             xytext=(stock_vols['FPT.VN'] * 100 - 9.5, annual_returns['FPT.VN'] * 100 - 1.6),
             arrowprops=dict(facecolor='#d63031', arrowstyle='->', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#d63031', alpha=0.95),
             fontsize=9, fontweight='bold')

ax1.annotate(f"▲ MIN VARIANCE (MVP)\n"
             f"FPT: {w_fpt_mvp*100:.1f}% | VNM: {w_vnm_mvp*100:.1f}%\n"
             f"E(R): {ret_mvp*100:.1f}% | Vol: {vol_mvp*100:.1f}% (Rủi ro thấp nhất)",
             xy=(vol_mvp * 100, ret_mvp * 100), 
             xytext=(vol_mvp * 100 - 14.5, ret_mvp * 100 - 1.2),
             arrowprops=dict(facecolor='#00b894', arrowstyle='->', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f8f5', edgecolor='#00b894', alpha=0.95),
             fontsize=9, fontweight='bold')

ax1.annotate(f"● 100% VNM (INEFFICIENT / DOMINATED)\n"
             f"Ret: {annual_returns['VNM.VN']*100:.1f}% | Vol: {stock_vols['VNM.VN']*100:.1f}%\n"
             f"Bị thống trị bởi các danh mục ở nhánh trên",
             xy=(stock_vols['VNM.VN'] * 100, annual_returns['VNM.VN'] * 100),
             xytext=(stock_vols['VNM.VN'] * 100 - 13.5, annual_returns['VNM.VN'] * 100 - 0.8),
             arrowprops=dict(facecolor='#0984e3', arrowstyle='->', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ebf5fb', edgecolor='#0984e3', alpha=0.95),
             fontsize=8.5, fontweight='bold')

ax1.set_title('MARKOWITZ EFFICIENT FRONTIER (FPT & VNM - 3 YEARS: 2023-2026)', fontsize=13.5, fontweight='bold', pad=15)
ax1.set_xlabel('Annualized Volatility - Risk σ (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Annualized Expected Return E(R) (%)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)
ax1.set_xlim(left=-1, right=stock_vols['FPT.VN']*100 + 5)
ax1.set_ylim(bottom=-0.5, top=max(cal_returns)*100 + 1.0)
plt.tight_layout()
fig1.savefig('1_Efficient_Frontier.png', dpi=300)
print("[+] Đã lưu biểu đồ: 1_Efficient_Frontier.png")

# ==============================================================================
# FIGURE 2: RETURN DISTRIBUTION ANALYSIS & RISK PROFILE
# ==============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6), dpi=100)

colors = {'FPT.VN': '#d63031', 'VNM.VN': '#0984e3'}

for idx, ticker in enumerate(stock_tickers):
    ax = axes2[idx]
    rets = stock_returns[ticker] * 100  # Chuyển về %
    m = calc_dist_metrics(rets)
    color = colors[ticker]
    
    # 1. Histogram
    count, bins, _ = ax.hist(rets, bins=50, density=True, alpha=0.45, color=color, 
                             edgecolor='white', linewidth=0.5, label='Actual Returns')
    
    # 2. Fitted Gaussian Curve (Phân phối chuẩn lý thuyết)
    x_axis = np.linspace(rets.min(), rets.max(), 300)
    normal_pdf = (1.0 / (m['std'] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - m['mean']) / m['std']) ** 2)
    ax.plot(x_axis, normal_pdf, color='black', linestyle='--', linewidth=1.8, label='Normal Distribution Fit')
    
    # 3. Đường thẳng đánh dấu Mean, Median và VaR 95%
    ax.axvline(m['mean'], color='green', linestyle='-', linewidth=1.5, label=f"Mean: {m['mean']:.2f}%")
    ax.axvline(m['median'], color='purple', linestyle=':', linewidth=1.5, label=f"Median: {m['median']:.2f}%")
    ax.axvline(m['var_95'], color='red', linestyle='-', linewidth=2.0, label=f"VaR 95% (1-Day): {m['var_95']:.2f}%")
    
    # Tô màu vùng rủi ro đuôi (Tail Risk: < VaR 95%)
    tail_x = np.linspace(rets.min(), m['var_95'], 100)
    tail_norm = (1.0 / (m['std'] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((tail_x - m['mean']) / m['std']) ** 2)
    ax.fill_between(tail_x, 0, tail_norm, color='red', alpha=0.3, label='Tail Risk (< VaR 95%)')
    
    # Bảng số liệu thống kê trực tiếp trên đồ thị
    stat_box = (
        f"STATISTICAL METRICS\n"
        f"────────────────────────\n"
        f"Ann. Return:   {annual_returns[ticker]*100:>6.2f}%\n"
        f"Ann. Volatility: {stock_vols[ticker]*100:>5.2f}%\n"
        f"Daily Std Dev:  {m['std']:>6.2f}%\n"
        f"Skewness:       {m['skew']:>6.2f}\n"
        f"Excess Kurtosis:{m['kurt']:>6.2f}\n"
        f"VaR 95% (1D):   {m['var_95']:>6.2f}%\n"
        f"CVaR 95% (1D):  {m['cvar_95']:>6.2f}%"
    )
    ax.text(0.96, 0.95, stat_box, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.92))
    
    ax.set_title(f'Daily Return Distribution: {ticker}', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Daily Return (%)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=10.5, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    ax.set_xlim(left=min(rets.min(), -5), right=max(rets.max(), 5))

plt.tight_layout()
fig2.savefig('2_Return_Distribution.png', dpi=300)
print("[+] Đã lưu biểu đồ: 2_Return_Distribution.png")

# ==============================================================================
# FIGURE 3: MÔ HÌNH CAPM & PHÂN TÍCH ĐỘ NHẠY BETA / ALPHA DANH MỤC (WEIGHT 0 -> 1)
# ==============================================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(15.5, 6.5), dpi=100)

# Panel A: Security Characteristic Line (SCL) - Beta Regression cho FPT & VNM
ax_scl = axes3[0]
for ticker in stock_tickers:
    excess_stock = (stock_returns[ticker] - rf_daily) * 100
    excess_mkt = excess_market * 100
    color = colors[ticker]
    st = capm_stats[ticker]
    
    # Scatter plot
    ax_scl.scatter(excess_mkt, excess_stock, alpha=0.35, s=20, color=color, 
                   label=f"{ticker} Returns")
    
    # Regression line
    x_line = np.linspace(excess_mkt.min(), excess_mkt.max(), 100)
    y_line = (st['alpha_annual'] / trading_days) * 100 + st['beta'] * x_line
    ax_scl.plot(x_line, y_line, color=color, linewidth=2.2,
                label=f"{ticker} Fit: β={st['beta']:.2f}, α={st['alpha_annual']*100:+.2f}%, R²={st['r_squared']:.2f}")

ax_scl.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax_scl.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax_scl.set_title('Security Characteristic Line (SCL) - Beta Regression', fontsize=13, fontweight='bold', pad=12)
ax_scl.set_xlabel('Market Excess Return: $R_m - R_f$ (%)', fontsize=10.5, fontweight='bold')
ax_scl.set_ylabel('Stock Excess Return: $R_i - R_f$ (%)', fontsize=10.5, fontweight='bold')
ax_scl.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

# Panel B: Phân tích độ nhạy rủi ro hệ thống Beta & Alpha của danh mục theo tỷ trọng FPT (0 -> 1)
ax_sens = axes3[1]
color_beta = '#2980b9'
color_alpha = '#e74c3c'

ax_sens.plot(weights_capm * 100, port_betas, color=color_beta, linewidth=2.5, label=r'Portfolio Beta $\beta_p(w)$')
ax_sens.set_xlabel('Tỷ trọng FPT trong danh mục $w_{FPT}$ (%)', fontsize=10.5, fontweight='bold')
ax_sens.set_ylabel(r'Hệ số Beta danh mục ($\beta_p$)', color=color_beta, fontsize=10.5, fontweight='bold')
ax_sens.tick_params(axis='y', labelcolor=color_beta)
ax_sens.grid(True, linestyle='--', alpha=0.6)

# Trục thứ 2 cho Alpha danh mục
ax_sens_right = ax_sens.twinx()
ax_sens_right.plot(weights_capm * 100, port_alphas * 100, color=color_alpha, linewidth=2.5, linestyle='-.', label=r"Jensen's Alpha $\alpha_p(w)$ (%)")
ax_sens_right.set_ylabel("Jensen's Alpha danh mục (%)", color=color_alpha, fontsize=10.5, fontweight='bold')
ax_sens_right.tick_params(axis='y', labelcolor=color_alpha)
ax_sens_right.grid(False)

# Đánh dấu các mốc trọng số đặc biệt
ax_sens.scatter(w_fpt_mvp * 100, beta_mvp, color='#00b894', s=130, marker='^', zorder=5, label=f'MVP (w={w_fpt_mvp*100:.0f}%, β={beta_mvp:.2f})')
ax_sens.scatter(w_fpt_tan * 100, beta_tan, color='#fdcb6e', s=180, marker='*', edgecolor='black', zorder=5, label=f'Tangency (w={w_fpt_tan*100:.0f}%, β={beta_tan:.2f})')
ax_sens.scatter(0, beta_vnm, color=colors['VNM.VN'], s=100, marker='o', zorder=5, label=f'100% VNM (β={beta_vnm:.2f})')
ax_sens.scatter(100, beta_fpt, color=colors['FPT.VN'], s=100, marker='o', zorder=5, label=f'100% FPT (β={beta_fpt:.2f})')

ax_sens.set_title('Portfolio Systematic Risk (Beta) & Alpha vs Weight (0% -> 100%)', fontsize=13, fontweight='bold', pad=12)

# Ghép legend của cả 2 trục
lines1, labels1 = ax_sens.get_legend_handles_labels()
lines2, labels2 = ax_sens_right.get_legend_handles_labels()
ax_sens.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

plt.tight_layout()
fig3.savefig('3_CAPM_Model_and_Portfolio_Beta.png', dpi=300)
print("[+] Đã lưu biểu đồ: 3_CAPM_Model_and_Portfolio_Beta.png")

# ==============================================================================
# FIGURE 4: MÔ HÌNH SECURITY MARKET LINE (SML) & ĐỊNH GIÁ DANH MỤC (WEIGHT 0 -> 1)
# ==============================================================================
fig4, ax_sml = plt.subplots(figsize=(12, 7.2), dpi=100)

# 1. Đường SML lý thuyết: E(R) = Rf + Beta * [E(Rm) - Rf]
betas_sml = np.linspace(0, 1.5, 100)
sml_returns = rf_annual + betas_sml * (market_annual_return - rf_annual)
ax_sml.plot(betas_sml, sml_returns * 100, color='#2c3e50', linewidth=2.5, 
            label=f'Security Market Line (SML)\n$E(R) = R_f + \\beta \\cdot [E(R_m) - R_f]$')

# Tô màu phân chia vùng Undervalued / Overvalued
ax_sml.fill_between(betas_sml, sml_returns * 100, sml_returns * 100 + 40, color='#2ecc71', alpha=0.08, label='Undervalued Region (Alpha > 0 - Vượt trội)')
ax_sml.fill_between(betas_sml, sml_returns * 100 - 30, sml_returns * 100, color='#e74c3c', alpha=0.08, label='Overvalued Region (Alpha < 0 - Kém hấp dẫn)')

# 2. Quỹ đạo danh mục đầu tư FPT - VNM khi weight chạy từ 0 đến 1
ax_sml.plot(port_betas, port_actual_capm_rets * 100, color='#7f8c8d', linestyle='--', linewidth=2.0, zorder=3,
            label='Đường tỷ trọng danh mục (w từ 0% đến 100%)')
scatter_port = ax_sml.scatter(port_betas, port_actual_capm_rets * 100, c=weights_capm * 100, cmap='coolwarm', 
                              s=45, alpha=0.85, zorder=4, edgecolor='none')
cbar_sml = plt.colorbar(scatter_port, ax=ax_sml, pad=0.02)
cbar_sml.set_label('Tỷ trọng cổ phiếu FPT ($w_{FPT}$ %)', fontsize=10.5, fontweight='bold')

# 3. Điểm Benchmark Market (Beta=1.0) và Risk-free (Beta=0.0)
ax_sml.scatter(1.0, market_annual_return * 100, color='#8e44ad', marker='D', s=130, zorder=6, 
               label=f'Market Benchmark (VN30): β=1.0, Ret={market_annual_return*100:.1f}%')
ax_sml.scatter(0.0, rf_annual * 100, color='#636e72', marker='s', s=110, zorder=6, 
               label=f'Risk-Free Asset ($R_f$): β=0.0, Ret={rf_annual*100:.1f}%')

# 4. Đánh dấu FPT (100%), VNM (100%), MVP và Tangency Portfolio
ret_fpt = annual_returns['FPT.VN'] * 100
exp_capm_fpt = capm_stats['FPT.VN']['expected_capm'] * 100
alpha_fpt_val = ret_fpt - exp_capm_fpt
ax_sml.scatter(beta_fpt, ret_fpt, color=colors['FPT.VN'], s=180, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"100% FPT: β={beta_fpt:.2f}, Ret={ret_fpt:.1f}% (α={alpha_fpt_val:+.1f}%)")
ax_sml.plot([beta_fpt, beta_fpt], [exp_capm_fpt, ret_fpt], color=colors['FPT.VN'], linestyle=':', linewidth=2.2)

ret_vnm = annual_returns['VNM.VN'] * 100
exp_capm_vnm = capm_stats['VNM.VN']['expected_capm'] * 100
alpha_vnm_val = ret_vnm - exp_capm_vnm
ax_sml.scatter(beta_vnm, ret_vnm, color=colors['VNM.VN'], s=180, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"100% VNM: β={beta_vnm:.2f}, Ret={ret_vnm:.1f}% (α={alpha_vnm_val:+.1f}%)")
ax_sml.plot([beta_vnm, beta_vnm], [exp_capm_vnm, ret_vnm], color=colors['VNM.VN'], linestyle=':', linewidth=2.2)

ret_mvp_pct = ret_mvp * 100
exp_capm_mvp_pct = capm_ret_mvp * 100
alpha_mvp_pct = ret_mvp_pct - exp_capm_mvp_pct
ax_sml.scatter(beta_mvp, ret_mvp_pct, color='#00b894', marker='^', s=200, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"MVP Portfolio (w_FPT={w_fpt_mvp*100:.0f}%): β={beta_mvp:.2f}, Ret={ret_mvp_pct:.1f}%")
ax_sml.plot([beta_mvp, beta_mvp], [exp_capm_mvp_pct, ret_mvp_pct], color='#00b894', linestyle=':', linewidth=2.2)

ret_tan_pct = ret_tan * 100
exp_capm_tan_pct = capm_ret_tan * 100
alpha_tan_pct = ret_tan_pct - exp_capm_tan_pct
ax_sml.scatter(beta_tan, ret_tan_pct, color='#fdcb6e', marker='*', s=350, edgecolor='black', linewidth=1.5, zorder=8,
               label=f"Tangency Portfolio (w_FPT={w_fpt_tan*100:.0f}%): β={beta_tan:.2f}, Ret={ret_tan_pct:.1f}%")
ax_sml.plot([beta_tan, beta_tan], [exp_capm_tan_pct, ret_tan_pct], color='#e67e22', linestyle=':', linewidth=2.2)

# Annotations vị trí định giá
ax_sml.annotate(f"FPT (α={alpha_fpt_val:+.1f}%)\n[{'Undervalued' if alpha_fpt_val > 0 else 'Overvalued'}]",
                xy=(beta_fpt, ret_fpt), xytext=(beta_fpt + 0.05, ret_fpt + 2.5),
                arrowprops=dict(facecolor=colors['FPT.VN'], arrowstyle='->', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor=colors['FPT.VN'], alpha=0.92),
                fontsize=9, fontweight='bold')

ax_sml.annotate(f"VNM (α={alpha_vnm_val:+.1f}%)\n[{'Undervalued' if alpha_vnm_val > 0 else 'Overvalued'}]",
                xy=(beta_vnm, ret_vnm), xytext=(beta_vnm - 0.22, ret_vnm - 5.5),
                arrowprops=dict(facecolor=colors['VNM.VN'], arrowstyle='->', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor=colors['VNM.VN'], alpha=0.92),
                fontsize=9, fontweight='bold')

ax_sml.annotate(f"Tangency Portfolio\nw_FPT={w_fpt_tan*100:.0f}%, β={beta_tan:.2f}\nα={alpha_tan_pct:+.1f}%",
                xy=(beta_tan, ret_tan_pct), xytext=(beta_tan + 0.06, ret_tan_pct - 3.5),
                arrowprops=dict(facecolor='#d35400', arrowstyle='->', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#d35400', alpha=0.92),
                fontsize=8.5, fontweight='bold')

ax_sml.set_title("SECURITY MARKET LINE (SML) & VALUATION OF PORTFOLIOS (WEIGHT 0% -> 100%)", fontsize=13, fontweight='bold', pad=14)
ax_sml.set_xlabel(r'Rủi ro hệ thống - Systematic Risk Beta ($\beta$)', fontsize=11, fontweight='bold')
ax_sml.set_ylabel('Tỷ suất sinh lời kỳ vọng / thực tế E(R) (%)', fontsize=11, fontweight='bold')
ax_sml.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)
ax_sml.set_xlim(left=-0.08, right=1.45)
ax_sml.set_ylim(bottom=min(-5.0, ret_vnm - 8), top=max(market_annual_return*100, ret_fpt) + 12)

plt.tight_layout()
fig4.savefig('4_Security_Market_Line_SML.png', dpi=300)
print("[+] Đã lưu biểu đồ: 4_Security_Market_Line_SML.png")

# ==============================================================================
# FIGURE 5: BẢNG TRỰC QUAN HÓA COVARIANCE & CORRELATION MATRIX
# ==============================================================================
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6.2), dpi=100)

labels = stock_tickers
cov_vals = cov_matrix.values
corr_vals = corr_matrix.values

# 1. Heatmap Ma trận hiệp phương sai năm hóa (Annualized Covariance)
im1 = axes5[0].imshow(cov_vals, cmap='Blues', alpha=0.85)
cbar1 = fig5.colorbar(im1, ax=axes5[0], fraction=0.046, pad=0.04)
cbar1.set_label('Covariance Value', fontsize=10, fontweight='bold')

axes5[0].set_xticks([0, 1])
axes5[0].set_yticks([0, 1])
axes5[0].set_xticklabels(labels, fontsize=11, fontweight='bold')
axes5[0].set_yticklabels(labels, fontsize=11, fontweight='bold')
axes5[0].set_title('Annualized Covariance Matrix (Σ)', fontsize=13, fontweight='bold', pad=12)

# Hiển thị số liệu chi tiết trong từng ô
for i in range(2):
    for j in range(2):
        val = cov_vals[i, j]
        desc = "Variance (σ²)" if i == j else "Covariance (Cov)"
        text_color = 'white' if val > 0.06 else 'black'
        axes5[0].text(j, i, f"{desc}\n{val:.6f}\n({np.sqrt(val)*100:.2f}% Vol)" if i == j else f"{desc}\n{val:.6f}",
                      ha='center', va='center', color=text_color, fontsize=11, fontweight='bold')

# 2. Heatmap Ma trận tương quan (Correlation Matrix)
im2 = axes5[1].imshow(corr_vals, cmap='YlGnBu', vmin=0, vmax=1, alpha=0.85)
cbar2 = fig5.colorbar(im2, ax=axes5[1], fraction=0.046, pad=0.04)
cbar2.set_label('Correlation Coefficient (r)', fontsize=10, fontweight='bold')

axes5[1].set_xticks([0, 1])
axes5[1].set_yticks([0, 1])
axes5[1].set_xticklabels(labels, fontsize=11, fontweight='bold')
axes5[1].set_yticklabels(labels, fontsize=11, fontweight='bold')
axes5[1].set_title('Correlation Matrix & Diversification', fontsize=13, fontweight='bold', pad=12)

for i in range(2):
    for j in range(2):
        val = corr_vals[i, j]
        text_color = 'white' if val > 0.6 else 'black'
        note = "Perfect Self-Corr" if i == j else f"Low Correlation\nr = {val:.4f}"
        axes5[1].text(j, i, f"{val:.4f}\n({note})",
                      ha='center', va='center', color=text_color, fontsize=11, fontweight='bold')

# Bổ sung giải thích tài chính cho Analyst
fig5.subplots_adjust(bottom=0.22, top=0.88, wspace=0.3)
diversification_text = (
    f"PORTFOLIO DIVERSIFICATION INSIGHT:\n"
    f"• Correlation giữa FPT và VNM là r = {corr_fpt_vnm:.4f} (ở mức thấp/trung bình).\n"
    f"• Do r < 1.0, việc kết hợp FPT & VNM tạo ra hiệu ứng đa dạng hóa (Diversification Benefit),\n"
    f"  giúp rủi ro danh mục MVP ({vol_mvp*100:.2f}%) thấp hơn độ lệch chuẩn của cả FPT ({stock_vols['FPT.VN']*100:.2f}%) và VNM ({stock_vols['VNM.VN']*100:.2f}%)."
)
fig5.text(0.5, 0.03, diversification_text, ha='center', va='bottom', fontsize=9.5,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95))

fig5.savefig('5_Covariance_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
print("[+] Đã lưu biểu đồ: 5_Covariance_Correlation_Matrix.png")

# ==============================================================================
# BÁO CÁO TỔNG HỢP CHO ANALYST (CONCISE EXECUTIVE SUMMARY)
# ==============================================================================
print("\n" + "="*70)
print("             EXECUTIVE PORTFOLIO ANALYST REPORT")
print("="*70)
print(f"1. OPTIMAL ALLOCATION (Tangency Portfolio):")
print(f"   - Tỷ trọng khuyến nghị: FPT = {w_fpt_tan*100:.1f}% | VNM = {w_vnm_tan*100:.1f}%")
print(f"   - Lợi nhuận kỳ vọng:    {ret_tan*100:.2f}%/năm")
print(f"   - Mức độ rủi ro (Vol):  {vol_tan*100:.2f}%/năm")
print(f"   - Hệ số Sharpe:         {sharpe_tan:.2f}")

print(f"\n2. MINIMUM RISK ALLOCATION (Minimum Variance Portfolio - MVP):")
print(f"   - Tỷ trọng an toàn:     FPT = {w_fpt_mvp*100:.1f}% | VNM = {w_vnm_mvp*100:.1f}%")
print(f"   - Mức độ rủi ro tối thiểu: {vol_mvp*100:.2f}%/năm (Lợi nhuận: {ret_mvp*100:.2f}%)")

print(f"\n3. COVARIANCE & CORRELATION ANALYSIS (3-Year):")
print(f"   - Annualized Covariance (FPT, VNM): {cov_fpt_vnm_annual:.6f}")
print(f"   - Correlation Coefficient (r):     {corr_fpt_vnm:.4f}")
print(f"   - Diversification Benefit:         Đạt hiệu ứng giảm rủi ro vượt trội (MVP Vol = {vol_mvp*100:.2f}%)")

print(f"\n4. CAPM & SYSTEMATIC RISK (BETA) ANALYSIS:")
for ticker in stock_tickers:
    st = capm_stats[ticker]
    print(f"   - {ticker}:")
    print(f"       + Hệ số Beta (β):     {st['beta']:.2f} (Đo lường rủi ro hệ thống so với VN30)")
    print(f"       + Jensen's Alpha (α):  {st['alpha_annual']*100:+.2f}%/năm")
    print(f"       + Hệ số R²:           {st['r_squared']:.2f}")
print(f"   - Danh mục Tangency (FPT = {w_fpt_tan*100:.0f}%, VNM = {w_vnm_tan*100:.0f}%):")
print(f"       + Hệ số Beta (β_p):    {beta_tan:.2f}")
print(f"       + Jensen's Alpha (α_p): {alpha_tan*100:+.2f}%/năm")
print(f"   - Danh mục MVP (FPT = {w_fpt_mvp*100:.0f}%, VNM = {w_vnm_mvp*100:.0f}%):")
print(f"       + Hệ số Beta (β_p):    {beta_mvp:.2f}")
print(f"       + Jensen's Alpha (α_p): {alpha_mvp*100:+.2f}%/năm")
print("="*70)

# ==============================================================================
# HIỂN THỊ TẤT CẢ CÁC BIỂU ĐỒ (FIGURES)
# ==============================================================================
print("\n[+] Đang mở 5 cửa sổ hiển thị đồ thị (Figures) trực quan trên màn hình...")
plt.show()


