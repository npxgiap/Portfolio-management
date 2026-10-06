"""
========================================================================================
CHARTS MODULE: ADVANCED QUANTITATIVE & ECONOMETRIC VISUALIZATION
========================================================================================
Module trực quan hóa định lượng cao cấp tích hợp:
  - Scipy: Vẽ đường Kernel Density Estimation (KDE), Gaussian Normal Fit, Cornish-Fisher VaR.
  - Statsmodels: Vẽ đường hồi quy OLS và Dải khoảng tin cậy 95% (95% Confidence Band),
    bảng tham số kinh tế lượng OLS (t-stat, p-value, R², F-stat).
  - Matplotlib: Tối ưu hóa đồ họa chuyên nghiệp, trả về đối tượng Figure vector.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#e0e0e0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7

setup_plot_style()

def create_fig_efficient_frontier(data):
    """1. Đồ thị Đường biên hiệu quả Markowitz (SLSQP Scipy) & Đường phân bổ vốn CAL"""
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=100)
    
    min_vol_idx = data['min_vol_idx']
    port_volatilities = data['port_volatilities']
    port_returns = data['port_returns']
    port_sharpes = data['port_sharpes']
    rf_annual = data['rf_annual']
    sharpe_tan = data['sharpe_tan']
    stock_vols = data['stock_vols']
    annual_returns = data['annual_returns']
    vol_mvp = data['vol_mvp']
    ret_mvp = data['ret_mvp']
    vol_tan = data['vol_tan']
    ret_tan = data['ret_tan']
    w_fpt_mvp = data['w_fpt_mvp']
    w_vnm_mvp = data['w_vnm_mvp']
    w_fpt_tan = data['w_fpt_tan']
    w_vnm_tan = data['w_vnm_tan']

    # 1. Nhánh trên (Efficient Frontier) & Nhánh dưới (Inefficient)
    eff_vols = port_volatilities[min_vol_idx:] * 100
    eff_rets = port_returns[min_vol_idx:] * 100
    eff_sharpes = port_sharpes[min_vol_idx:]

    ineff_vols = port_volatilities[:min_vol_idx+1] * 100
    ineff_rets = port_returns[:min_vol_idx+1] * 100

    ax.plot(ineff_vols, ineff_rets, color='#b2bec3', linestyle='--', linewidth=2.2, 
            label='Dominated / Inefficient Branch', zorder=2)

    scatter = ax.scatter(eff_vols, eff_rets, c=eff_sharpes, cmap='viridis', s=38, 
                         alpha=0.9, edgecolors='none', label='Efficient Frontier (Upper Branch)', zorder=3)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(f'Sharpe Ratio ($R_f = {rf_annual*100:.1f}\\%$)', fontsize=10.5, fontweight='bold')

    # Đường Capital Allocation Line (CAL)
    cal_vols = np.linspace(0, max(port_volatilities) * 1.15, 100)
    cal_returns = rf_annual + sharpe_tan * cal_vols
    ax.plot(cal_vols * 100, cal_returns * 100, color='#e67e22', linestyle='-', linewidth=2.2, 
            label=f'Capital Allocation Line (CAL, Slope={sharpe_tan:.2f})', zorder=2)

    # Đánh dấu các điểm
    ax.scatter(stock_vols['FPT.VN'] * 100, annual_returns['FPT.VN'] * 100,
               color='#d63031', marker='o', s=170, edgecolor='black', linewidth=1.5, zorder=6,
               label=f"100% FPT: Ret={annual_returns['FPT.VN']*100:.1f}%, Vol={stock_vols['FPT.VN']*100:.1f}%")
    ax.scatter(stock_vols['VNM.VN'] * 100, annual_returns['VNM.VN'] * 100,
               color='#0984e3', marker='o', s=170, edgecolor='black', linewidth=1.5, zorder=6,
               label=f"100% VNM: Ret={annual_returns['VNM.VN']*100:.1f}%, Vol={stock_vols['VNM.VN']*100:.1f}%")
    
    # MVP tối ưu chính xác bằng Scipy SLSQP
    ax.scatter(vol_mvp * 100, ret_mvp * 100,
               color='#00b894', marker='^', s=220, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"MVP [Scipy SLSQP] (FPT={w_fpt_mvp*100:.1f}%, VNM={w_vnm_mvp*100:.1f}%): Vol={vol_mvp*100:.1f}%")
    
    # Tangency Portfolio tối ưu chính xác bằng Scipy SLSQP
    ax.scatter(vol_tan * 100, ret_tan * 100,
               color='#fdcb6e', marker='*', s=380, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"Tangency [Scipy SLSQP] (FPT={w_fpt_tan*100:.1f}%, VNM={w_vnm_tan*100:.1f}%): SR={sharpe_tan:.2f}")
    
    ax.scatter(0, rf_annual * 100, color='#636e72', marker='s', s=110, zorder=5, 
               label=f'Risk-free Rate $R_f$ ({rf_annual*100:.1f}%)')

    ax.set_title('MARKOWITZ EFFICIENT FRONTIER & CAPITAL ALLOCATION LINE (SCIPY SLSQP OPTIMIZED)', 
                 fontsize=12.5, fontweight='bold', pad=14)
    ax.set_xlabel('Annualized Volatility - Risk σ (%)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Annualized Expected Return E(R) (%)', fontsize=10.5, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.5)
    ax.set_xlim(left=-1, right=stock_vols['FPT.VN']*100 + 5)
    ax.set_ylim(bottom=-0.5, top=max(cal_returns)*100 + 1.0)
    fig.tight_layout()
    return fig

def create_fig_return_distribution(data):
    """2. Đồ thị Phân phối lợi nhuận, KDE (Scipy Gaussian KDE), Gaussian Fit, VaR & CVaR"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=100)
    colors = {'FPT.VN': '#d63031', 'VNM.VN': '#0984e3'}
    stock_tickers = data['stock_tickers']
    annual_returns = data['annual_returns']
    stock_vols = data['stock_vols']

    for idx, ticker in enumerate(stock_tickers):
        ax = axes[idx]
        m = data['dist_metrics'][ticker]
        rets = m['rets_pct']
        color = colors[ticker]
        
        # 1. Histogram tần suất thực nghiệm
        ax.hist(rets, bins=50, density=True, alpha=0.35, color=color, 
                edgecolor='white', linewidth=0.5, label='Actual Returns (Histogram)')
        
        # 2. Kernel Density Estimation (KDE) từ scipy.stats
        ax.plot(m['kde_x'], m['kde_pdf'], color=color, linewidth=2.4, 
                label='Empirical KDE (Scipy gaussian_kde)')
        
        # 3. Phân phối chuẩn Gauss lý thuyết từ scipy.stats.norm
        ax.plot(m['kde_x'], m['norm_pdf'], color='#2c3e50', linestyle='--', linewidth=1.8, 
                label='Fitted Normal Distribution')
        
        # 4. Vertical markers: Mean, Median, VaR
        ax.axvline(m['mean'], color='green', linestyle='-', linewidth=1.5, label=f"Mean: {m['mean']:.2f}%")
        ax.axvline(m['median'], color='purple', linestyle=':', linewidth=1.5, label=f"Median: {m['median']:.2f}%")
        ax.axvline(m['var_95'], color='red', linestyle='-', linewidth=2.0, label=f"Hist VaR 95%: {m['var_95']:.2f}%")
        ax.axvline(m['var_cf_95'], color='#d35400', linestyle='-.', linewidth=1.6, label=f"Cornish-Fisher VaR: {m['var_cf_95']:.2f}%")
        
        # Shading vùng đuôi rủi ro
        tail_mask = m['kde_x'] <= m['var_95']
        ax.fill_between(m['kde_x'][tail_mask], 0, m['kde_pdf'][tail_mask], color='red', alpha=0.3, label='Tail Risk (< VaR 95%)')
        
        # Hộp thông số thống kê & Kiểm định Jarque-Bera từ scipy.stats
        jb_status = "Non-Normal (p<0.01)" if m['jb_pvalue'] < 0.01 else "Normal"
        stat_box = (
            f"ECONOMETRIC & RISK METRICS\n"
            f"─────────────────────────────\n"
            f"Ann. Return:     {annual_returns[ticker]*100:>6.2f}%\n"
            f"Ann. Volatility: {stock_vols[ticker]*100:>5.2f}%\n"
            f"Daily Std Dev:    {m['std']:>6.2f}%\n"
            f"Skewness:         {m['skew']:>6.2f}\n"
            f"Excess Kurtosis:  {m['kurt']:>6.2f}\n"
            f"Jarque-Bera Stat: {m['jb_stat']:>6.1f}\n"
            f"JB Test p-value:  {m['jb_pvalue']:>6.2e}\n"
            f"Distribution:     {jb_status}\n"
            f"Hist VaR 95%:     {m['var_95']:>6.2f}%\n"
            f"Cornish-Fisher:   {m['var_cf_95']:>6.2f}%\n"
            f"CVaR 95% (1D):    {m['cvar_95']:>6.2f}%"
        )
        ax.text(0.96, 0.95, stat_box, transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', fontsize=8.0, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.92))
        
        ax.set_title(f'Return Distribution & Risk Modeling: {ticker}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=7.8)
        ax.set_xlim(left=min(rets.min(), -5.5), right=max(rets.max(), 5.5))

    fig.tight_layout()
    return fig

def create_fig_capm_regression(data):
    """3. Mô hình CAPM: Hồi quy OLS Statsmodels với 95% Confidence Band & Độ nhạy Portfolio Beta"""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=100)
    stock_tickers = data['stock_tickers']
    stock_returns = data['stock_returns']
    rf_daily = data['rf_daily']
    excess_market = data['excess_market']
    capm_stats = data['capm_stats']
    colors = {'FPT.VN': '#d63031', 'VNM.VN': '#0984e3'}

    # Panel A: Hồi quy OLS Statsmodels và Dải khoảng tin cậy 95%
    ax_scl = axes[0]
    for ticker in stock_tickers:
        excess_stock = (stock_returns[ticker] - rf_daily) * 100
        excess_mkt = excess_market * 100
        color = colors[ticker]
        st = capm_stats[ticker]
        
        # Scatter điểm dữ liệu thực tế
        ax_scl.scatter(excess_mkt, excess_stock, alpha=0.35, s=20, color=color, label=f"{ticker} Excess Returns")
        
        # Đường hồi quy OLS Statsmodels
        x_pts = st['x_grid'] * 100
        y_pts = (st['alpha_annual'] / data['trading_days'] + st['beta'] * st['x_grid']) * 100
        ax_scl.plot(x_pts, y_pts, color=color, linewidth=2.2,
                    label=f"{ticker} OLS: β={st['beta']:.2f} (p={st['beta_pvalue']:.1e}), R²={st['r_squared']:.2f}")
        
        # Dải khoảng tin cậy 95% (Confidence Band) từ statsmodels get_prediction
        ci_lower = st['pred_summary']['mean_ci_lower'] * 100
        ci_upper = st['pred_summary']['mean_ci_upper'] * 100
        ax_scl.fill_between(x_pts, ci_lower, ci_upper, color=color, alpha=0.15, label=f"{ticker} 95% Confidence Band")

    ax_scl.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_scl.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_scl.set_title('Security Characteristic Line (SCL) - Statsmodels OLS with 95% CI', fontsize=12, fontweight='bold', pad=10)
    ax_scl.set_xlabel('Market Excess Return: $R_m - R_f$ (%)', fontsize=10, fontweight='bold')
    ax_scl.set_ylabel('Stock Excess Return: $R_i - R_f$ (%)', fontsize=10, fontweight='bold')
    ax_scl.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.2)

    # Panel B: Phân tích độ nhạy Portfolio Beta & Alpha theo weight (0 -> 1)
    ax_sens = axes[1]
    color_beta = '#2980b9'
    color_alpha = '#e74c3c'
    weights_capm = data['weights_capm']
    port_betas = data['port_betas']
    port_alphas = data['port_alphas']
    w_fpt_mvp = data['w_fpt_mvp']
    beta_mvp = data['beta_mvp']
    w_fpt_tan = data['w_fpt_tan']
    beta_tan = data['beta_tan']
    beta_vnm = data['beta_vnm']
    beta_fpt = data['beta_fpt']

    ax_sens.plot(weights_capm * 100, port_betas, color=color_beta, linewidth=2.5, label=r'Portfolio Beta $\beta_p(w)$')
    ax_sens.set_xlabel('Tỷ trọng FPT trong danh mục $w_{FPT}$ (%)', fontsize=10.5, fontweight='bold')
    ax_sens.set_ylabel(r'Hệ số Beta danh mục ($\beta_p$)', color=color_beta, fontsize=10.5, fontweight='bold')
    ax_sens.tick_params(axis='y', labelcolor=color_beta)
    ax_sens.grid(True, linestyle='--', alpha=0.6)

    ax_sens_right = ax_sens.twinx()
    ax_sens_right.plot(weights_capm * 100, port_alphas * 100, color=color_alpha, linewidth=2.5, linestyle='-.', label=r"Jensen's Alpha $\alpha_p(w)$ (%)")
    ax_sens_right.set_ylabel("Jensen's Alpha danh mục (%)", color=color_alpha, fontsize=10.5, fontweight='bold')
    ax_sens_right.tick_params(axis='y', labelcolor=color_alpha)
    ax_sens_right.grid(False)

    ax_sens.scatter(w_fpt_mvp * 100, beta_mvp, color='#00b894', s=130, marker='^', zorder=5, label=f'MVP (w={w_fpt_mvp*100:.0f}%, β={beta_mvp:.2f})')
    ax_sens.scatter(w_fpt_tan * 100, beta_tan, color='#fdcb6e', s=180, marker='*', edgecolor='black', zorder=5, label=f'Tangency (w={w_fpt_tan*100:.0f}%, β={beta_tan:.2f})')
    ax_sens.scatter(0, beta_vnm, color=colors['VNM.VN'], s=90, marker='o', zorder=5, label=f'100% VNM (β={beta_vnm:.2f})')
    ax_sens.scatter(100, beta_fpt, color=colors['FPT.VN'], s=90, marker='o', zorder=5, label=f'100% FPT (β={beta_fpt:.2f})')

    ax_sens.set_title('Portfolio Systematic Risk (Beta) & Alpha vs Weight (0% -> 100%)', fontsize=12, fontweight='bold', pad=10)
    lines1, labels1 = ax_sens.get_legend_handles_labels()
    lines2, labels2 = ax_sens_right.get_legend_handles_labels()
    ax_sens.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.0)

    fig.tight_layout()
    return fig

def create_fig_sml(data):
    """4. Mô hình SML: Security Market Line & Định giá danh mục với kiểm định Alpha"""
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=100)
    rf_annual = data['rf_annual']
    market_annual_return = data['market_annual_return']
    port_betas = data['port_betas']
    port_actual_capm_rets = data['port_actual_capm_rets']
    weights_capm = data['weights_capm']
    beta_fpt = data['beta_fpt']
    beta_vnm = data['beta_vnm']
    beta_mvp = data['beta_mvp']
    beta_tan = data['beta_tan']
    annual_returns = data['annual_returns']
    capm_stats = data['capm_stats']
    colors = {'FPT.VN': '#d63031', 'VNM.VN': '#0984e3'}

    # 1. Đường SML lý thuyết
    betas_sml = np.linspace(0, 1.5, 100)
    sml_returns = rf_annual + betas_sml * (market_annual_return - rf_annual)
    ax.plot(betas_sml, sml_returns * 100, color='#2c3e50', linewidth=2.5, 
            label=f'Security Market Line (SML)\n$E(R) = R_f + \\beta \\cdot [E(R_m) - R_f]$')

    ax.fill_between(betas_sml, sml_returns * 100, sml_returns * 100 + 40, color='#2ecc71', alpha=0.08, label='Undervalued Region (Alpha > 0)')
    ax.fill_between(betas_sml, sml_returns * 100 - 30, sml_returns * 100, color='#e74c3c', alpha=0.08, label='Overvalued Region (Alpha < 0)')

    # 2. Quỹ đạo danh mục theo weight
    ax.plot(port_betas, port_actual_capm_rets * 100, color='#7f8c8d', linestyle='--', linewidth=2.0, zorder=3,
            label='Đường tỷ trọng danh mục (w: 0% -> 100%)')
    scatter_port = ax.scatter(port_betas, port_actual_capm_rets * 100, c=weights_capm * 100, cmap='coolwarm', 
                              s=40, alpha=0.85, zorder=4, edgecolor='none')
    cbar = fig.colorbar(scatter_port, ax=ax, pad=0.02)
    cbar.set_label('Tỷ trọng cổ phiếu FPT ($w_{FPT}$ %)', fontsize=10, fontweight='bold')

    # 3. Điểm Benchmark
    ax.scatter(1.0, market_annual_return * 100, color='#8e44ad', marker='D', s=120, zorder=6, 
               label=f'VN30 Benchmark: β=1.0, Ret={market_annual_return*100:.1f}%')
    ax.scatter(0.0, rf_annual * 100, color='#636e72', marker='s', s=100, zorder=6, 
               label=f'Risk-Free Asset ($R_f$): β=0.0, Ret={rf_annual*100:.1f}%')

    # 4. FPT, VNM, MVP, Tangency kèm kiểm định ý nghĩa thống kê của Alpha
    ret_fpt = annual_returns['FPT.VN'] * 100
    exp_capm_fpt = capm_stats['FPT.VN']['expected_capm'] * 100
    alpha_fpt_val = ret_fpt - exp_capm_fpt
    fpt_p = capm_stats['FPT.VN']['alpha_pvalue']
    fpt_sig = "(p<0.05*)" if fpt_p < 0.05 else "(p>0.05 not sig)"
    ax.scatter(beta_fpt, ret_fpt, color=colors['FPT.VN'], s=160, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"100% FPT: β={beta_fpt:.2f}, Ret={ret_fpt:.1f}% (α={alpha_fpt_val:+.1f}%) {fpt_sig}")
    ax.plot([beta_fpt, beta_fpt], [exp_capm_fpt, ret_fpt], color=colors['FPT.VN'], linestyle=':', linewidth=2.2)

    ret_vnm = annual_returns['VNM.VN'] * 100
    exp_capm_vnm = capm_stats['VNM.VN']['expected_capm'] * 100
    alpha_vnm_val = ret_vnm - exp_capm_vnm
    vnm_p = capm_stats['VNM.VN']['alpha_pvalue']
    vnm_sig = "(p<0.05*)" if vnm_p < 0.05 else "(p>0.05 not sig)"
    ax.scatter(beta_vnm, ret_vnm, color=colors['VNM.VN'], s=160, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"100% VNM: β={beta_vnm:.2f}, Ret={ret_vnm:.1f}% (α={alpha_vnm_val:+.1f}%) {vnm_sig}")
    ax.plot([beta_vnm, beta_vnm], [exp_capm_vnm, ret_vnm], color=colors['VNM.VN'], linestyle=':', linewidth=2.2)

    ret_mvp_pct = data['ret_mvp'] * 100
    exp_capm_mvp_pct = data['capm_ret_mvp'] * 100
    ax.scatter(beta_mvp, ret_mvp_pct, color='#00b894', marker='^', s=180, edgecolor='black', linewidth=1.5, zorder=7,
               label=f"MVP Portfolio: β={beta_mvp:.2f}, Ret={ret_mvp_pct:.1f}%")

    ret_tan_pct = data['ret_tan'] * 100
    exp_capm_tan_pct = data['capm_ret_tan'] * 100
    ax.scatter(beta_tan, ret_tan_pct, color='#fdcb6e', marker='*', s=320, edgecolor='black', linewidth=1.5, zorder=8,
               label=f"Tangency Portfolio: β={beta_tan:.2f}, Ret={ret_tan_pct:.1f}%")

    ax.set_title("SECURITY MARKET LINE (SML) & ASSET PRICING VALUATION", fontsize=12.5, fontweight='bold', pad=12)
    ax.set_xlabel(r'Rủi ro hệ thống - Systematic Risk Beta ($\beta$)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Expected / Actual Return E(R) (%)', fontsize=10.5, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8.0)
    ax.set_xlim(left=-0.08, right=1.45)
    ax.set_ylim(bottom=min(-5.0, ret_vnm - 8), top=max(market_annual_return*100, ret_fpt) + 12)
    fig.tight_layout()
    return fig

def create_fig_covariance_correlation(data):
    """5. Heatmaps Ma trận Hiệp phương sai & Ma trận tương quan"""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=100)
    labels = data['stock_tickers']
    cov_vals = data['cov_matrix'].values
    corr_vals = data['corr_matrix'].values

    # Heatmap 1: Covariance
    im1 = axes[0].imshow(cov_vals, cmap='Blues', alpha=0.85)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Covariance Value', fontsize=9.5, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(labels, fontsize=10.5, fontweight='bold')
    axes[0].set_yticklabels(labels, fontsize=10.5, fontweight='bold')
    axes[0].set_title('Annualized Covariance Matrix (Σ)', fontsize=12, fontweight='bold', pad=10)

    for i in range(2):
        for j in range(2):
            val = cov_vals[i, j]
            desc = "Variance (σ²)" if i == j else "Covariance (Cov)"
            text_color = 'white' if val > 0.06 else 'black'
            axes[0].text(j, i, f"{desc}\n{val:.6f}\n({np.sqrt(val)*100:.2f}% Vol)" if i == j else f"{desc}\n{val:.6f}",
                          ha='center', va='center', color=text_color, fontsize=10.5, fontweight='bold')

    # Heatmap 2: Correlation
    im2 = axes[1].imshow(corr_vals, cmap='YlGnBu', vmin=0, vmax=1, alpha=0.85)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Correlation Coefficient (r)', fontsize=9.5, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(labels, fontsize=10.5, fontweight='bold')
    axes[1].set_yticklabels(labels, fontsize=10.5, fontweight='bold')
    axes[1].set_title('Correlation Matrix & Diversification', fontsize=12, fontweight='bold', pad=10)

    for i in range(2):
        for j in range(2):
            val = corr_vals[i, j]
            text_color = 'white' if val > 0.6 else 'black'
            note = "Perfect Self-Corr" if i == j else f"Low Correlation\nr = {val:.4f}"
            axes[1].text(j, i, f"{val:.4f}\n({note})",
                          ha='center', va='center', color=text_color, fontsize=10.5, fontweight='bold')

    fig.tight_layout()
    return fig
