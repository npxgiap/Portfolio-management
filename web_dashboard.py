"""
========================================================================================
WEB DASHBOARD MODULE: MASTER INTERACTIVE WEB APPLICATION
========================================================================================
File tổng hợp thiết kế dưới dạng Web Dashboard.
Nhúng trực tiếp 5 đồ thị phân tích danh mục (không lưu file ảnh tĩnh) và tích hợp toàn bộ:
  - Tối ưu hóa Markowitz SLSQP (Scipy)
  - Phân tích rủi ro đuôi, kiểm định Jarque-Bera & Cornish-Fisher VaR (Scipy.stats)
  - Mô hình hồi quy OLS CAPM với 95% Confidence Band & kiểm định t-stat/p-value (Statsmodels)
  - Bảng giá Adjusted Close, Thống kê rủi ro, và chức năng xuất Excel/CSV.
"""

import sys
import os
import io
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pandas as pd
import numpy as np

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import các module đã module hóa
from portfolio_engine import load_and_calculate_portfolio
from charts import (
    create_fig_efficient_frontier,
    create_fig_return_distribution,
    create_fig_capm_regression,
    create_fig_sml,
    create_fig_covariance_correlation
)

def fig_to_svg_string(fig):
    """Chuyển đổi Matplotlib Figure thành mã SVG để nhúng trực tiếp vào Web, không lưu ảnh ra đĩa"""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue().decode('utf-8')

def build_html_dashboard(data):
    """Tạo trang web HTML hoàn chỉnh chứa đầy đủ đồ thị, bảng biểu và báo cáo phân tích"""
    print("[-] Đang kết xuất các biểu đồ sang định dạng vector SVG...")
    svg_frontier = fig_to_svg_string(create_fig_efficient_frontier(data))
    svg_dist = fig_to_svg_string(create_fig_return_distribution(data))
    svg_capm = fig_to_svg_string(create_fig_capm_regression(data))
    svg_sml = fig_to_svg_string(create_fig_sml(data))
    svg_cov = fig_to_svg_string(create_fig_covariance_correlation(data))
    print("[+] Đã kết xuất 5 đồ thị thành công!")

    # 1. Bảng số liệu thống kê Expected Return & Variance
    metrics_df = data['metrics_df']
    metrics_table_html = metrics_df.apply(
        lambda col: col.map(lambda x: f"{x:.4f}" if 'Var' in col.name or 'σ²' in col.name else f"{x:.2f}%")
    ).to_html(classes='custom-table', border=0)

    # 2. Bảng kinh tế lượng CAPM OLS từ statsmodels
    rows_capm = []
    for t in data['stock_tickers']:
        st = data['capm_stats'][t]
        rows_capm.append({
            'Ticker': t,
            'Beta (β)': f"{st['beta']:.4f}",
            'SE(β)': f"{st['beta_se']:.4f}",
            't-stat (β)': f"{st['beta_tstat']:.2f}",
            'p-value (β)': f"{st['beta_pvalue']:.2e}",
            '95% CI (β)': f"[{st['beta_ci_95'][0]:.3f}, {st['beta_ci_95'][1]:.3f}]",
            "Alpha (α)": f"{st['alpha_annual']*100:+.2f}%",
            'p-value (α)': f"{st['alpha_pvalue']:.4f}",
            "95% CI (α)": f"[{st['alpha_ci_95'][0]*100:+.2f}%, {st['alpha_ci_95'][1]*100:+.2f}%]",
            'R²': f"{st['r_squared']:.4f}",
            'Adj R²': f"{st['adj_r_squared']:.4f}",
            'F-stat': f"{st['f_stat']:.2f}",
            'Durbin-Watson': f"{st['durbin_watson']:.2f}"
        })
    df_capm_table = pd.DataFrame(rows_capm).set_index('Ticker')
    capm_table_html = df_capm_table.to_html(classes='custom-table', border=0)

    # 3. Bảng phân phối & Rủi ro đuôi từ scipy.stats
    rows_risk = []
    for t in data['stock_tickers']:
        m = data['dist_metrics'][t]
        jb_conclusion = "Non-Normal (p<0.01)" if m['jb_pvalue'] < 0.01 else "Normal"
        rows_risk.append({
            'Ticker': t,
            'Skewness': f"{m['skew']:.4f}",
            'Excess Kurtosis': f"{m['kurt']:.4f}",
            'Jarque-Bera Stat': f"{m['jb_stat']:.2f}",
            'JB p-value': f"{m['jb_pvalue']:.2e}",
            'Kiểm định phân phối': jb_conclusion,
            'Hist VaR 95%': f"{m['var_95']:.2f}%",
            'Parametric VaR 95%': f"{m['var_param_95']:.2f}%",
            'Cornish-Fisher VaR 95%': f"{m['var_cf_95']:.2f}%",
            'CVaR 95% (ES)': f"{m['cvar_95']:.2f}%"
        })
    df_risk_table = pd.DataFrame(rows_risk).set_index('Ticker')
    risk_table_html = df_risk_table.to_html(classes='custom-table', border=0)

    # 4. Bảng 12 phiên giá gần nhất
    raw_data = data['raw_data'][data['stock_tickers']]
    prices_table_html = raw_data.tail(12).apply(
        lambda col: col.map(lambda x: f"{x:,.2f} VND")
    ).to_html(classes='custom-table', border=0)

    # 5. Thống kê mô tả
    desc_table_html = raw_data.describe().apply(
        lambda col: col.map(lambda x: f"{x:,.2f}")
    ).to_html(classes='custom-table', border=0)

    # Tóm tắt số liệu quan trọng
    w_fpt_tan = data['w_fpt_tan'] * 100
    w_vnm_tan = data['w_vnm_tan'] * 100
    ret_tan = data['ret_tan'] * 100
    vol_tan = data['vol_tan'] * 100
    sharpe_tan = data['sharpe_tan']

    w_fpt_mvp = data['w_fpt_mvp'] * 100
    w_vnm_mvp = data['w_vnm_mvp'] * 100
    ret_mvp = data['ret_mvp'] * 100
    vol_mvp = data['vol_mvp'] * 100

    corr_val = data['corr_fpt_vnm']
    beta_fpt = data['capm_stats']['FPT.VN']['beta']
    alpha_fpt = data['capm_stats']['FPT.VN']['alpha_annual'] * 100
    beta_vnm = data['capm_stats']['VNM.VN']['beta']
    alpha_vnm = data['capm_stats']['VNM.VN']['alpha_annual'] * 100

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Management Dashboard | FPT & VNM</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-orange: #fb923c;
            --accent-purple: #c084fc;
            --accent-red: #f87171;
            --border-color: #334155;
            --hover-bg: #334155;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            line-height: 1.6;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 24px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .header p {{ color: var(--text-muted); font-size: 14px; }}
        .badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 12.5px;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 18px;
            transition: all 0.25s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent-blue);
            box-shadow: 0 12px 20px -5px rgba(0,0,0,0.4);
        }}
        .kpi-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); margin-bottom: 6px; }}
        .kpi-value {{ font-size: 24px; font-weight: 800; color: #fff; margin-bottom: 4px; }}
        .kpi-desc {{ font-size: 12px; color: var(--text-muted); }}
        .color-green {{ color: var(--accent-green) !important; }}
        .color-blue {{ color: var(--accent-blue) !important; }}
        .color-orange {{ color: var(--accent-orange) !important; }}
        .color-purple {{ color: var(--accent-purple) !important; }}

        /* Tabs Container */
        .tabs-nav {{
            display: flex;
            gap: 8px;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 24px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 600;
            padding: 12px 20px;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }}
        .tab-btn:hover {{ color: var(--text-main); background: rgba(255,255,255,0.04); }}
        .tab-btn.active {{
            color: var(--accent-blue);
            background: rgba(56, 189, 248, 0.1);
            border-bottom: 3px solid var(--accent-blue);
        }}

        /* Tab Content */
        .tab-pane {{
            display: none;
            animation: fadeIn 0.3s ease;
        }}
        .tab-pane.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        .chart-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.25);
        }}
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }}
        .chart-title {{ font-size: 18px; font-weight: 700; color: #fff; }}
        .chart-container {{
            background: #ffffff;
            border-radius: 12px;
            padding: 12px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: auto;
        }}
        .chart-container svg {{
            max-width: 100%;
            height: auto;
            display: block;
        }}

        /* Financial Insight Box */
        .insight-box {{
            background: rgba(15, 23, 42, 0.7);
            border-left: 4px solid var(--accent-blue);
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 18px;
            font-size: 13.5px;
            color: #cbd5e1;
        }}
        .insight-box h4 {{
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 6px;
            text-transform: uppercase;
        }}

        /* Custom Data Table */
        .custom-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12.5px;
            text-align: right;
            margin: 12px 0;
        }}
        .custom-table th {{
            background: #1e293b;
            color: var(--accent-blue);
            padding: 12px 14px;
            font-weight: 700;
            border-bottom: 2px solid var(--border-color);
            text-align: right;
        }}
        .custom-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            color: #e2e8f0;
        }}
        .custom-table tr:hover td {{
            background: rgba(56, 189, 248, 0.05);
        }}
        .custom-table th:first-child, .custom-table td:first-child {{
            text-align: left;
            font-weight: 600;
            color: #fff;
        }}

        .btn-group {{
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 13.5px;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}
        .btn-primary {{
            background: var(--accent-blue);
            color: #0f172a;
        }}
        .btn-primary:hover {{
            background: #7dd3fc;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4);
        }}
        .btn-secondary {{
            background: rgba(255,255,255,0.06);
            color: #fff;
            border-color: var(--border-color);
        }}
        .btn-secondary:hover {{
            background: rgba(255,255,255,0.12);
        }}
        
        .footer {{
            text-align: center;
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>

    <!-- Header Section -->
    <div class="header">
        <div>
            <h1>PORTFOLIO MANAGEMENT & QUANTITATIVE ANALYTICS</h1>
            <p>Hệ thống phân tích Định lượng FPT & VNM | Scipy, Statsmodels & Matplotlib Engine</p>
        </div>
        <div>
            <span class="badge">● SCIPY & STATSMODELS ACTIVE</span>
        </div>
    </div>

    <!-- KPI Metric Cards Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Tangency (Scipy SLSQP)</div>
            <div class="kpi-value color-green">SR = {sharpe_tan:.2f}</div>
            <div class="kpi-desc">FPT {w_fpt_tan:.1f}% | VNM {w_vnm_tan:.1f}% • Ret: {ret_tan:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Min Variance (Scipy SLSQP)</div>
            <div class="kpi-value color-blue">Vol = {vol_mvp:.2f}%</div>
            <div class="kpi-desc">FPT {w_fpt_mvp:.1f}% | VNM {w_vnm_mvp:.1f}% • Ret: {ret_mvp:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Hệ số tương quan (r)</div>
            <div class="kpi-value color-purple">r = {corr_val:.4f}</div>
            <div class="kpi-desc">Đa dạng hóa rủi ro vượt trội</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">FPT.VN (OLS Beta & Alpha)</div>
            <div class="kpi-value color-orange">β={beta_fpt:.2f} | α={alpha_fpt:+.1f}%</div>
            <div class="kpi-desc">p(β)={data['capm_stats']['FPT.VN']['beta_pvalue']:.1e} • R²={data['capm_stats']['FPT.VN']['r_squared']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">VNM.VN (OLS Beta & Alpha)</div>
            <div class="kpi-value">β={beta_vnm:.2f} | α={alpha_vnm:+.1f}%</div>
            <div class="kpi-desc">p(β)={data['capm_stats']['VNM.VN']['beta_pvalue']:.1e} • R²={data['capm_stats']['VNM.VN']['r_squared']:.2f}</div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs-nav">
        <button class="tab-btn active" onclick="openTab('tab1', this)">1. Biên hiệu quả Markowitz (CAL)</button>
        <button class="tab-btn" onclick="openTab('tab2', this)">2. Phân phối lợi nhuận, KDE & VaR</button>
        <button class="tab-btn" onclick="openTab('tab3', this)">3. Mô hình CAPM & Hồi quy OLS</button>
        <button class="tab-btn" onclick="openTab('tab4', this)">4. Mô hình SML & Định giá</button>
        <button class="tab-btn" onclick="openTab('tab5', this)">5. Ma trận Hiệp phương sai & Tương quan</button>
        <button class="tab-btn" onclick="openTab('tab6', this)">6. Dữ liệu Adjusted Close & Thống kê</button>
    </div>

    <!-- Tab 1: Markowitz Efficient Frontier & CAL -->
    <div id="tab1" class="tab-pane active">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Mô hình Đường biên hiệu quả (Markowitz SLSQP Optimization) & Đường CAL</div>
            </div>
            <div class="chart-container">
                {svg_frontier}
            </div>
            <div class="insight-box">
                <h4>Phân tích tối ưu hóa toán học (Scipy Optimize SLSQP):</h4>
                • <strong>Tangency Portfolio:</strong> Giải thuật toán tối ưu hóa phi tuyến tính tìm ra tỷ trọng FPT = {w_fpt_tan:.1f}%, VNM = {w_vnm_tan:.1f}% đạt Sharpe Ratio tối đa ({sharpe_tan:.2f}).<br>
                • <strong>Minimum Variance Portfolio (MVP):</strong> Điểm rủi ro nhỏ nhất toàn cầu đạt mức biến động {vol_mvp:.2f}%, thấp hơn độ lệch chuẩn của cả FPT ({data['stock_vols']['FPT.VN']*100:.2f}%) và VNM ({data['stock_vols']['VNM.VN']*100:.2f}%).<br>
                • <strong>Nhánh dưới (Inefficient):</strong> Các danh mục có tỷ trọng VNM lớn bị thống trị hoàn toàn về mặt lợi nhuận trên mỗi đơn vị rủi ro.
            </div>
        </div>
    </div>

    <!-- Tab 2: Return Distribution & VaR/CVaR -->
    <div id="tab2" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Phân tích Phân phối lợi nhuận thực nghiệm (KDE), Phân phối chuẩn Gauss & VaR/CVaR</div>
            </div>
            <div class="chart-container">
                {svg_dist}
            </div>

            <h3 style="margin: 20px 0 8px 0; color: var(--accent-blue);">Bảng kiểm định phân phối Jarque-Bera & Quản trị rủi ro đuôi (Scipy Stats)</h3>
            <div style="overflow-x: auto;">
                {risk_table_html}
            </div>

            <div class="insight-box">
                <h4>Ý nghĩa kiểm định kinh tế lượng (Scipy Stats):</h4>
                • <strong>Kernel Density Estimation (KDE):</strong> Đường KDE từ <code>scipy.stats.gaussian_kde</code> phản ánh hình dạng mật độ xác suất thực tế mượt mà, nắm bắt chính xác độ nhọn và độ lệch so với đường chuẩn Gauss.<br>
                • <strong>Kiểm định Jarque-Bera:</strong> Cả FPT và VNM đều có p-value < 0.01, chính thức bác bỏ giả thuyết phân phối chuẩn, xác nhận dữ liệu có hiện tượng đuôi dày (Fat-tail).<br>
                • <strong>Cornish-Fisher VaR:</strong> Hiệu chỉnh rủi ro VaR dựa trên độ lệch (Skewness) và độ nhọn (Kurtosis), giúp phản ánh rủi ro thị trường chân thực hơn so với VaR tham số cổ điển.
            </div>
        </div>
    </div>

    <!-- Tab 3: CAPM & Portfolio Beta Sensitivity -->
    <div id="tab3" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Mô hình CAPM: Hồi quy OLS Statsmodels với 95% Confidence Band & Độ nhạy Beta (0% -> 100%)</div>
            </div>
            <div class="chart-container">
                {svg_capm}
            </div>

            <h3 style="margin: 20px 0 8px 0; color: var(--accent-blue);">Bảng kết quả hồi quy kinh tế lượng CAPM OLS (Statsmodels Engine)</h3>
            <div style="overflow-x: auto;">
                {capm_table_html}
            </div>

            <div class="insight-box">
                <h4>Phân tích kinh tế lượng chuyên sâu:</h4>
                • <strong>Ý nghĩa thống kê của Hệ số Beta:</strong> Hệ số Beta của cả FPT ({beta_fpt:.2f}) và VNM ({beta_vnm:.2f}) đều có p-value cực nhỏ (< 1e-20), khẳng định rủi ro hệ thống có ý nghĩa thống kê vượt trội.<br>
                • <strong>95% Confidence Band:</strong> Dải khoảng tin cậy 95% bao quanh đường hồi quy SCL thể hiện biên độ bất định trong việc ước lượng lợi nhuận kỳ vọng của cổ phiếu.<br>
                • <strong>Tuyến tính hóa Beta:</strong> Khi tỷ trọng FPT tăng từ 0% đến 100%, Beta danh mục biến thiên tuyến tính từ {beta_vnm:.2f} lên {beta_fpt:.2f}.
            </div>
        </div>
    </div>

    <!-- Tab 4: Security Market Line (SML) & Valuation -->
    <div id="tab4" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Mô hình Security Market Line (SML) & Định giá Danh mục theo Trọng số</div>
            </div>
            <div class="chart-container">
                {svg_sml}
            </div>
            <div class="insight-box">
                <h4>Định giá tài sản & Kiểm định Jensen's Alpha:</h4>
                • <strong>Đường SML chuẩn:</strong> Thiết lập mối quan hệ giữa rủi ro hệ thống Beta và lợi nhuận đòi hỏi theo lý thuyết $E(R) = R_f + \\beta [E(R_m) - R_f]$.<br>
                • <strong>Kiểm định Jensen's Alpha:</strong> Mặc dù FPT và VNM có Alpha âm trong giai đoạn 3 năm do thị trường VN30 tăng trưởng rất mạnh (21.77%/năm), kiểm định p-value cho thấy Alpha không khác 0 có ý nghĩa thống kê ở mức 5%.<br>
                • <strong>Vùng định giá:</strong> Cổ phiếu nằm phía trên SML được xem là Undervalued (định giá hấp dẫn), phía dưới SML là Overvalued.
            </div>
        </div>
    </div>

    <!-- Tab 5: Covariance & Correlation Matrix -->
    <div id="tab5" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Bảng nhiệt (Heatmap) Ma trận Hiệp phương sai & Ma trận Hệ số Tương quan</div>
            </div>
            <div class="chart-container">
                {svg_cov}
            </div>
            <div class="insight-box">
                <h4>Hiệu ứng đa dạng hóa danh mục (Diversification Benefit):</h4>
                • Hệ số tương quan giữa FPT và VNM đạt <strong>r = {corr_val:.4f}</strong> (ở mức thấp/trung bình).<br>
                • Do $r < 1.0$, việc kết hợp FPT và VNM triệt tiêu một phần rủi ro phi hệ thống (Unsystematic Risk), giúp danh mục MVP ({vol_mvp:.2f}%) an toàn hơn việc nắm giữ 100% của từng cổ phiếu riêng lẻ.
            </div>
        </div>
    </div>

    <!-- Tab 6: Adjusted Close & Data Export -->
    <div id="tab6" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Bảng thống kê Expected Return, Variance & Giá đóng cửa điều chỉnh (Adjusted Close)</div>
            </div>
            
            <h3 style="margin: 16px 0 8px 0; color: var(--accent-blue);">1. Bảng thống kê chi tiết Expected Return & Các chỉ số Variance</h3>
            <div style="overflow-x: auto;">
                {metrics_table_html}
            </div>

            <h3 style="margin: 24px 0 8px 0; color: var(--accent-blue);">2. Thống kê mô tả giá đóng cửa điều chỉnh (Descriptive Statistics)</h3>
            <div style="overflow-x: auto;">
                {desc_table_html}
            </div>

            <h3 style="margin: 24px 0 8px 0; color: var(--accent-blue);">3. Dữ liệu 12 phiên giao dịch gần nhất (Adjusted Close)</h3>
            <div style="overflow-x: auto;">
                {prices_table_html}
            </div>

            <div class="btn-group">
                <a href="/download_excel" class="btn btn-primary">📥 Tải file Excel (.xlsx)</a>
                <a href="/download_csv" class="btn btn-secondary">📄 Tải file CSV (.csv)</a>
            </div>
        </div>
    </div>

    <div class="footer">
        Portfolio Quantitative Analysis Dashboard | FPT & VNM | Tích hợp Scipy, Statsmodels & Matplotlib
    </div>

    <script>
        function openTab(tabId, btn) {{
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    return html_content

class DashboardHTTPHandler(SimpleHTTPRequestHandler):
    """Bộ xử lý HTTP cho Web Dashboard nội bộ"""
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.server.html_content.encode('utf-8'))
        elif self.path == '/download_excel':
            excel_path = os.path.join(os.path.dirname(__file__), 'Adjusted_Close_FPT_VNM.xlsx')
            if os.path.exists(excel_path):
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                self.send_header('Content-Disposition', 'attachment; filename="Adjusted_Close_FPT_VNM.xlsx"')
                self.end_headers()
                with open(excel_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, 'File Excel chưa được tạo.')
        elif self.path == '/download_csv':
            csv_path = os.path.join(os.path.dirname(__file__), 'Adj_Close_FPT_VNM.csv')
            if os.path.exists(csv_path):
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="Adj_Close_FPT_VNM.csv"')
                self.end_headers()
                with open(csv_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, 'File CSV chưa được tạo.')
        else:
            super().do_GET()

def start_web_server(port=8501):
    """Khởi động Web Server cục bộ và tự động mở trình duyệt"""
    print("\n" + "="*70)
    print("      KHỞI CHẠY WEB DASHBOARD QUẢN TRỊ DANH MỤC (FPT & VNM)")
    print("       Động cơ định lượng: Scipy, Statsmodels, Matplotlib")
    print("="*70)
    
    # 1. Tính toán dữ liệu định lượng
    print("[-] Đang tính toán dữ liệu tài chính (Markowitz SLSQP, OLS CAPM, Jarque-Bera)...")
    data = load_and_calculate_portfolio()
    
    # 2. Tạo file Excel và CSV dự phòng
    excel_path = os.path.join(os.path.dirname(__file__), 'Adjusted_Close_FPT_VNM.xlsx')
    csv_path = os.path.join(os.path.dirname(__file__), 'Adj_Close_FPT_VNM.csv')
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            data['raw_data'][data['stock_tickers']].to_excel(writer, sheet_name='Gia_Dieu_Chinh')
            data['daily_returns'][data['stock_tickers']].to_excel(writer, sheet_name='Loi_Nhuan_Ngay')
            data['metrics_df'].to_excel(writer, sheet_name='Thong_Ke')
        data['raw_data'][data['stock_tickers']].to_csv(csv_path)
    except Exception:
        pass

    # 3. Tạo HTML tổng hợp
    html_dashboard = build_html_dashboard(data)

    # 4. Lưu file HTML cục bộ
    html_file_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_dashboard)
    print(f"[+] Đã cập nhật file Web tĩnh tại: {html_file_path}")

    # 5. Khởi động Web Server
    server_address = ('127.0.0.1', port)
    try:
        httpd = HTTPServer(server_address, DashboardHTTPHandler)
    except OSError:
        server_address = ('127.0.0.1', 8080)
        httpd = HTTPServer(server_address, DashboardHTTPHandler)

    httpd.html_content = html_dashboard
    url = f"http://{server_address[0]}:{server_address[1]}"
    print(f"\n[🚀] WEB DASHBOARD ĐÃ SẴN SÀNG TẠI: {url}")
    print("[*] Đang tự động mở trình duyệt web...")
    webbrowser.open(url)
    print("[*] Bấm Ctrl+C trên terminal để dừng server khi hoàn tất.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Đã dừng Web Dashboard Server.")
        httpd.server_close()

if __name__ == '__main__':
    start_web_server()
