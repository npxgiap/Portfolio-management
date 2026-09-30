"""
========================================================================================
WEB DASHBOARD MODULE: MASTER INTERACTIVE WEB APPLICATION
========================================================================================
File tổng hợp thiết kế dưới dạng Web Dashboard.
Nhúng trực tiếp 5 đồ thị phân tích danh mục (không lưu file ảnh tĩnh) và tích hợp toàn bộ
bảng giá Adjusted Close, Thống kê rủi ro, và chức năng xuất Excel/CSV.
"""

import sys
import os
import io
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pandas as pd
import numpy as np

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import 2 module phân tích và vẽ biểu đồ đã được tách riêng
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

    # Bảng số liệu thống kê Expected Return & Variance
    metrics_df = data['metrics_df']
    metrics_table_html = metrics_df.apply(
        lambda col: col.map(lambda x: f"{x:.4f}" if 'Var' in col.name or 'σ²' in col.name else f"{x:.2f}%")
    ).to_html(classes='custom-table', border=0)

    # Bảng 10 phiên giá gần nhất
    raw_data = data['raw_data'][data['stock_tickers']]
    prices_table_html = raw_data.tail(12).apply(
        lambda col: col.map(lambda x: f"{x:,.2f} VND")
    ).to_html(classes='custom-table', border=0)

    # Thống kê mô tả
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
            font-size: 13px;
            text-align: right;
            margin: 12px 0;
        }}
        .custom-table th {{
            background: #1e293b;
            color: var(--accent-blue);
            padding: 12px 16px;
            font-weight: 700;
            border-bottom: 2px solid var(--border-color);
            text-align: right;
        }}
        .custom-table td {{
            padding: 10px 16px;
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
            <p>Hệ thống phân tích Danh mục đầu tư Định lượng FPT & VNM | 3 Năm (2023 - 2026)</p>
        </div>
        <div>
            <span class="badge">● DỮ LIỆU ĐÃ ĐỒNG BỘ YFINANCE</span>
        </div>
    </div>

    <!-- KPI Metric Cards Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Tangency Portfolio</div>
            <div class="kpi-value color-green">SR = {sharpe_tan:.2f}</div>
            <div class="kpi-desc">FPT {w_fpt_tan:.0f}% | VNM {w_vnm_tan:.0f}% • Ret: {ret_tan:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Min Variance (MVP)</div>
            <div class="kpi-value color-blue">Vol = {vol_mvp:.2f}%</div>
            <div class="kpi-desc">FPT {w_fpt_mvp:.0f}% | VNM {w_vnm_mvp:.0f}% • Ret: {ret_mvp:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Hệ số tương quan (r)</div>
            <div class="kpi-value color-purple">r = {corr_val:.4f}</div>
            <div class="kpi-desc">Đa dạng hóa rủi ro vượt trội</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">FPT.VN (Beta & Alpha)</div>
            <div class="kpi-value color-orange">β={beta_fpt:.2f} | α={alpha_fpt:+.1f}%</div>
            <div class="kpi-desc">Rủi ro hệ thống thấp hơn VN30</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">VNM.VN (Beta & Alpha)</div>
            <div class="kpi-value">β={beta_vnm:.2f} | α={alpha_vnm:+.1f}%</div>
            <div class="kpi-desc">Biến động thấp, tính phòng thủ cao</div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs-nav">
        <button class="tab-btn active" onclick="openTab('tab1', this)">1. Biên hiệu quả Markowitz (CAL)</button>
        <button class="tab-btn" onclick="openTab('tab2', this)">2. Phân phối lợi nhuận & VaR</button>
        <button class="tab-btn" onclick="openTab('tab3', this)">3. Mô hình CAPM & Độ nhạy Beta</button>
        <button class="tab-btn" onclick="openTab('tab4', this)">4. Mô hình SML & Định giá</button>
        <button class="tab-btn" onclick="openTab('tab5', this)">5. Ma trận Hiệp phương sai & Tương quan</button>
        <button class="tab-btn" onclick="openTab('tab6', this)">6. Bảng giá Adjusted Close & Dữ liệu</button>
    </div>

    <!-- Tab 1: Markowitz Efficient Frontier & CAL -->
    <div id="tab1" class="tab-pane active">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Mô hình Đường biên hiệu quả (Markowitz) & Đường phân bổ vốn (CAL)</div>
            </div>
            <div class="chart-container">
                {svg_frontier}
            </div>
            <div class="insight-box">
                <h4>Phân tích dành cho Analyst:</h4>
                • <strong>Tangency Portfolio:</strong> Tỷ trọng tối ưu FPT = {w_fpt_tan:.0f}%, VNM = {w_vnm_tan:.0f}% đạt hệ số Sharpe cao nhất ({sharpe_tan:.2f}), tiếp xúc với đường phân bổ vốn CAL.<br>
                • <strong>Minimum Variance Portfolio (MVP):</strong> Tỷ trọng FPT = {w_fpt_mvp:.0f}%, VNM = {w_vnm_mvp:.0f}% mang lại rủi ro thấp nhất ({vol_mvp:.2f}%), thấp hơn cả việc chỉ đầu tư vào VNM ({data['stock_vols']['VNM.VN']*100:.2f}%).<br>
                • <strong>Nhánh dưới (Inefficient):</strong> Các danh mục có tỷ trọng VNM quá lớn bị thống trị bởi các danh mục ở nhánh trên cùng mức rủi ro nhưng có lợi nhuận cao hơn.
            </div>
        </div>
    </div>

    <!-- Tab 2: Return Distribution & VaR/CVaR -->
    <div id="tab2" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Phân tích Phân phối lợi nhuận hàng ngày & Quản trị rủi ro đuôi (VaR 95%, CVaR 95%)</div>
            </div>
            <div class="chart-container">
                {svg_dist}
            </div>
            <div class="insight-box">
                <h4>Phân tích rủi ro & Định lượng:</h4>
                • <strong>Value at Risk (VaR 95% 1 ngày):</strong> Cho biết trong 95% các phiên giao dịch bình thường, mức lỗ tối đa không vượt quá giá trị VaR.<br>
                • <strong>Conditional VaR (CVaR 95% / Expected Shortfall):</strong> Đo lường tổn thất kỳ vọng trung bình khi xảy ra tình huống xấu nhất (rơi vào 5% đuôi rủi ro bên trái).<br>
                • <strong>Kurtosis & Skewness:</strong> Lợi nhuận của cả hai cổ phiếu đều có hiện tượng đuôi dày (Fat-tail) và lệch so với phân phối chuẩn lý thuyết Gauss.
            </div>
        </div>
    </div>

    <!-- Tab 3: CAPM & Portfolio Beta Sensitivity -->
    <div id="tab3" class="tab-pane">
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Mô hình CAPM: Hồi quy SCL & Độ nhạy Rủi ro hệ thống (Beta) theo Tỷ trọng (0% -> 100%)</div>
            </div>
            <div class="chart-container">
                {svg_capm}
            </div>
            <div class="insight-box">
                <h4>Ý nghĩa tài chính của Mô hình CAPM:</h4>
                • <strong>Security Characteristic Line (SCL):</strong> Hồi quy lợi nhuận vượt trội của cổ phiếu theo VN30 ETF để bóc tách rủi ro hệ thống (Beta) và hiệu suất vượt trội (Alpha).<br>
                • <strong>Tuyến tính hóa Beta danh mục:</strong> Khi trọng số FPT thay đổi từ 0% đến 100%, Beta danh mục biến thiên tuyến tính từ {beta_vnm:.2f} (VNM) lên {beta_fpt:.2f} (FPT). Cả hai cổ phiếu đều có Beta < 1.0 (ít biến động hơn thị trường chung VN30).
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
                <h4>Định giá tài sản theo Jensen's Alpha:</h4>
                • <strong>Đường SML chuẩn:</strong> Thiết lập mối quan hệ giữa rủi ro hệ thống Beta và lợi nhuận đòi hỏi theo lý thuyết $E(R) = R_f + \\beta [E(R_m) - R_f]$.<br>
                • <strong>Định vị tài sản:</strong> Khoảng cách thẳng đứng từ điểm thực tế đến đường SML chính là <strong>Jensen's Alpha</strong>.<br>
                • Điểm nằm <strong>phía trên SML</strong> đại diện cho tài sản sinh lời vượt kỳ vọng bù đắp rủi ro (Undervalued - định giá rẻ). Điểm nằm <strong>phía dưới SML</strong> đại diện cho tài sản sinh lời thấp hơn bù đắp rủi ro (Overvalued).
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
                • Hệ số tương quan giữa FPT và VNM đạt <strong>r = {corr_val:.4f}</strong> (ở mức thấp đến trung bình).<br>
                • Nhờ hệ số tương quan $r < 1.0$, việc kết hợp FPT và VNM đã triệt tiêu một phần rủi ro phi hệ thống (Idiosyncratic Risk), giúp danh mục MVP ({vol_mvp:.2f}%) an toàn hơn việc nắm giữ 100% của từng cổ phiếu riêng lẻ.
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
        Portfolio Quantitative Analysis Dashboard | FPT & VNM | Hỗ trợ tương tác toàn diện trên trình duyệt
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
    print("="*70)
    
    # 1. Tính toán dữ liệu định lượng
    print("[-] Đang tính toán dữ liệu tài chính (Markowitz, CAPM, SML, VaR)...")
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
    print(f"[+] Đã lưu bản sao Web tĩnh tại: {html_file_path}")

    # 5. Khởi động Web Server
    server_address = ('127.0.0.1', port)
    try:
        httpd = HTTPServer(server_address, DashboardHTTPHandler)
    except OSError:
        # Nếu cổng 8501 đang bận, đổi sang cổng 8080
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
