"""
========================================================================================
PORTFOLIO ENGINE: QUANTITATIVE & ECONOMETRIC CALCULATION ENGINE
========================================================================================
Module xử lý dữ liệu tài chính định lượng, tích hợp chuyên sâu:
  - Scipy (scipy.optimize, scipy.stats): Tối ưu hóa Markowitz SLSQP, kiểm định phân phối
    Jarque-Bera, Kernel Density Estimation (KDE), Cornish-Fisher Modified VaR & CVaR.
  - Statsmodels (statsmodels.api): Hồi quy OLS mô hình CAPM, kiểm định ý nghĩa thống kê
    (t-stat, p-value, 95% Confidence Interval, F-statistic, R-squared).
  - Matplotlib (matplotlib.pyplot): Cấu hình hiển thị và chuẩn hóa đồ họa định lượng.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_and_calculate_portfolio(start_date='2023-09-27', end_date='2026-09-28', rf_annual=0.045, trading_days=252):
    stock_tickers = ['FPT.VN', 'VNM.VN']
    market_ticker = 'FUEKIV30.VN'
    all_tickers = stock_tickers + [market_ticker]

    # 1. Tải dữ liệu từ Yahoo Finance (auto_adjust=True thì Close chính là Adjusted Close)
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)['Close'][all_tickers].dropna()
    
    # 2. Tỷ suất sinh lời hàng ngày
    daily_returns = raw_data.pct_change().dropna()
    stock_returns = daily_returns[stock_tickers]
    market_returns = daily_returns[market_ticker]

    # 3. Ma trận hiệp phương sai & tương quan
    daily_cov_matrix = stock_returns.cov()
    cov_matrix = daily_cov_matrix * trading_days  # Năm hóa
    corr_matrix = stock_returns.corr()
    
    annual_returns = stock_returns.mean() * trading_days
    stock_vols = stock_returns.std() * np.sqrt(trading_days)
    market_annual_return = market_returns.mean() * trading_days
    market_vol = market_returns.std() * np.sqrt(trading_days)

    corr_fpt_vnm = corr_matrix.loc['FPT.VN', 'VNM.VN']
    cov_fpt_vnm_annual = cov_matrix.loc['FPT.VN', 'VNM.VN']
    cov_fpt_vnm_daily = daily_cov_matrix.loc['FPT.VN', 'VNM.VN']

    # ==========================================================================
    # 4. TỐI ƯU HÓA MARKOWITZ CHÍNH XÁC BẰNG SCIPY (scipy.optimize - SLSQP)
    # ==========================================================================
    num_assets = len(stock_tickers)
    init_weights = np.array([0.5, 0.5])
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    # Hàm mục tiêu phương sai
    def get_port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))

    # A. Tối ưu Minimum Variance Portfolio (MVP)
    opt_mvp = sco.minimize(get_port_vol, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    w_mvp_exact = opt_mvp.x
    w_fpt_mvp = w_mvp_exact[0]
    w_vnm_mvp = w_mvp_exact[1]
    ret_mvp = np.dot(w_mvp_exact, annual_returns.values)
    vol_mvp = get_port_vol(w_mvp_exact)
    sharpe_mvp = (ret_mvp - rf_annual) / vol_mvp

    # B. Tối ưu Maximum Sharpe Ratio Portfolio (Tangency Portfolio)
    def get_neg_sharpe(w):
        r = np.dot(w, annual_returns.values)
        v = get_port_vol(w)
        return -(r - rf_annual) / v

    opt_tan = sco.minimize(get_neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    w_tan_exact = opt_tan.x
    w_fpt_tan = w_tan_exact[0]
    w_vnm_tan = w_tan_exact[1]
    ret_tan = np.dot(w_tan_exact, annual_returns.values)
    vol_tan = get_port_vol(w_tan_exact)
    sharpe_tan = (ret_tan - rf_annual) / vol_tan

    # C. Mô phỏng dải đường cong Efficient Frontier (201 kịch bản trọng số)
    weights_fpt = np.linspace(0.0, 1.0, 201)
    port_returns = []
    port_volatilities = []
    port_sharpes = []

    for w_f in weights_fpt:
        w = np.array([w_f, 1.0 - w_f])
        ret = np.dot(w, annual_returns.values)
        vol = get_port_vol(w)
        sr = (ret - rf_annual) / vol
        port_returns.append(ret)
        port_volatilities.append(vol)
        port_sharpes.append(sr)

    port_returns = np.array(port_returns)
    port_volatilities = np.array(port_volatilities)
    port_sharpes = np.array(port_sharpes)
    min_vol_idx = np.argmin(port_volatilities)

    # ==========================================================================
    # 5. HỒI QUY KINH TẾ LƯỢNG CAPM CHUẨN XÁC BẰNG STATSMODELS (sm.OLS)
    # ==========================================================================
    rf_daily = rf_annual / trading_days
    excess_market = market_returns - rf_daily
    X_market = sm.add_constant(excess_market)  # Thêm hệ số chặn Alpha

    capm_stats = {}
    for ticker in stock_tickers:
        excess_stock = stock_returns[ticker] - rf_daily
        ols_model = sm.OLS(excess_stock, X_market).fit()
        
        alpha_daily = ols_model.params.iloc[0]
        beta = ols_model.params.iloc[1]
        alpha_annual = alpha_daily * trading_days
        
        # Kiểm định t-stat & p-value
        alpha_pvalue = ols_model.pvalues.iloc[0]
        beta_pvalue = ols_model.pvalues.iloc[1]
        alpha_tstat = ols_model.tvalues.iloc[0]
        beta_tstat = ols_model.tvalues.iloc[1]
        
        # Sai số chuẩn (Standard Errors)
        beta_se = ols_model.bse.iloc[1]
        alpha_se_annual = ols_model.bse.iloc[0] * trading_days
        
        # Khoảng tin cậy 95% (Confidence Intervals 95%)
        conf_int = ols_model.conf_int(alpha=0.05)
        alpha_ci_95 = (conf_int.iloc[0, 0] * trading_days, conf_int.iloc[0, 1] * trading_days)
        beta_ci_95 = (conf_int.iloc[1, 0], conf_int.iloc[1, 1])
        
        # Các chỉ số kiểm định mô hình
        r_squared = ols_model.rsquared
        adj_r_squared = ols_model.rsquared_adj
        f_stat = ols_model.fvalue
        f_pvalue = ols_model.f_pvalue
        durbin_watson = sm.stats.stattools.durbin_watson(ols_model.resid)
        
        # Kỳ vọng lợi nhuận CAPM: E(R) = Rf + Beta * [E(Rm) - Rf]
        expected_capm = rf_annual + beta * (market_annual_return - rf_annual)
        
        # Dải dự báo khoảng tin cậy 95% cho đường hồi quy SCL
        x_grid = np.linspace(excess_market.min(), excess_market.max(), 100)
        X_grid = sm.add_constant(x_grid)
        pred = ols_model.get_prediction(X_grid)
        pred_summary = pred.summary_frame(alpha=0.05)

        capm_stats[ticker] = {
            'beta': beta,
            'beta_se': beta_se,
            'beta_tstat': beta_tstat,
            'beta_pvalue': beta_pvalue,
            'beta_ci_95': beta_ci_95,
            'alpha_annual': alpha_annual,
            'alpha_se_annual': alpha_se_annual,
            'alpha_tstat': alpha_tstat,
            'alpha_pvalue': alpha_pvalue,
            'alpha_ci_95': alpha_ci_95,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue,
            'durbin_watson': durbin_watson,
            'expected_capm': expected_capm,
            'ols_model': ols_model,
            'x_grid': x_grid,
            'pred_summary': pred_summary
        }

    # CAPM theo dải tỷ trọng w từ 0 đến 1 (101 kịch bản)
    weights_capm = np.linspace(0.0, 1.0, 101)
    beta_fpt = capm_stats['FPT.VN']['beta']
    beta_vnm = capm_stats['VNM.VN']['beta']
    alpha_fpt = capm_stats['FPT.VN']['alpha_annual']
    alpha_vnm = capm_stats['VNM.VN']['alpha_annual']

    port_betas = weights_capm * beta_fpt + (1.0 - weights_capm) * beta_vnm
    port_alphas = weights_capm * alpha_fpt + (1.0 - weights_capm) * alpha_vnm
    port_actual_capm_rets = weights_capm * annual_returns['FPT.VN'] + (1.0 - weights_capm) * annual_returns['VNM.VN']
    port_capm_rets = rf_annual + port_betas * (market_annual_return - rf_annual)

    beta_mvp = w_fpt_mvp * beta_fpt + w_vnm_mvp * beta_vnm
    alpha_mvp = w_fpt_mvp * alpha_fpt + w_vnm_mvp * alpha_vnm
    capm_ret_mvp = rf_annual + beta_mvp * (market_annual_return - rf_annual)

    beta_tan = w_fpt_tan * beta_fpt + w_vnm_tan * beta_vnm
    alpha_tan = w_fpt_tan * alpha_fpt + w_vnm_tan * alpha_vnm
    capm_ret_tan = rf_annual + beta_tan * (market_annual_return - rf_annual)

    # ==========================================================================
    # 6. PHÂN TÍCH PHÂN PHỐI & RỦI RO CHUYÊN SÂU BẰNG SCIPY (scipy.stats)
    # ==========================================================================
    dist_metrics = {}
    for ticker in stock_tickers:
        rets = stock_returns[ticker] * 100
        n = len(rets)
        mean_val = np.mean(rets)
        std_val = np.std(rets, ddof=1)
        median_val = np.median(rets)
        
        # Skewness & Kurtosis chuẩn xác từ scipy.stats
        skew_val = float(stats.skew(rets))
        kurt_val = float(stats.kurtosis(rets))  # Excess Kurtosis
        
        # Kiểm định phân phối chuẩn Jarque-Bera
        jb_stat, jb_pvalue = stats.jarque_bera(rets)
        
        # 1. Historical VaR 95%
        var_hist_95 = float(np.percentile(rets, 5))
        cvar_95 = float(rets[rets <= var_hist_95].mean())
        
        # 2. Parametric Gaussian VaR 95%
        var_param_95 = float(stats.norm.ppf(0.05, loc=mean_val, scale=std_val))
        
        # 3. Modified VaR (Cornish-Fisher Expansion) - Hiệu chỉnh theo Skewness & Kurtosis
        z = stats.norm.ppf(0.05)
        z_cf = z + (z**2 - 1)*skew_val/6 + (z**3 - 3*z)*kurt_val/24 - (2*z**3 - 5*z)*(skew_val**2)/36
        var_cf_95 = float(mean_val + z_cf * std_val)
        
        # 4. Kernel Density Estimation (KDE) mượt mà thực nghiệm
        kde = stats.gaussian_kde(rets)
        kde_x = np.linspace(min(rets.min(), -5), max(rets.max(), 5), 300)
        kde_pdf = kde(kde_x)
        norm_pdf = stats.norm.pdf(kde_x, loc=mean_val, scale=std_val)
        
        dist_metrics[ticker] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'skew': skew_val,
            'kurt': kurt_val,
            'jb_stat': jb_stat,
            'jb_pvalue': jb_pvalue,
            'var_95': var_hist_95,
            'var_param_95': var_param_95,
            'var_cf_95': var_cf_95,
            'cvar_95': cvar_95,
            'kde_x': kde_x,
            'kde_pdf': kde_pdf,
            'norm_pdf': norm_pdf,
            'rets_pct': rets
        }

    # ==========================================================================
    # 7. BẢNG THỐNG KÊ KINH TẾ LƯỢNG & ĐỊNH LƯỢNG TỔNG HỢP
    # ==========================================================================
    daily_mean_returns = daily_returns.mean()
    annual_arithmetic_returns = daily_mean_returns * trading_days
    total_days = len(raw_data)
    cagr_returns = (raw_data.iloc[-1] / raw_data.iloc[0]) ** (trading_days / total_days) - 1
    daily_variances = daily_returns.var()
    annual_variances = daily_variances * trading_days
    daily_stds = daily_returns.std()
    annual_stds = daily_stds * np.sqrt(trading_days)
    downside_var_daily = daily_returns.apply(lambda col: np.mean(np.minimum(col, 0)**2))
    downside_var_annual = downside_var_daily * trading_days
    downside_std_annual = np.sqrt(downside_var_annual)

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

    return {
        'raw_data': raw_data,
        'daily_returns': daily_returns,
        'stock_returns': stock_returns,
        'market_returns': market_returns,
        'stock_tickers': stock_tickers,
        'market_ticker': market_ticker,
        'rf_annual': rf_annual,
        'trading_days': trading_days,
        'rf_daily': rf_daily,
        'excess_market': excess_market,
        'annual_returns': annual_returns,
        'stock_vols': stock_vols,
        'market_annual_return': market_annual_return,
        'market_vol': market_vol,
        'daily_cov_matrix': daily_cov_matrix,
        'cov_matrix': cov_matrix,
        'corr_matrix': corr_matrix,
        'corr_fpt_vnm': corr_fpt_vnm,
        'cov_fpt_vnm_annual': cov_fpt_vnm_annual,
        'cov_fpt_vnm_daily': cov_fpt_vnm_daily,
        'weights_fpt': weights_fpt,
        'port_returns': port_returns,
        'port_volatilities': port_volatilities,
        'port_sharpes': port_sharpes,
        'min_vol_idx': min_vol_idx,
        'w_fpt_mvp': w_fpt_mvp,
        'w_vnm_mvp': w_vnm_mvp,
        'ret_mvp': ret_mvp,
        'vol_mvp': vol_mvp,
        'sharpe_mvp': sharpe_mvp,
        'w_fpt_tan': w_fpt_tan,
        'w_vnm_tan': w_vnm_tan,
        'ret_tan': ret_tan,
        'vol_tan': vol_tan,
        'sharpe_tan': sharpe_tan,
        'capm_stats': capm_stats,
        'beta_fpt': beta_fpt,
        'beta_vnm': beta_vnm,
        'alpha_fpt': alpha_fpt,
        'alpha_vnm': alpha_vnm,
        'weights_capm': weights_capm,
        'port_betas': port_betas,
        'port_alphas': port_alphas,
        'port_actual_capm_rets': port_actual_capm_rets,
        'port_capm_rets': port_capm_rets,
        'beta_mvp': beta_mvp,
        'alpha_mvp': alpha_mvp,
        'capm_ret_mvp': capm_ret_mvp,
        'beta_tan': beta_tan,
        'alpha_tan': alpha_tan,
        'capm_ret_tan': capm_ret_tan,
        'dist_metrics': dist_metrics,
        'metrics_df': metrics_df
    }
