"""
========================================================================================
PORTFOLIO ENGINE: QUANTITATIVE CALCULATION & DATA PROCESSING MODULE
========================================================================================
Module xử lý dữ liệu tài chính, tính toán Markowitz, CAPM, SML, VaR, CVaR cho FPT và VNM.
"""

import yfinance as yf
import pandas as pd
import numpy as np

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

    # 4. Mô phỏng Efficient Frontier (201 kịch bản trọng số)
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

    # Điểm MVP (Minimum Variance Portfolio)
    min_vol_idx = np.argmin(port_volatilities)
    w_fpt_mvp = weights_fpt[min_vol_idx]
    w_vnm_mvp = 1.0 - w_fpt_mvp
    ret_mvp = port_returns[min_vol_idx]
    vol_mvp = port_volatilities[min_vol_idx]
    sharpe_mvp = port_sharpes[min_vol_idx]

    # Điểm Tangency (Max Sharpe Portfolio)
    max_sharpe_idx = np.argmax(port_sharpes)
    w_fpt_tan = weights_fpt[max_sharpe_idx]
    w_vnm_tan = 1.0 - w_fpt_tan
    ret_tan = port_returns[max_sharpe_idx]
    vol_tan = port_volatilities[max_sharpe_idx]
    sharpe_tan = port_sharpes[max_sharpe_idx]

    # 5. Tính toán CAPM OLS Regression & Beta
    rf_daily = rf_annual / trading_days
    excess_market = market_returns - rf_daily

    capm_stats = {}
    for ticker in stock_tickers:
        excess_stock = stock_returns[ticker] - rf_daily
        beta, alpha_daily = np.polyfit(excess_market, excess_stock, 1)
        alpha_annual = alpha_daily * trading_days
        
        y_pred = alpha_daily + beta * excess_market
        ss_res = np.sum((excess_stock - y_pred) ** 2)
        ss_tot = np.sum((excess_stock - np.mean(excess_stock)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        expected_capm = rf_annual + beta * (market_annual_return - rf_annual)
        
        capm_stats[ticker] = {
            'beta': beta,
            'alpha_annual': alpha_annual,
            'r_squared': r_squared,
            'expected_capm': expected_capm
        }

    # 6. CAPM theo dải tỷ trọng w từ 0 đến 1 (101 kịch bản)
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

    # 7. Thống kê phân phối & VaR / CVaR
    dist_metrics = {}
    for ticker in stock_tickers:
        rets = stock_returns[ticker] * 100
        n = len(rets)
        mean_val = np.mean(rets)
        std_val = np.std(rets, ddof=1)
        median_val = np.median(rets)
        skew_val = (np.sum((rets - mean_val)**3) / n) / (std_val**3)
        kurt_val = (np.sum((rets - mean_val)**4) / n) / (std_val**4) - 3
        var_95 = np.percentile(rets, 5)
        cvar_95 = rets[rets <= var_95].mean()
        
        dist_metrics[ticker] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'skew': skew_val,
            'kurt': kurt_val,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'rets_pct': rets
        }

    # 8. Bảng thống kê chi tiết Expected Return & Variance
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
        'max_sharpe_idx': max_sharpe_idx,
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
