import numpy as np
import yfinance as yf
from src.data import MarketDataHandler
from src.cov import RMTFilter
from src.opt import PortfolioOptimizer
from src.bt import PortfolioBacktester

class PortfolioService:

    def run_analysis(self, tickers, start_date, end_date, split_ratio):
        # 1. DATA
        handler = MarketDataHandler(tickers, start_date, end_date)
        handler.fetch_data()
        returns = handler.compute_log_returns()

        # 2. BENCHMARK
        spy = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
        spy_price = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
        spy_ret = np.log(spy_price / spy_price.shift(1)).dropna()

        # 3. RMT & OPT
        bt = PortfolioBacktester(returns, split_ratio=split_ratio)
        x_train, sigma_train = bt.prepare_train_data()
        filter_rmt = RMTFilter(x_train, sigma_train)
        sigma_rmt = filter_rmt.process_all()

        opt = PortfolioOptimizer()
        weights_df = opt.compare_portfolios(bt.train_returns.cov(), sigma_rmt)

        # 4. PERFORMANCE
        aligned_test = bt.test_returns[weights_df.index]
        port_ret_rmt = aligned_test.dot(weights_df['Poids_RMT']).fillna(0)
        perf_rmt = (np.exp(port_ret_rmt.cumsum()) * 100).values.flatten().tolist()
        spy_test = spy_ret.reindex(aligned_test.index).fillna(0)
        perf_spy = (np.exp(spy_test.cumsum()) * 100).values.flatten().tolist()
        dates = aligned_test.index.strftime('%Y-%m-%d').tolist()

        # 5. METRIQUES
        vol_n = float(bt.compute_realized_volatility(weights_df['Poids_Naifs']))
        vol_r = float(bt.compute_realized_volatility(weights_df['Poids_RMT']))
        gain_val = float(((vol_n / vol_r) - 1) * 100) if vol_r != 0 else 0.0
        var_95 = float(np.percentile(port_ret_rmt.values, 5))

        # Sharpe ratios
        sharpe_n = float(bt.compute_sharpe_ratio(weights_df['Poids_Naifs']))
        sharpe_r = float(bt.compute_sharpe_ratio(weights_df['Poids_RMT']))

        # 6. MARCHENKO-PASTUR
        Q = filter_rmt.Q
        lambda_max = filter_rmt.lambda_max
        lambda_min = float((1 - np.sqrt(1 / Q)) ** 2)
        eigenvalues = filter_rmt.eigenvalues.tolist()
        x_mp = np.linspace(lambda_min * 1.001, lambda_max * 0.999, 300)
        y_mp = (Q / (2 * np.pi)) * np.sqrt(
            (lambda_max - x_mp) * (x_mp - lambda_min)
        ) / x_mp
        n_signal = int(np.sum(filter_rmt.eigenvalues > lambda_max))

        return {
            "metrics": {
                "vol_naive": round(vol_n * 100, 2),
                "vol_rmt": round(vol_r * 100, 2),
                "gain": round(gain_val, 2),
                "var_95": round(var_95 * 100, 2),
                "sharpe_naive": round(sharpe_n, 2),
                "sharpe_rmt": round(sharpe_r, 2)
            },
            "marchenko_pastur": {
                "eigenvalues": [float(round(v, 4)) for v in eigenvalues],
                "curve": [
                    {"x": float(round(x, 4)), "y": float(round(y, 4))}
                    for x, y in zip(x_mp.tolist(), y_mp.tolist())
                ],
                "lambda_max": float(round(lambda_max, 4)),
                "lambda_min": float(round(lambda_min, 4)),
                "Q": float(round(Q, 2)),
                "n_signal": n_signal
            },
            "heatmap": {
                "labels": list(weights_df.index),
                "data": [[float(round(val, 2)) for val in row] 
                         for row in sigma_rmt.corr().values.tolist()]
            },
            "chart_data": [
                {"date": d, "rmt": float(round(r, 2)), "spy": float(round(s, 2))}
                for d, r, s in zip(dates, perf_rmt, perf_spy)
            ],
            "weights": [
                {"name": str(n), "rmt": float(round(w * 100, 2))}
                for n, w in zip(weights_df.index, weights_df['Poids_RMT'])
            ]
        }