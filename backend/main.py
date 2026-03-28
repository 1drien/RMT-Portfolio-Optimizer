from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf

# Imports de tes modules src
from src.data import MarketDataHandler
from src.cov import RMTFilter
from src.opt import PortfolioOptimizer
from src.bt import PortfolioBacktester

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    split_ratio: float

@app.post("/api/analyze")
async def analyze_portfolio(req: AnalysisRequest):
    try:
        # 1. DATA
        handler = MarketDataHandler(req.tickers, req.start_date, req.end_date)
        handler.fetch_data()
        returns = handler.compute_log_returns()
        
        # 2. BENCHMARK
        spy = yf.download("^GSPC", start=req.start_date, end=req.end_date, progress=False)
        spy_price = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
        spy_ret = np.log(spy_price / spy_price.shift(1)).dropna()

        # 3. RMT & OPT
        bt = PortfolioBacktester(returns, split_ratio=req.split_ratio)
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

        return {
            "metrics": {
                "vol_naive": round(vol_n * 100, 2),
                "vol_rmt": round(vol_r * 100, 2),
                "gain": round(gain_val, 2),
                "var_95": round(var_95 * 100, 2)
            },
            "heatmap": {
                "labels": list(weights_df.index),
                "data": [[float(round(val, 2)) for val in row] for row in sigma_rmt.corr().values.tolist()]
            },
            "chart_data": [
                {"date": d, "rmt": float(round(r, 2)), "spy": float(round(s, 2))} 
                for d, r, s in zip(dates, perf_rmt, perf_spy)
            ],
            "weights": [
                {"name": str(n), "rmt": float(round(w * 100, 2))} 
                for n, w in zip(weights_df.index, weights_df['Poids_RMT'])]
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)