# RMT Portfolio Optimizer

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3-06B6D4?style=flat&logo=tailwindcss&logoColor=white)
![License](https://img.shields.io/badge/license-Academic-lightgrey)
![Status](https://img.shields.io/badge/status-active-success)

A quantitative portfolio optimization engine based on **Random Matrix Theory** (RMT) for filtering statistical noise from financial covariance matrices.

Academic project for the Financial Programming course — M2 FinTech.

---

## Context and problem statement

Markowitz portfolio optimization (Global Minimum Variance Portfolio) relies on inverting the covariance matrix of asset returns. In practice, this matrix is estimated from a limited historical sample, which introduces substantial statistical noise. Inverting a noisy matrix amplifies the noise and produces unstable allocations with extreme weights.

**Random Matrix Theory** (Marchenko-Pastur, 1967) provides a mathematical framework for distinguishing genuine signal from pure statistical noise in empirical correlation matrices. By filtering out eigenvalues identified as noise, we obtain a more stable covariance matrix that yields more robust portfolios **out-of-sample**.

---

## Theoretical pipeline

1. **Data acquisition** of historical prices via the Yahoo Finance API
2. **Log-returns computation** to stationarize the time series
3. **Z-score standardization** (matrix X with zero mean, unit variance)
4. **Empirical correlation matrix** C = X·Xᵀ / T
5. **Spectral decomposition** C = V·Λ·Vᵀ
6. **Marchenko-Pastur filtering**: computation of the critical threshold λ_max
7. **Clipping**: noisy eigenvalues replaced by their mean
8. **Reconstruction** of the filtered covariance matrix Σ̃
9. **GMVP optimization**: w* = Σ̃⁻¹·1 / (1ᵀ·Σ̃⁻¹·1)
10. **Out-of-sample backtesting**: realized volatility comparison vs naive approach

---

## Technical architecture

```
RMT-Portfolio-Optimizer/
├── backend/                 # FastAPI service
│   ├── main.py              # HTTP entry point
│   ├── requirements.txt
│   └── src/
│       ├── data.py          # MarketDataHandler: acquisition & normalization
│       ├── cov.py           # RMTFilter: Marchenko-Pastur filtering
│       ├── opt.py           # PortfolioOptimizer: GMVP
│       ├── bt.py            # PortfolioBacktester: out-of-sample validation
│       ├── service.py       # Pipeline orchestration
│       └── plots.py         # Matplotlib visualizations (dev)
└── frontend/                # React + Vite + Tailwind dashboard
    └── src/
        ├── App.jsx          # Main interface
        └── main.jsx
```

The backend follows **Object-Oriented Programming** principles with a single responsibility per class.

---

## Tech stack

**Backend**
- Python 3.9+
- FastAPI (REST API)
- NumPy, Pandas (matrix algebra)
- yfinance (market data)

**Frontend**
- React 18
- Vite (bundler)
- Tailwind CSS (styling)
- Recharts (visualizations)

---

## Installation and launch

### Prerequisites
- Python 3.9+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI is available on `http://localhost:5173`.

---

## Usage

1. Select an analysis period (historical window).
2. Choose the assets to include in the portfolio (basket of tickers).
3. Click **Generate Alpha**.

The dashboard displays:
- **KPIs**: volatilities (naive vs RMT), stability index, Sharpe ratios, 95% VaR
- **Performance**: RMT portfolio vs S&P 500 on the test set
- **Marchenko-Pastur**: theoretical noise distribution, empirical eigenvalues, λ_max threshold
- **Correlation matrix** (filtered)
- **Allocation**: optimal weights of the RMT portfolio

---

## Key metrics

| Metric | Definition |
|--------|-----------|
| Annualized volatility | σ × √252 |
| Stability Index | (vol_naive / vol_RMT - 1) × 100 |
| Sharpe Ratio | (R_p - R_f) / σ_p, R_f = 3% |
| 95% VaR | 5th percentile of daily returns |
| Q factor | T / N (days-to-assets ratio) |

---

## References

- Marchenko, V. A. & Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices.*
- Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). *Noise dressing of financial correlation matrices.*
- Bouchaud, J.-P. & Potters, M. (2020). *A First Course in Random Matrix Theory.*

---

## Limitations and possible extensions

This project focuses on the standard RMT pipeline with an unconstrained GMVP. Several natural extensions were intentionally left out of scope:

- **No-short constraint** (w ≥ 0) via SLSQP optimization
- **Walk-forward backtesting** with rolling windows
- **Ledoit-Wolf shrinkage** as an alternative regularization method
- **Expected returns integration** to move from GMVP to the tangency portfolio (max Sharpe)

---

## Contributors

- [@1drien](https://github.com/1drien)
- [@elkiliayma-sys](https://github.com/elkiliayma-sys)

---

## License

Academic project — M2 FinTech, 2025
