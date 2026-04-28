<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:0f2027,50:203a43,100:2c5364&height=280&section=header&text=RMT%20Portfolio%20Optimizer&fontSize=50&fontColor=00f5d4&animation=fadeIn&fontAlignY=40&desc=Random%20Matrix%20Theory%20%E2%97%A6%20Quantitative%20Finance%20Engine&descAlignY=62&descColor=ffd700" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00F5D4&center=true&vCenter=true&width=750&lines=Marchenko-Pastur+Eigenvalue+Filtering;Global+Minimum+Variance+Portfolio+Optimization;FastAPI+%E2%97%A6+React+%E2%97%A6+Tailwind;%241drien+%E2%97%A6+%40elkiliayma-sys" alt="Typing SVG" />

<br/><br/>

[![Python](https://img.shields.io/badge/Python_3.9+-FFD43B?style=for-the-badge&logo=python&logoColor=306998)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![React](https://img.shields.io/badge/React_18-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![Tailwind](https://img.shields.io/badge/Tailwind-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

<br/>

[![1drien](https://img.shields.io/badge/─%20@1drien%20─-0d1117?style=flat-square&logo=github&logoColor=00f5d4)](https://github.com/1drien)
[![elkiliayma](https://img.shields.io/badge/─%20@elkiliayma--sys%20─-0d1117?style=flat-square&logo=github&logoColor=9d8df1)](https://github.com/elkiliayma-sys)

</div>

<br/>

> A quantitative portfolio optimization engine based on **Random Matrix Theory** for filtering statistical noise from financial covariance matrices.
> Academic project — M2 FinTech, *Financial Programming*.

---

## Context & Problem

Markowitz portfolio optimization (Global Minimum Variance Portfolio) relies on inverting the covariance matrix of asset returns. In practice, this matrix is estimated from a limited historical sample, which introduces substantial **statistical noise**. Inverting a noisy matrix amplifies that noise and produces unstable allocations with extreme weights.

**Random Matrix Theory** (Marchenko–Pastur, 1967) provides a mathematical framework to distinguish genuine signal from pure statistical noise in empirical correlation matrices. By filtering out eigenvalues identified as noise, we obtain a more stable covariance matrix that yields more robust portfolios **out-of-sample**.

---

## The Mathematical Pipeline

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   Yahoo Finance  ──▶  log-returns  ──▶  Z-score  ──▶  C = X·Xᵀ/T   ║
║                                                          │           ║
║                                                          ▼           ║
║                                                   C = V·Λ·Vᵀ         ║
║                                                          │           ║
║                                                          ▼           ║
║                                            Marchenko–Pastur          ║
║                                              λ_max threshold         ║
║                                                          │           ║
║                                                          ▼           ║
║                                              Eigenvalue clipping     ║
║                                                          │           ║
║                                                          ▼           ║
║                                              Reconstruct  Σ̃          ║
║                                                          │           ║
║                                                          ▼           ║
║                              w* = Σ̃⁻¹·𝟙 / (𝟙ᵀ·Σ̃⁻¹·𝟙)                ║
║                                                          │           ║
║                                                          ▼           ║
║                              Out-of-sample backtest vs naive         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

</div>

<table align="center">
<tr>
<td align="center" width="50%">

### Marchenko–Pastur
Theoretical eigenvalue distribution
of a pure-noise correlation matrix
**→ defines the noise threshold λ_max**

</td>
<td align="center" width="50%">

### GMVP Optimization
Global Minimum Variance Portfolio
on the **filtered** covariance matrix
**→ stable out-of-sample weights**

</td>
</tr>
</table>

---

## Architecture

```
RMT-Portfolio-Optimizer/
│
├── backend/                          ← FastAPI service
│   ├── main.py                       ← HTTP entry point
│   ├── requirements.txt
│   └── src/
│       ├── data.py                   ← MarketDataHandler — acquisition & normalization
│       ├── cov.py                    ← RMTFilter — Marchenko–Pastur filtering
│       ├── opt.py                    ← PortfolioOptimizer — GMVP
│       ├── bt.py                     ← PortfolioBacktester — out-of-sample validation
│       ├── service.py                ← Pipeline orchestration
│       └── plots.py                  ← Matplotlib visualizations (dev)
│
└── frontend/                         ← React + Vite + Tailwind dashboard
    └── src/
        ├── App.jsx                   ← Main interface
        └── main.jsx
```

> The backend follows **Object-Oriented Programming** principles with a single responsibility per class.

---

## Tech Stack

<div align="center">

[![Python](https://skillicons.dev/icons?i=python)](https://python.org)
[![FastAPI](https://skillicons.dev/icons?i=fastapi)](https://fastapi.tiangolo.com)
[![React](https://skillicons.dev/icons?i=react)](https://react.dev)
[![Vite](https://skillicons.dev/icons?i=vite)](https://vitejs.dev)
[![Tailwind](https://skillicons.dev/icons?i=tailwind)](https://tailwindcss.com)

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.9+ · FastAPI · NumPy · Pandas · yfinance |
| **Frontend** | React 18 · Vite · Tailwind CSS · Recharts |
| **Data Source** | Yahoo Finance API |
| **Paradigm** | OOP (single-responsibility classes) |

</div>

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## ◈ Usage

1. Select an analysis period *(historical window)*
2. Choose the assets to include in the portfolio *(basket of tickers)*
3. Click **Generate Alpha**

The dashboard displays:

- **KPIs** — volatilities (naive vs RMT), stability index, Sharpe ratios, 95% VaR
- **Performance** — RMT portfolio vs S&P 500 on the test set
- **Marchenko–Pastur** — theoretical noise distribution, empirical eigenvalues, λ_max threshold
- **Correlation Matrix** — filtered
- **Allocation** — optimal weights of the RMT portfolio

---

## Key Metrics

<div align="center">

| Metric | Definition |
|--------|-----------|
| **Annualized Volatility** | σ × √252 |
| **Stability Index** | (vol_naive / vol_RMT − 1) × 100 |
| **Sharpe Ratio** | (R_p − R_f) / σ_p,  R_f = 3% |
| **95% VaR** | 5th percentile of daily returns |
| **Q Factor** | T / N (days-to-assets ratio) |

</div>

---

## References

- Marchenko, V. A. & Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices.*
- Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). *Noise dressing of financial correlation matrices.*
- Bouchaud, J.-P. & Potters, M. (2020). *A First Course in Random Matrix Theory.*

---

## Limitations & Possible Extensions

This project focuses on the standard RMT pipeline with an unconstrained GMVP. Several natural extensions were intentionally left out of scope:

- **No-short constraint** (w ≥ 0) via SLSQP optimization
- **Walk-forward backtesting** with rolling windows
- **Ledoit–Wolf shrinkage** as an alternative regularization method
- **Expected returns integration** to move from GMVP to the tangency portfolio *(max Sharpe)*

---

## License

Academic project — M2 FinTech, 2025

---

<div align="center">

[![@1drien](https://img.shields.io/badge/Made%20with%20RMT%20%26%20%CE%BB-eigenvalues-0d1117?style=for-the-badge&logo=github&logoColor=00f5d4)](https://github.com/1drien)
&nbsp;
[![@elkiliayma-sys](https://img.shields.io/badge/@elkiliayma--sys-0d1117?style=for-the-badge&logo=github&logoColor=9d8df1)](https://github.com/elkiliayma-sys)

<br/>

![wave](https://raw.githubusercontent.com/Trilokia/Trilokia/379277808c61ef204768a61bbc5d25bc7798ccf/bottom_header.svg)

</div>
