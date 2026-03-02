import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data import MarketDataHandler
from src.cov import RMTFilter
from src.opt import PortfolioOptimizer
from src.bt import PortfolioBacktester
from src.plots import PortfolioVisualizer

def run_pipeline():
    print("="*70)
    print("   ANALYSE COMPARATIVE : PORTEFEUILLE NAÏF VS FILTRAGE RMT (20 ACTIFS)")
    print("="*70 + "\n")

    # --- CONFIGURATION ---
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', # Tech
        'JPM', 'BAC', 'GS', 'V', 'MA',                          # Finance/Paiement
        'PG', 'KO', 'PEP', 'COST',                              # Consommation
        'JNJ', 'PFE', 'UNH', 'XOM'                               # Santé/Energie
    ]
    start_date = "2018-01-01"
    end_date = "2023-12-31"

    # --- ETAPE 1 : ACQUISITION ---
    print("--- ETAPE 1 : ACQUISITION DES DONNÉES ---")
    handler = MarketDataHandler(tickers, start_date, end_date)
    handler.fetch_data() 
    returns = handler.compute_log_returns()
    print(f"OK : {len(tickers)} actifs récupérés.\n")

    # --- ETAPE 2 : BACKTEST SPLIT ---
    print("--- ETAPE 2 : PRÉPARATION DU TEST ---")
    backtester = PortfolioBacktester(returns, split_ratio=0.7)
    x_norm_train, sigma_train = backtester.prepare_train_data()
    sigma_naive = backtester.train_returns.cov()

    # --- ETAPE 3 : FILTRAGE RMT ---
    print("--- ETAPE 3 : FILTRAGE RMT (MARCHENKO-PASTUR) ---")
    filter_rmt = RMTFilter(x_norm_train, sigma_train)
    sigma_rmt = filter_rmt.process_all()
    print(f"Lambda Max (Bruit) : {filter_rmt.lambda_max:.4f}\n")

    # --- ETAPE 4 : OPTIMISATION ET ANALYSE DES POIDS ---
    print("--- ETAPE 4 : OPTIMISATION & STABILITÉ DES POIDS ---")
    optimizer = PortfolioOptimizer()
    weights_df = optimizer.compare_portfolios(sigma_naive, sigma_rmt)
    
    # Calcul de métriques de stabilité
    # Une somme des valeurs absolues élevée indique des positions "long/short" extrêmes (instabilité)
    gross_exposure_naive = weights_df['Poids_Naifs'].abs().sum()
    gross_exposure_rmt = weights_df['Poids_RMT'].abs().sum()

    print(f"{'Métrique':<25} | {'Naïf (Bruité)':<15} | {'RMT (Filtré)':<15}")
    print("-" * 65)
    print(f"{'Plus grosse position':<25} | {weights_df['Poids_Naifs'].max():>14.2%} | {weights_df['Poids_RMT'].max():>14.2%}")
    print(f"{'Plus petite position':<25} | {weights_df['Poids_Naifs'].min():>14.2%} | {weights_df['Poids_RMT'].min():>14.2%}")
    print(f"{'Exposition Brute (Somme abs)':<25} | {gross_exposure_naive:>14.2f} | {gross_exposure_rmt:>14.2f}")
    print("\n*Note: Une exposition brute proche de 1.00 indique un portefeuille plus stable (moins d'effet de levier)*\n")

    # --- ETAPE 5 : PERFORMANCE RÉALISÉE ---
    print("--- ETAPE 5 : VOLATILITÉ RÉALISÉE (OUT-OF-SAMPLE) ---")
    vol_naive = backtester.compute_realized_volatility(weights_df['Poids_Naifs'])
    vol_rmt = backtester.compute_realized_volatility(weights_df['Poids_RMT'])

    print(f"Volatilité Naïve : {vol_naive:.2%}")
    print(f"Volatilité RMT   : {vol_rmt:.2%}")
    
    diff = vol_naive - vol_rmt
    if diff > 0:
        print(f"Résultat : Gain de {diff*100:.2f} points de risque grâce au RMT.")
    else:
        print(f"Résultat : La méthode naïve est marginalement plus stable sur cette période précise de {-diff*100:.2f} pts.")

    # --- ETAPE 6 : VISUALISATION ---
    print("\n--- ETAPE 6 : AFFICHAGE DES GRAPHIQUES ---")
    viz = PortfolioVisualizer()
    
    # Histogramme des valeurs propres
    viz.plot_marchenko_pastur(filter_rmt.eigenvalues, filter_rmt.Q, filter_rmt.lambda_max)
    
    # Comparaison visuelle des poids (Bar chart)
    viz.plot_weights_comparison(weights_df)

if __name__ == "__main__":
    run_pipeline()