import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Imports de vos fichiers source
from src.data import MarketDataHandler
from src.cov import RMTFilter
from src.opt import PortfolioOptimizer
from src.bt import PortfolioBacktester
from src.plots import PortfolioVisualizer

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="CY Fintech - RMT Optimizer Portfolio", layout="wide")

ASSETS_DATABASE = {
    "Apple Inc.": "AAPL", "Microsoft Corporation": "MSFT", "Alphabet Inc. (Google)": "GOOGL",
    "Amazon.com Inc.": "AMZN", "NVIDIA Corporation": "NVDA", "Tesla, Inc.": "TSLA",
    "Meta Platforms (Facebook)": "META", "JPMorgan Chase & Co.": "JPM", "Visa Inc.": "V",
    "Mastercard Incorporated": "MA", "The Coca-Cola Company": "KO", "PepsiCo, Inc.": "PEP",
    "Procter & Gamble Co.": "PG", "Johnson & Johnson": "JNJ", "Pfizer Inc.": "PFE",
    "Exxon Mobil Corporation": "XOM", "Chevron Corporation": "CVX", "The Walt Disney Company": "DIS",
    "Netflix, Inc.": "NFLX", "Goldman Sachs Group, Inc.": "GS"
}

# 2. STYLE CSS (Correctif visuel pour les dates et listes)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #003366; }
    [data-testid="stSidebar"] * { color: white !important; }
    input, select, .stSelectbox, div[role="listbox"], .stMultiSelect span { color: black !important; }
    div[data-baseweb="datepicker"] * { color: black !important; }
    .stButton>button {
        background-color: #0056b3; color: white; border-radius: 8px;
        width: 100%; border: none; padding: 12px; font-weight: bold;
    }
    h1, h2, h3 { color: #003366; }
    </style>
    """, unsafe_allow_html=True)

# 3. BARRE LATÉRALE (Navbar CY Fintech)
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: white;'>CY Fintech</h1>", unsafe_allow_html=True)
    st.markdown("---")
    options_list = list(ASSETS_DATABASE.keys())
    selected_names = st.multiselect("Sélectionnez les actifs :", options=options_list, default=options_list[:10])
    
    st.subheader("Période d'analyse")
    start_dt = st.date_input("Date de début", value=datetime(2018, 1, 1))
    end_dt = st.date_input("Date de fin", value=datetime(2023, 12, 31))
    
    run_btn = st.button("GÉNÉRER LE RAPPORT D'EXPERTISE")

# 4. LOGIQUE DE CALCUL
if run_btn:
    if len(selected_names) < 2:
        st.error("Veuillez sélectionner au moins 2 actifs.")
    else:
        tickers = [ASSETS_DATABASE[name] for name in selected_names]
        inv_map = {v: k for k, v in ASSETS_DATABASE.items()}

        with st.spinner("Calculs matriciels et alignement des données..."):
            try:
                # Récupération et préparation
                handler = MarketDataHandler(tickers, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
                handler.fetch_data()
                returns = handler.compute_log_returns()
                
                bt = PortfolioBacktester(returns, split_ratio=0.7)
                x_train, sigma_train = bt.prepare_train_data()
                
                # Filtrage RMT
                filter_rmt = RMTFilter(x_train, sigma_train)
                sigma_rmt = filter_rmt.process_all()
                sigma_naive = bt.train_returns.cov()
                
                # Optimisation
                opt = PortfolioOptimizer()
                weights_df = opt.compare_portfolios(sigma_naive, sigma_rmt)
                
                # --- CORRECTION : ALIGNEMENT DES MATRICES ---
                # On force les rendements de test à suivre l'ordre des actifs dans les poids
                current_tickers = weights_df.index.tolist()
                aligned_test_returns = bt.test_returns[current_tickers]
                
                # Traduction pour l'affichage
                display_weights = weights_df.copy()
                display_weights.index = [inv_map.get(t, t) for t in display_weights.index]

                # --- AFFICHAGE DES RÉSULTATS ---
                st.markdown("### 📊 Performance et Stabilité")
                v_n = bt.compute_realized_volatility(weights_df['Poids_Naifs'])
                v_r = bt.compute_realized_volatility(weights_df['Poids_RMT'])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Volatilité Naïve", f"{v_n:.2%}")
                c2.metric("Volatilité RMT", f"{v_r:.2%}", delta=f"{v_r-v_n:.2%}", delta_color="inverse")
                c3.metric("Gain de Stabilité", f"{((v_n/v_r)-1)*100:.2f}%")

                st.divider()

                # --- GRAPHIQUES LIGNE 1 ---
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### 🔍 Spectre de Marchenko-Pastur")
                    viz = PortfolioVisualizer()
                    viz.plot_marchenko_pastur(filter_rmt.eigenvalues, filter_rmt.Q, filter_rmt.lambda_max)
                    st.pyplot(plt.gcf())
                    plt.clf()
                
                with col2:
                    st.write("#### 🌡️ Corrélation Filtrée (Heatmap)")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(filter_rmt.filtered_C, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)
                    plt.clf()

                st.divider()

                # --- GRAPHIQUES LIGNE 2 ---
                col3, col4 = st.columns(2)
                with col3:
                    st.write("#### ⚖️ Comparaison des Poids")
                    fig, ax = plt.subplots()
                    display_weights[['Poids_Naifs', 'Poids_RMT']].plot(kind='bar', ax=ax, color=['#CCCCCC', '#003366'])
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                    plt.clf()

                with col4:
                    st.write("#### 🍩 Allocation Cible RMT")
                    fig, ax = plt.subplots()
                    ax.pie(np.abs(display_weights['Poids_RMT']), labels=display_weights.index, autopct='%1.1f%%', colors=sns.color_palette("Blues_r", len(display_weights)))
                    st.pyplot(fig)
                    plt.clf()

                st.divider()

                # --- GRAPHIQUES LIGNE 3 : BACKTEST ---
                st.write("#### 📈 Évolution de la Richesse Cumulée (Hors-Échantillon)")
                # Calcul propre du produit matriciel (Aligned)
                port_ret_rmt = aligned_test_returns @ weights_df['Poids_RMT']
                port_ret_naive = aligned_test_returns @ weights_df['Poids_Naifs']
                
                wealth_rmt = np.exp(port_ret_rmt.cumsum())
                wealth_naive = np.exp(port_ret_naive.cumsum())
                
                fig_p, ax_p = plt.subplots(figsize=(12, 4))
                ax_p.plot(wealth_rmt, label="Stratégie RMT (Filtrée)", color='#003366', lw=2.5)
                ax_p.plot(wealth_naive, label="Stratégie Naïve (Bruitée)", color='#CCCCCC', ls='--')
                ax_p.set_ylabel("Valeur du capital (Base 1)")
                ax_p.legend()
                st.pyplot(fig_p)
                plt.clf()

                # --- GRAPHIQUES LIGNE 4 : RISQUE ---
                st.write("#### 🔔 Distribution des Rendements (Profil de Risque)")
                fig_d, ax_d = plt.subplots(figsize=(12, 4))
                sns.kdeplot(port_ret_rmt, fill=True, color='#003366', label="RMT", ax=ax_d)
                sns.kdeplot(port_ret_naive, fill=True, color='#E0E0E0', label="Naïf", ax=ax_d)
                ax_d.legend()
                st.pyplot(fig_d)
                plt.clf()

                st.success("✅ Rapport CY Fintech généré avec succès.")

            except Exception as e:
                st.error(f"Erreur d'alignement ou de calcul : {e}")
else:
    st.info("Sélectionnez vos actifs dans la Navbar CY Fintech et lancez l'analyse.")