import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PortfolioVisualizer:
    """
    Classe responsable de la génération des graphiques.
    """

    def __init__(self):
        # On configure le style global des graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_marchenko_pastur(self, eigenvalues: np.ndarray, Q: float, lambda_max: float):
        """
        Trace la distribution empirique des valeurs propres vs la courbe théorique RMT.
        """
        # 1. Calcul des bornes théoriques
        lambda_min = (1 - np.sqrt(1/Q))**2
        
        # 2. Création de l'axe X pour la courbe théorique; de lambda_min à lambda_max
        x = np.linspace(lambda_min, lambda_max, 1000)
        
        # 3. Formule de la densité de Marchenko-Pastur
        # f(x) = (Q / (2*pi)) * sqrt((lambda_max - x)*(x - lambda_min)) / x
        y = (Q / (2 * np.pi)) * np.sqrt((lambda_max - x) * (x - lambda_min)) / x
        
        # 4. Création de la figure
        plt.figure(figsize=(10, 6))
        
        # Tracer la courbe théorique (Le Bruit) en rouge
        plt.plot(x, y, color='red', linewidth=2, label='Densité de Marchenko-Pastur (Bruit théorique)')
        
        # Tracer les valeurs propres empiriques (Nos actions) en points bleus
        # On met des hauteurs arbitraires justes pour les visualiser sur l'axe X
        plt.scatter(eigenvalues, np.zeros_like(eigenvalues), color='blue', s=100, zorder=5, label='Valeurs propres empiriques')
        
        # Ajouter une ligne verticale pour le seuil; Lambda Max
        plt.axvline(x=lambda_max, color='black', linestyle='--', label=f'Seuil critique Lambda Max ({lambda_max:.2f})')
        
        # Zone de bruit
        plt.axvspan(0, lambda_max, alpha=0.2, color='red', label='Zone de Bruit')
        # Zone de signal
        plt.axvspan(lambda_max, max(eigenvalues.max() + 0.5, lambda_max + 0.5), alpha=0.2, color='green', label='Zone de Signal')

        plt.title('Filtrage RMT : Séparation du Bruit et du Signal', fontsize=14, fontweight='bold')
        plt.xlabel('Valeur de la valeur propre (\u03BB)', fontsize=12)
        plt.ylabel('Densité de probabilité', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_weights_comparison(self, comparison_df: pd.DataFrame):
        """
        Trace un graphique en barres comparant les poids Naïfs vs RMT.
        Les barres permettent de visualiser facilement les ventes à découvert (poids négatifs).
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Paramètres pour décaler les barres côte à côte
        x = np.arange(len(comparison_df.index))
        width = 0.35  
        
        # Création des barres
        rects1 = ax.bar(x - width/2, comparison_df['Poids_Naifs'] * 100, width, label='Classique (Matrice Bruitée)', color='salmon')
        rects2 = ax.bar(x + width/2, comparison_df['Poids_RMT'] * 100, width, label='Robuste (Filtre RMT)', color='steelblue')
        
        # Ligne horizontale à 0%
        ax.axhline(0, color='black', linewidth=1.5)
        
        # Textes et labels
        ax.set_ylabel('Poids dans le portefeuille (%)', fontsize=12)
        ax.set_title("Comparaison des Allocations : Markowitz Classique vs RMT", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.show()





# ==========================================
# ZONE DE TEST (Lancement de tout le pipeline)
# ==========================================
if __name__ == "__main__":
    from data import MarketDataHandler
    from cov import RMTFilter
    from opt import PortfolioOptimizer
    
    print("=== DÉMARRAGE DU PIPELINE COMPLET ===")
    
    # 1. Données
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data_handler = MarketDataHandler(tickers=tickers, start_date="2020-01-01", end_date="2023-01-01")
    x_norm, sigma = data_handler.process_all()
    
    # 2. Matrices (Naïve & RMT)
    sigma_naive = data_handler.log_returns.cov()
    
    rmt_filter = RMTFilter(x_norm=x_norm, sigma=sigma)
    sigma_rmt = rmt_filter.process_all()
    
    # 3. Optimisation Markowitz
    optimizer = PortfolioOptimizer()
    comparatif = optimizer.compare_portfolios(sigma_naive, sigma_rmt)
    
    # 4. VISUALISATION !
    print("\n=== GÉNÉRATION DES GRAPHIQUES ===")
    visualizer = PortfolioVisualizer()
    
    # Graphique 1 : La théorie RMT
    print("-> Affichage de la courbe de Marchenko-Pastur... (Fermez la fenêtre pour voir le suivant)")
    visualizer.plot_marchenko_pastur(
        eigenvalues=rmt_filter.eigenvalues, 
        Q=rmt_filter.Q, 
        lambda_max=rmt_filter.lambda_max
    )
    
    # Graphique 2 : Le résultat d'investissement
    print("-> Affichage des Poids du Portefeuille...")
    visualizer.plot_weights_comparison(comparison_df=comparatif)
    
    print("=== FIN DU PROGRAMME ===")