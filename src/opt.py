import numpy as np
import pandas as pd

class PortfolioOptimizer:
    """
    Classe responsable de l'allocation d'actifs selon la théorie de Markowitz.
    Objectif : Trouver le Global Minimum Variance Portfolio (GMVP).
    """

    def __init__(self):
        # Pour le GMVP, nous n'avons besoin que de la matrice de covariance
        pass

    def gmvp_optimization(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calcule les poids optimaux w* selon la formule analytique de Lagrange.
        Formule : w* = (Sigma^-1 * 1) / (1^T * Sigma^-1 * 1)
        """
        # 1. Nombre d'actifs (N)
        N = cov_matrix.shape[0]
        
        # 2. Vecteur colonne de taille N rempli de 1; Le vecteur '1'
        ones = np.ones(N)
        
        # 3. Inversion de la matrice de covariance (Sigma^-1)
        # On utilise 'pinv' (pseudo-inverse) au lieu de 'inv' car c'est numériquement 
        # plus stable si la matrice a des valeurs proches de zéro.
        inv_cov = np.linalg.pinv(cov_matrix.values)
        
        # 4. Le Numérateur : (Sigma^-1 * 1)
        numerator = inv_cov @ ones
        
        # 5. Le Dénominateur : un scalaire
        denominator = ones.T @ inv_cov @ ones
        
        # 6. Le calcul final des poids
        weights = numerator / denominator
        
        # On retourne une Série Pandas avec le nom des actions pour une belle présentation
        return pd.Series(weights, index=cov_matrix.index, name="Poids")

    def compare_portfolios(self, sigma_naive: pd.DataFrame, sigma_rmt: pd.DataFrame) -> pd.DataFrame:
        """
        Génère la sortie attendue par le cahier des charges : 
        Le comparatif entre l'approche classique (bruitée) et l'approche RMT (robuste).
        """
        # Calcul des poids avec la matrice empirique
        w_naive = self.gmvp_optimization(sigma_naive)
        
        # Calcul des poids avec la matrice nettoyée
        w_rmt = self.gmvp_optimization(sigma_rmt)
        
        # Regroupement dans un tableau comparatif
        comparison_df = pd.DataFrame({
            'Poids_Naifs': w_naive,
            'Poids_RMT': w_rmt
        })
        
        # On ajoute une colonne de "Différence" pour voir à quel point le filtre a modifié la décision
        comparison_df['Difference'] = comparison_df['Poids_RMT'] - comparison_df['Poids_Naifs']
        
        return comparison_df