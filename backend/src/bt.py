import numpy as np
import pandas as pd

class PortfolioBacktester:
    """
    Classe responsable du Backtesting; évaluation de la performance Out-of-Sample.
    Objectif : Prouver que le risque réalisé du portefeuille RMT est inférieur au Naïf.
    """

    def __init__(self, log_returns: pd.DataFrame, split_ratio: float = 0.7):
        """
        :param log_returns: L'historique complet des rendements logarithmiques.
        :param split_ratio: Le pourcentage de données à utiliser pour l'apprentissage ex: 0.7 = 70%.
        """
        self.log_returns = log_returns
        self.split_ratio = split_ratio
        
        # On calcule l'indice de coupure
        self.split_index = int(len(self.log_returns) * self.split_ratio)
        
        # 1. Découpage : Période d'Apprentissage; In-Sample
        self.train_returns = self.log_returns.iloc[:self.split_index]
        
        # 2. Découpage : Période de Test Out-of-Sample
        self.test_returns = self.log_returns.iloc[self.split_index:]

    def prepare_train_data(self):
        """
        Standardise uniquement les données d'apprentissage pour ne pas tricher en regardant le futur.
        """
        mu_train = self.train_returns.mean()
        sigma_train = self.train_returns.std()
        
        # Standardisation du Z-Score sur le Train Set
        x_norm_train = (self.train_returns - mu_train) / sigma_train
        
        return x_norm_train, sigma_train

    def compute_realized_volatility(self, weights: pd.Series) -> float:
        """
        Calcule la volatilité réalisée sur la période de test.
        """
        # Matrice de covariance empirique sur le FUTUR, Période de test
        sigma_test = self.test_returns.cov().values
        w = weights.values
        
        # Calcul mathématique de la variance du portefeuille
        variance_test = w.T @ sigma_test @ w
        
        # La volatilité est la racine carrée de la variance
        daily_volatility = np.sqrt(variance_test)
        
        # On annualise la volatilité (en multipliant par la racine de 252 jours de bourse)
        # pour que le chiffre soit lisible; ex: 15% par an au lieu de 0.8% par jour
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility