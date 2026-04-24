import numpy as np
import pandas as pd

class PortfolioBacktester:
    """
    Classe responsable du Backtesting; évaluation de la performance Out-of-Sample.
    Objectif : Prouver que le risque réalisé du portefeuille RMT est inférieur au Naïf.
    """

    def __init__(self, log_returns: pd.DataFrame, split_ratio: float = 0.7):
        self.log_returns = log_returns
        self.split_ratio = split_ratio
        self.split_index = int(len(self.log_returns) * self.split_ratio)
        self.train_returns = self.log_returns.iloc[:self.split_index]
        self.test_returns = self.log_returns.iloc[self.split_index:]

    def prepare_train_data(self):
        mu_train = self.train_returns.mean()
        sigma_train = self.train_returns.std()
        x_norm_train = (self.train_returns - mu_train) / sigma_train
        return x_norm_train, sigma_train

    def compute_realized_volatility(self, weights: pd.Series) -> float:
        sigma_test = self.test_returns.cov().values
        w = weights.values
        variance_test = w.T @ sigma_test @ w
        daily_volatility = np.sqrt(variance_test)
        annualized_volatility = daily_volatility * np.sqrt(252)
        return annualized_volatility

    def compute_sharpe_ratio(self, weights: pd.Series, risk_free_rate: float = 0.03) -> float:
        """
        Calcule le Sharpe Ratio annualisé sur la période de test.
        :param weights: poids du portefeuille
        :param risk_free_rate: taux sans risque annualisé (défaut 3%)
        """
        # Rendement du portefeuille sur le test
        port_returns = self.test_returns[weights.index].dot(weights)

        # Rendement annualisé (252 jours de bourse)
        annualized_return = port_returns.mean() * 252

        # Volatilité annualisée
        annualized_vol = port_returns.std() * np.sqrt(252)

        # Sharpe = (R_p - R_f) / sigma_p
        if annualized_vol == 0:
            return 0.0

        return (annualized_return - risk_free_rate) / annualized_vol