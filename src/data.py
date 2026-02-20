import yfinance as yf
import numpy as np
import pandas as pd

class MarketDataHandler:
    """
    Classe responsable de l'acquisition, du nettoyage et de la normalisation 
    des données financières (Étape A du projet).
    """

    def __init__(self, tickers: list, start_date: str, end_date: str):
        """
        Constructeur de la classe.
        :param tickers: Liste des symboles boursiers (ex: ['AAPL', 'MSFT', 'TSLA'])
        :param start_date: Date de début au format 'YYYY-MM-DD'
        :param end_date: Date de fin au format 'YYYY-MM-DD'
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        # Variables qui stockeront les matrices au fur et à mesure
        self.prices = None
        self.log_returns = None
        self.mu = None      # Moyennes (µ_i)
        self.sigma = None   # Écarts-types (σ_i)
        self.x_norm = None  # Rendements standardisés (Z-score)

    def fetch_data(self) -> pd.DataFrame:
        """
        Télécharge les prix de clôture depuis Yahoo Finance.
        """
        print(f"Téléchargement des données pour {len(self.tickers)} actifs...")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, threads=False)
        
        # Gestion de la mise à jour de yfinance
        try:
            self.prices = data['Adj Close']
        except KeyError:
            self.prices = data['Close']
            
        # Si un seul ticker est demandé, yfinance renvoie une "Series", on la transforme en "DataFrame"
        if isinstance(self.prices, pd.Series):
            self.prices = self.prices.to_frame(name=self.tickers[0])
            
        # Nettoyage : suppression des jours fériés ou données manquantes; NaN
        self.prices = self.prices.dropna()
        print(f"Données téléchargées : {self.prices.shape[0]} jours de cotation (T) pour {self.prices.shape[1]} actifs (N).")
        
        return self.prices

    def compute_log_returns(self) -> pd.DataFrame:
        """
        Calcule les rendements logarithmiques : r_{i,t} = ln(P_{i,t}) - ln(P_{i,t-1})
        """
        if self.prices is None:
            raise ValueError("Veuillez télécharger les données (fetch_data) en premier.")
        
        # Calcul vectoriel : np.log(Prix du jour / Prix de la veille)
        self.log_returns = np.log(self.prices / self.prices.shift(1))
        
        # On supprime la première ligne car elle ne peut pas avoir de rendement; pas de veille
        self.log_returns = self.log_returns.dropna()
        
        return self.log_returns

    def standardize_returns(self) -> tuple:
        """
        Standardise les rendements Z-Score : x_{i,t} = (r_{i,t} - µ_i) / σ_i
        Retourne la matrice normalisée ET le vecteur des volatilités (σ)
        """
        if self.log_returns is None:
            raise ValueError("Veuillez calculer les rendements (compute_log_returns) en premier.")
        
        # 1. Calcul de la moyenne temporelle pour chaque actif (µ_i)
        self.mu = self.log_returns.mean()
        
        # 2. Calcul de l'écart-type temporel pour chaque actif (σ_i)
        self.sigma = self.log_returns.std()
        
        # 3. Standardisation; Soustraction de la moyenne et division par l'écart-type
        self.x_norm = (self.log_returns - self.mu) / self.sigma
        
        return self.x_norm, self.sigma

    def process_all(self):
        """
        Méthode "Chef d'orchestre" interne à la classe qui exécute toutes les étapes d'un coup.
        """
        self.fetch_data()
        self.compute_log_returns()
        x_norm, sigma = self.standardize_returns()
        return x_norm, sigma