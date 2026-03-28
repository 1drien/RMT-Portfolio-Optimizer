import numpy as np
import pandas as pd

class RMTFilter:
    """
    Classe responsable du filtrage de la matrice de covariance via la Théorie 
    des Matrices Aléatoires.
    """

    def __init__(self, x_norm: pd.DataFrame, sigma: pd.Series):
        """
        :param x_norm: Matrice des rendements standardisés; T jours x N actifs
        :param sigma: Vecteur des volatilités empiriques
        """
        self.x_norm = x_norm
        self.sigma = sigma
        
        # Dimensions T = nombre de jours, N = nombre d'actifs
        self.T, self.N = x_norm.shape
        self.Q = self.T / self.N  # Facteur de qualité
        
        # Variables de stockage
        self.C_emp = None         # Matrice de corrélation empirique
        self.eigenvalues = None   # Valeurs propres; lambda
        self.eigenvectors = None  # Vecteurs propres V
        self.lambda_max = None    # Borne de Marchenko-Pastur
        self.filtered_C = None    # Matrice de corrélation nettoyée
        self.filtered_Sigma = None # Matrice de covariance nettoyée

    def compute_empirical_correlation(self):
        """ Étape B.1 : Calcul de la matrice de corrélation empirique C """
        # La méthode .corr() de Pandas permet de le calculer exactementpour des données centrées-réduites
        self.C_emp = self.x_norm.corr()
        return self.C_emp

    def spectral_decomposition(self):
        """ Étape B.2 : Diagonalisation; extraction des valeurs et vecteurs propres """
        # np.linalg.eigh est optimisé pour les matrices symétriques, comme les corrélations
        eigenvalues, eigenvectors = np.linalg.eigh(self.C_emp.values)
        
        # np.linalg.eigh les trie par ordre croissant. On les veut par ordre décroissant
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        return self.eigenvalues, self.eigenvectors

    def filter_noise_marchenko_pastur(self):
        """ Étape C : Application de la borne MP et Algorithme de Clipping """
        # 1. Calcul de la borne critique; Lambda Max
        # Comme on travaille sur une matrice de corrélation, la variance théorique est 1
        self.lambda_max = 1 * (1 + np.sqrt(1 / self.Q))**2
        
        # 2. Séparation du Signal et du Bruit
        # On trouve les indices des valeurs propres qui sont inférieures ou égales à lambda_max
        noise_indices = self.eigenvalues <= self.lambda_max
        
        # 3. Calcul de la moyenne du bruit, Lambda_barre
        # S'il n'y a pas de bruit, la moyenne est 0
        if noise_indices.any():
            lambda_bar = self.eigenvalues[noise_indices].mean()
        else:
            lambda_bar = 0
            
        # 4. Création de la nouvelle matrice diagonale filtrée, Lambda_tilde
        filtered_eigenvalues = self.eigenvalues.copy()
        filtered_eigenvalues[noise_indices] = lambda_bar
        
        # 5. Reconstruction de la matrice de corrélation propre
        self.filtered_C = self.eigenvectors @ np.diag(filtered_eigenvalues) @ self.eigenvectors.T
        
        # Astuce mathématique : On remet des 1 sur la diagonale principale car une corrélation a toujours 1 avec elle-même
        np.fill_diagonal(self.filtered_C, 1.0)
        
        # On reconvertit en DataFrame Pandas pour que ce soit joli avec les noms des actions
        self.filtered_C = pd.DataFrame(self.filtered_C, index=self.C_emp.index, columns=self.C_emp.columns)
        
        return self.filtered_C

    def reconstruct_covariance(self):
        """ 
        Reconstruction finale
        Sigma_tilde_ij = C_tilde_ij * sigma_i * sigma_j 
        """
        # On convertit le vecteur de volatilités en une matrice diagonale
        D = np.diag(self.sigma.values)
        
        # Multiplications matricielles : D * C_tilde * D
        cov_matrix = D @ self.filtered_C.values @ D
        
        self.filtered_Sigma = pd.DataFrame(cov_matrix, index=self.C_emp.index, columns=self.C_emp.columns)
        
        return self.filtered_Sigma

    def process_all(self):
        """ Le chef d'orchestre interne de la classe """
        self.compute_empirical_correlation()
        self.spectral_decomposition()
        self.filter_noise_marchenko_pastur()
        self.reconstruct_covariance()
        return self.filtered_Sigma