import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

def preparation_donnees(file_path):
    """
    Charge les données à partir d'un fichier CSV et retourne un tableau numpy.
    
    Args:
        file_path (str): Chemin vers le fichier CSV.
        
    Returns:
        np.ndarray: Données chargées sous forme de tableau numpy.
    """
    df = pd.read_csv(file_path)
    df = df.to_numpy()

    X = df[:, :-1]  
    y = df[:, -1] 

    print("Taille(X):", X.shape)
    print("Taille(Y):", Y.shape)

    total_samples = y.shape[0]
    positive_pourcentage = np.mean(y == 1) *100
    negative_pourcentage = np.mean(y == 0) *100

    print(f'Taille de l\'échantillon: {total_samples}')
    print(f'Pourcentage d\'exemples positifs: {positive_pourcentage:.2f}%')
    print(f'Pourcentage d\'exemples négatifs: {negative_pourcentage:.2f}%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    print("Dimensions de X_train:", X_train.shape)
    print("Dimensions de y_train:", y_train.shape)
    print("Dimensions de X_test:", X_test.shape)
    print("Dimensions de y_test:", y_test.shape)

