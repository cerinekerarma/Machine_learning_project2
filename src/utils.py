import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

SEED = 1
clfs = {
    'MLP' : MLPClassifier(hidden_layer_sizes=(40,20), random_state=SEED), # 13 (variables) -40, 40-20, 20-1
    'DT' : DecisionTreeClassifier(criterion='gini', random_state=SEED),
    'KNN' : KNeighborsClassifier(n_neighbors=5, n_jobs=1),
    'CART': DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=SEED),
    'ID3': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=SEED),  # ID3 ≈ entropie
    'MLP': MLPClassifier(hidden_layer_sizes=(20, 10), random_state=SEED, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
    'Bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
        n_estimators=200,
        random_state=SEED,
        n_jobs=1
    ),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
        n_estimators=200,
        random_state=SEED
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=SEED,
        n_jobs=1
    )

}


'''clfs = {
    'MLP' : MLPClassifier(hidden_layer_sizes=(40,20), random_state=SEED), # 13 (variables) -40, 40-20, 20-1
    'DT' : DecisionTreeClassifier(criterion='gini', random_state=SEED),
    'KNN' : KNeighborsClassifier(n_neighbors=5, n_jobs=1),
    'CART': DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=SEED),
    'ID3': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=SEED),  # ID3 ≈ entropie
    'MLP': MLPClassifier(hidden_layer_sizes=(20, 10), random_state=SEED, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    ),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
        n_estimators=200,
        random_state=SEED
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=SEED,
        n_jobs=-1
    )

}'''

def preparer_donnees(file_path):
    df = pd.read_csv('../data/'+file_path, sep=';', header=0)

    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].values

    # print("Taille(X):", X.shape)
    # print("Taille(Y):", Y.shape)

    positive_pourcentage = np.mean(y == 1) *100
    negative_pourcentage = np.mean(y == 0) *100

    variables = df.columns[:-1].values
    print("Répartition des données : {pos:.2f}% positives, {neg:.2f}% négatives".format(pos=positive_pourcentage, neg=negative_pourcentage))
    # print(f'Taille de l\'échantillon: {total_samples}')
    # print(f'Pourcentage d\'exemples positifs: {positive_pourcentage:.2f}%')
    # print(f'Pourcentage d\'exemples négatifs: {negative_pourcentage:.2f}%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    # print("Dimensions de X_train:", X_train.shape)
    # print("Dimensions de y_train:", y_train.shape)
    # print("Dimensions de X_test:", X_test.shape)
    # print("Dimensions de y_test:", y_test.shape)

    print("Données prêtes pour l'entrainement...")
    return X ,y, X_train, X_test, y_train, y_test, variables

def score(y_reel, y_pred):
    return (accuracy_score(y_reel, y_pred) + precision_score(y_reel, y_pred)) / 2 

monscore = make_scorer(score,greater_is_better=True)


def decoupage_train_test(X, Y, taux_test):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=taux_test,stratify=Y,random_state=SEED)
    return X_train, X_test, Y_train, Y_test

def apprentissage_train_test(X_train, X_test, Y_train, Y_test, clfs):
    for i in clfs:
        clf = clfs[i]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        test_score = score(Y_test, Y_pred)
        print(f"{i} -> Test Score: {test_score*100:.2f}%")
        print(confusion_matrix(Y_test, Y_pred))
        print()


def normalisation(X_train, X_test):
    scaler = StandardScaler()
    X_train_normalise = scaler.fit_transform(X_train)
    X_test_normalise = scaler.transform(X_test)
    
    return X_train_normalise, X_test_normalise