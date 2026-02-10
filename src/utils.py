import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import shap
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


SEED = 1


clfs = {
    'MLP': MLPClassifier(hidden_layer_sizes=(20, 10), random_state=SEED, max_iter=1000),
    'DT': DecisionTreeClassifier(criterion='gini', random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1), # n_jobs=-1 pour la vitesse
    'CART': DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=SEED),
    'ID3': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=SEED),
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
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, 
        random_state=SEED
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

    positive_pourcentage = np.mean(y == 1) *100
    negative_pourcentage = np.mean(y == 0) *100

    variables = df.columns[:-1].values
    print("Répartition des données : {pos:.2f}% positives, {neg:.2f}% négatives".format(pos=positive_pourcentage, neg=negative_pourcentage))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

    print("Données prêtes pour l'entrainement...")
    return X ,y, X_train, X_test, y_train, y_test, variables

def score(y_reel, y_pred):
    return (accuracy_score(y_reel, y_pred) + precision_score(y_reel, y_pred)) / 2 

monscore = make_scorer(score,greater_is_better=True)


def evaluate_classifier(clf, X_test, y_test):
    """
    Affiche la matrice de confusion, la courbe ROC et calcule le score métier.
    """
    y_pred = clf.predict(X_test)
    
    y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

    # 1. Matrice de Confusion
    print(f"\nMatrice de confusion pour {type(clf).__name__} :")
    print(confusion_matrix(y_test, y_pred))

    # 2. Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # Choix du meilleur critère métier (Précision vs Rappel)
    #  (accuracy + meilleur critère) / 2
    meilleur_critere = max(prec, rec)
    score_final = (acc + meilleur_critere) / 2
    
    # 3. Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, score_final, acc, prec, rec

def run_classifiers_train_test(X_train, X_test, y_train, y_test, clfs):
    """
    Entraîne, évalue et compare les modèles. Affiche les courbes ROC groupées.
    """
    plt.figure(figsize=(10, 8))
    results = {}
    best_score = 0
    best_model_name = ""

    for name, clf in clfs.items():
        # Entraînement
        clf.fit(X_train, y_train)
        
        # Évaluation
        fpr, tpr, roc_auc, final_s, acc, prec, rec = evaluate_classifier(clf, X_test, y_test)
        
        # Stockage pour comparaison
        results[name] = final_s
        
        # Plot de la courbe ROC
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        print(f"[{name}] Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | Score Final: {final_s:.3f}")
        
        if final_s > best_score:
            best_score = final_s
            best_model_name = name

    # ROC
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Comparaison des courbes ROC')
    plt.legend(loc='lower right')
    plt.show()

    print(f"\n Le meilleur modèle est : {best_model_name} avec un score de {best_score:.3f}")
    return best_model_name

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


def apprentissage_CV(X, Y, clfs):
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    
    best_score = 0
    best_clf_obj = None

    for name in clfs:
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clfs[name])
        ])
        
        cv_scores = cross_val_score(
            pipeline, X, Y, 
            cv=kf, 
            scoring=monscore, 
            n_jobs=-1 
        )
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"Score de {name}: {mean_score:.3f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_clf_obj = clfs[name]
    
    return best_clf_obj, best_score


def afficher_importance_variables(X, y, nom_cols):
    """
    Calcule l'importance des variables sur l'ensemble du dataset 
    avec une Random Forest.
    """
    
    nom_cols = np.array(nom_cols)
    
    # Entrainement global
    clf = RandomForestClassifier(n_estimators=1000, random_state=SEED, n_jobs=-1)
    clf.fit(X, y) 
    
    #  importances et calcul de l'écart-type (stabilité)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    
    # Tri par importance décroissante
    sorted_idx = np.argsort(importances)[::-1]
    
    # Graphique 
    plt.figure(figsize=(10, 8))
    padding = np.arange(len(nom_cols)) + 0.5
    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center', color='teal')
    plt.yticks(padding, nom_cols[sorted_idx])
    plt.xlabel("Importance Relative")
    plt.title("Importance des Variables (Random Forest - Global)")
    plt.gca().invert_yaxis() 
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return sorted_idx



def selection_nombre_variables_cv(X, Y, sorted_idx):
    # meilleur modèle (Gradient Boosting)
    model_top = GradientBoostingClassifier(n_estimators=200, random_state=1)
    
    #  KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    
    n_features = X.shape[1]
    scores_moyens = []

    print("--- ÉVALUATION PAR CROSS-VALIDATION (10-FOLD) ---")

    for f in range(1, n_features + 1):
        # 1. Sélection  meilleures variables
        X_sub = X[:, sorted_idx[:f]]
        
        # 2. Pipeline : Normalisation + Modèle
     
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_top)
        ])
        
        # 3. Calcul du score avec  monscore sur X et Y 
        cv_results = cross_val_score(pipeline, X_sub, Y, cv=kf, scoring=monscore)
        
        mean_score = np.mean(cv_results)
        scores_moyens.append(mean_score)
        print(f"Variables: {f} | Score CV moyen: {mean_score:.4f}")

    # Graphique
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_features + 1), scores_moyens, marker='o', color='forestgreen')
    plt.xlabel("Nombre de Variables")
    plt.ylabel("Score (Acc + Prec / 2)")
    plt.title("Évolution de la performance par Validation Croisée")
    plt.grid(True)
    plt.show()
    
    return scores_moyens



def expliquer_modele_shap(model, X_train, X_test, nom_cols):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("Résumé global des influences des variables :")
    shap.summary_plot(shap_values, X_test, feature_names=nom_cols)
    
    # Visualisation locale
    print("Explication locale pour la première prédiction :")
    shap.initjs()
    
    return shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:], feature_names=nom_cols)