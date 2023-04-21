import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

PAD_X = 10
PAD_Y = 5
P_CLASSES = {"1ère classe": 1, "2ème classe": 2, "3ème classe": 3}
BOARDING_CITIES = {"Cherbourg": 'C', "Queenstown": 'Q', "Southampton": 'S'}

TEXT_DT_METRICS = "..."
TEXT_KNN_METRICS = "..."
TEXT_REG_METRICS = "..."

TEXT_DT_RESULT = "..."
TEXT_KNN_RESULT = "..."
TEXT_REG_RESULT = "..."

# Création de la frame principale
root = tk.Tk()
root.title("Formulaire Titanic")
root.geometry("1920x1080")

# Création de la frame contenant formulaire
form_frame = tk.Frame(root)

# Création des frames contenant les métriques d'évaluation
metrics_frame = tk.Frame(root)
dt_metrics_frame = tk.Frame(metrics_frame)
knn_metrics_frame = tk.Frame(metrics_frame)
reg_metrics_frame = tk.Frame(metrics_frame)

# Création des frames contenant les résultats
result_frame = tk.Frame(root)
dt_result_frame = tk.Frame(result_frame)
knn_result_frame = tk.Frame(result_frame)
reg_result_frame = tk.Frame(result_frame)

def init():
    data = pd.read_csv('Titanic-Dataset.csv')
    df = data.copy()
    df = data.copy()

    df.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)

    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], prefix=["Sex", "Embarked", "Pclass"])

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    testdf = df[df['Age'].isnull() == True]
    traindf = df[df['Age'].isnull() == False]
    y = traindf['Age']
    traindf.drop("Age", axis=1, inplace=True)
    lr.fit(traindf, y)
    testdf.drop("Age", axis=1, inplace=True)
    pred = lr.predict(testdf)
    testdf['Age'] = pred
    traindf['Age'] = y

    df = testdf.append(traindf, ignore_index=False)

    # notre régression à prédit des individus avec un age en dessous de 0 donc on changes la valeurs pour ces individus à 0
    df['Age'] = np.where(df['Age'] <= 0, 0, df['Age'])

    df['Categ_Age'] = pd.cut(df['Age'], bins=[0, 12, 17, 59, 200], labels=['Enfant', 'Ado', 'Adulte', 'Ainé'])
    df['Categ_Prix'] = pd.cut(df['Fare'], bins=[0, df['Fare'].median(), 120, 1000],
                              labels=['Bon_marché', 'Prix_moyen', 'Prix_élévé'])

    df.drop(["Fare", "Age", "PassengerId"], axis=1, inplace=True)

    df = pd.get_dummies(df, columns=["Categ_Age", "Categ_Prix"], prefix=["Age", "Prix"])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.5,
                                                        random_state=42)

    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()

    # Définition des paramètres à tester pour un arbre de decisions
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(dt, param_grid, cv=5)

    # Entraînement du modèle
    grid_search.fit(X_train, y_train)
    best_model1 = grid_search.best_estimator_

    # Prédiction sur l'ensemble de test
    y_predDT = best_model1.predict(X_test)
    pred1 = y_predDT

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    print(f"Taux d'accuracy de l'arbre de décision est de : {accuracy_score(y_test, y_predDT)}")
    print(f"taux de precision de l'arbre de décision est de : {precision_score(y_test, y_predDT)}")
    print(f" Taux de recall de l'arbre de décision est de : {recall_score(y_test, y_predDT)}")
    print(f"F1 Score de l'arbre de décision est de: {f1_score(y_test, y_predDT)}")

    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=1)

    # Définition des paramètres à tester pour un KNN
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15]}
    grid_search = GridSearchCV(KNN, param_grid, cv=5)

    # Entraînement du modèle
    grid_search.fit(X_train, y_train)
    best_model2 = grid_search.best_estimator_

    # Prédiction sur l'ensemble de test
    y_predKN = best_model2.predict(X_test)
    pred2 = y_predKN

    print(f"Taux d'accuracy des K plus proches voisins est de : {accuracy_score(y_test, y_predKN)}")
    print(f"taux de precision des K plus proches voisins est de : {precision_score(y_test, y_predKN)}")
    print(f" Taux de recall des K plus proches voisins est de : {recall_score(y_test, y_predKN)}")
    print(f"F1 Score de des K plus proches voisins est de: {f1_score(y_test, y_predKN)}")

    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression()

    # Définition des paramètres à tester pour un régression logistique
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(reg, param_grid, cv=5)

    # Entraînement du modèle
    grid_search.fit(X_train, y_train)
    best_model3 = grid_search.best_estimator_

    # Prédiction sur l'ensemble de test
    y_pred = best_model3.predict(X_test)
    pred3 = np.where(y_pred < 0.5, 0, 1)

    print(f"Taux d'accuracy de la régréssion linéaire est de : {accuracy_score(y_test, pred3)}")
    print(f"taux de precision de la régréssion linéaire est de : {precision_score(y_test, pred3)}")
    print(f" Taux de recall de la régréssion linéaire est de : {recall_score(y_test, pred3)}")
    print(f"F1 Score de la régréssion linéaire est de: {f1_score(y_test, pred3)}")

    return pred1, pred2, pred3, KNN, dt, reg, y_test, best_model1, best_model2, best_model3

def prep(df):

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    testdf = df[df['Age'].isnull() == True]
    traindf = df[df['Age'].isnull() == False]
    y = traindf['Age']
    traindf.drop("Age", axis=1, inplace=True)
    lr.fit(traindf, y)
    testdf.drop("Age", axis=1, inplace=True)
    pred = lr.predict(testdf)
    testdf['Age'] = pred
    traindf['Age'] = y

    df = testdf.append(traindf, ignore_index=False)

    # notre régression à prédit des individus avec un age en dessous de 0 donc on changes la valeurs pour ces individus à 0
    df['Age'] = np.where(df['Age'] <= 0, 0, df['Age'])

    df['Categ_Age'] = pd.cut(df['Age'], bins=[0, 12, 17, 59, 200], labels=['Enfant', 'Ado', 'Adulte', 'Ainé'])
    df['Categ_Prix'] = pd.cut(df['Fare'], bins=[0, df['Fare'].median(), 120, 1000],
                              labels=['Bon_marché', 'Prix_moyen', 'Prix_élévé'])

    df.drop(["Fare", "Age", "PassengerId"], axis=1, inplace=True)

    df = pd.get_dummies(df, columns=["Categ_Age", "Categ_Prix"], prefix=["Age", "Prix"])

    return df
def valider():

    pred1, pred2, pred3, KNN, dt, reg, y_test, best_model1, best_model2, best_model3 = init()
    print(pred1, pred2, pred3, KNN, dt, reg)
    lastname = input_lastname.get()
    firstname = input_firstname.get()
    pclass = P_CLASSES[input_pclass.get()]
    sex = input_sex.get()
    age = input_age.get()
    sibsp = input_sibsp.get()
    parch = input_parch.get()
    fare = input_fare.get()
    embarked = BOARDING_CITIES[input_embarked.get()]

    pclass_1 = 1 if P_CLASSES[input_pclass.get()] == 1 else 0
    pclass_2 = 1 if P_CLASSES[input_pclass.get()] == 2 else 0
    pclass_3 = 1 if P_CLASSES[input_pclass.get()] == 3 else 0
    sex_female = 1 if sex == "female" else 0
    sex_male = 1 if sex == "male" else 0
    age_enfant = 1 if 0 <= int(age) <= 10 else 0
    age_ado = 1 if 10 < int(age) <= 20 else 0
    age_adulte = 1 if 20 < int(age) <= 50 else 0
    age_aine = 1 if 50 < int(age) else 0
    prix_bon_marche = 1 if 0 <= int(fare) <= 10 else 0
    prix_prix_moyen = 1 if 10 < int(fare) <= 20 else 0
    prix_prix_eleve = 1 if 20 < int(fare) else 0
    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    d = {'PClass_1': pclass_1,
         'PClass_2': pclass_2,
         'PClass_3': pclass_3,
         'Sex_female': sex_female,
         'Sex_male': sex_male,
         'Age_Enfant': age_enfant,
         'Age_Ado': age_ado,
         'Age_Adulte': age_adulte,
         'Age_Ainé': age_aine,
         'SibSp': sibsp,
         'Parch': parch,
         'Prix_Bon_marché': prix_bon_marche,
         'Prix_Prix_moyen': prix_prix_moyen,
         'Prix_Prix_élévé': prix_prix_eleve,
         'Embarked_C': embarked_C,
         'Embarked_Q': embarked_Q,
         'Embarked_S': embarked_S}

    array2d = np.array([[d['PClass_1'], d['PClass_2'], d['PClass_3'],
                         d['Sex_female'], d['Sex_male'], d['Age_Enfant'], d['Age_Ado'], d['Age_Adulte'], d['Age_Ainé'],
                         d['SibSp'], d['Parch'], d['Prix_Bon_marché'], d['Prix_Prix_moyen'], d['Prix_Prix_élévé'],
                         d['Embarked_C'], d['Embarked_Q'], d['Embarked_S']]])

    y_pred1 = best_model1.predict(array2d)
    print(str(y_pred1))

    y_pred2 = best_model2.predict(array2d)
    print(str(y_pred2))

    y_pred3 = best_model3.predict(array2d)
    y_pred3f = np.where(y_pred3 < 0.5, 0, 1)
    print(str(y_pred3f))

    label_dt_metrics.config(text=f"Taux d'accuracy : {accuracy_score(y_test, pred1)} \n Taux de précision : " \
                      f"{precision_score(y_test, pred1)}\n Taux de recall : {recall_score(y_test, pred1)}\n F1-score : " \
                      f"{f1_score(y_test, pred1)}")

    label_knn_metrics.config(text=f"Taux d'accuracy : {accuracy_score(y_test, pred2)}\n Taux de précision : " \
                      f"{precision_score(y_test, pred2)}\n Taux de recall : {recall_score(y_test, pred2)}\n F1-score : " \
                      f"{f1_score(y_test, pred2)}")

    label_reg_metrics.config(text=f"Taux d'accuracy : {accuracy_score(y_test, pred3)}\n Taux de précision : " \
                      f"{precision_score(y_test, pred3)}\n Taux de recall : {recall_score(y_test, pred3)}\n F1-score : " \
                      f"{f1_score(y_test, pred3)}")

    result = "survivre" if y_pred1 == 1 else "mourir"
    label_dt_result.config(text=f"D'après le DECISION TREE CLASSIFIER,\n{firstname} {lastname} devrait {result} (y_pred = {y_pred1})")
    result = "survivre" if y_pred2 == 1 else "mourir"
    label_knn_result.config(text=f"D'après le K NEIGHBORS CLASSIFIER,\n{firstname} {lastname} devrait {result} (y_pred = {y_pred2})")
    result = "survivre" if y_pred3f == 1 else "mourir"
    label_reg_result.config(text=f"D'après la Régréssion logistique,\n{firstname} {lastname} devrait {result} (y_pred = {y_pred3f})")

# -------------- H1 TITLE ---------------

title_label = tk.Label(root, text="Créez votre passager, et découvrez si il aurait survécu au naufrage du Titanic ?", font=("Times", 24, "bold"))
title_label.pack(pady=PAD_Y)

# -------------- FORM FRAME ---------------

# LABEL PRENOM
label_firstname = tk.Label(form_frame, text="Prénom : ")
label_firstname.grid(row=0, column=0, pady=PAD_Y, sticky="E")

# INPUT PRENOM
input_firstname = tk.Entry(form_frame)
input_firstname.insert(0, "Denis")
input_firstname.grid(row=0, column=1, sticky="W")

# LABEL NOM
label_lastname = tk.Label(form_frame, text="Nom : ")
label_lastname.grid(row=1, column=0, pady=PAD_Y, sticky="E")

# INPUT NOM
input_lastname = tk.Entry(form_frame)
input_lastname.insert(0, "Brognard")
input_lastname.grid(row=1, column=1, sticky="W")

# LABEL CLASSE
label_pclass = tk.Label(form_frame, text="Classe : ")
label_pclass.grid(row=2, column=0, pady=PAD_Y, sticky="E")

# INPUT CLASSE
input_pclass = ttk.Combobox(form_frame, values=list(P_CLASSES.keys()))
input_pclass.current(1)
input_pclass.grid(row=2, column=1, sticky="W")

# LABEL GENRE
label_sex = tk.Label(form_frame, text="Genre : ")
label_sex.grid(row=3, column=0, pady=PAD_Y, sticky="E")

# INPUT GENRE
input_sex = tk.StringVar()
input_sex.set("Homme")
bouton_homme = tk.Radiobutton(form_frame, text="Homme", variable=input_sex, value=1)
bouton_homme.grid(row=3, column=1)
bouton_femme = tk.Radiobutton(form_frame, text="Femme", variable=input_sex, value=0)
bouton_femme.grid(row=3, column=2)
bouton_femme.select()

# LABEL AGE
label_age = tk.Label(form_frame, text="Age : ")
label_age.grid(row=4, column=0, pady=PAD_Y, sticky="E")

# INPUT AGE
input_age = tk.Entry(form_frame)
input_age.insert(0, '25')
input_age.grid(row=4, column=1, sticky="W")

# LABEL SIBSP
label_sibsp = tk.Label(form_frame, text="Nombre de frères et sœurs / conjoints à bord : ")
label_sibsp.grid(row=5, column=0, pady=PAD_Y, sticky="E")

# INPUT SIBSP
input_sibsp = tk.Scale(form_frame, from_=0, to=10, orient="horizontal")
input_sibsp.set(2)
input_sibsp.grid(row=5, column=1, sticky="W")

# LABEL PARCH
label_parch = tk.Label(form_frame, text="Nombre de parents / enfants à bord : ")
label_parch.grid(row=6, column=0, pady=PAD_Y, sticky="E")

# INPUT PARCH
input_parch = tk.Scale(form_frame, from_=0, to=10, orient="horizontal")
input_parch.set(1)
input_parch.grid(row=6, column=1, sticky="W")

# LABEL TARIF
label_fare = tk.Label(form_frame, text="Tarif du billet : ")
label_fare.grid(row=7, column=0, pady=PAD_Y, sticky="E")

# INPUT TARIF
input_fare = tk.Entry(form_frame)
input_fare.insert(0, '10')
input_fare.grid(row=7, column=1, sticky="W")

# LABEL EMBARKED
label_embarked = tk.Label(form_frame, text="Port d'embarquement : ")
label_embarked.grid(row=8, column=0, pady=PAD_Y, sticky="E")

# INPUT EMBARKED
input_embarked = ttk.Combobox(form_frame, values=list(BOARDING_CITIES.keys()))
input_embarked.current(1)
input_embarked.grid(row=8, column=1, sticky="W")

form_frame.pack(pady=PAD_Y*2)

# -------------- VALIDATION BUTTON ---------------

bouton_valider = tk.Button(root, text="C'EST PARTI !", command=valider)
bouton_valider.pack(pady=PAD_Y)


root_separator_1 = ttk.Separator(root, orient='horizontal')
root_separator_1.pack(fill='x', pady=PAD_Y)

# -------------- METRICS FRAME ---------------

label_metrics = tk.Label(root, text="Métriques d'évaluation : ")
label_metrics.pack(padx=(PAD_X, 0), pady=PAD_Y, anchor='w')

title_dt_metrics = tk.Label(metrics_frame, text="DECISION TREE CLASSIFIER", font=("TkDefaultFont", 10, "bold"), anchor='center')
title_dt_metrics.grid(row=0, column=0)

label_dt_metrics = tk.Label(metrics_frame, text=TEXT_DT_METRICS)
label_dt_metrics.grid(row=1, column=0, padx=(0, PAD_X*10))

metrics_separator_1 = ttk.Separator(metrics_frame, orient='vertical')
metrics_separator_1.grid(row=0, column=1)

title_knn_metrics = tk.Label(metrics_frame, text="K NEIGHBORS CLASSIFIER", font=("TkDefaultFont", 10, "bold"), anchor='center')
title_knn_metrics.grid(row=0, column=2)

label_knn_metrics = tk.Label(metrics_frame, text=TEXT_KNN_METRICS)
label_knn_metrics.grid(row=1, column=2)

label_knn_metrics = tk.Label(metrics_frame, text="")
label_knn_metrics.grid(row=2, column=2)

metrics_separator_2 = ttk.Separator(metrics_frame, orient='vertical')
metrics_separator_2.grid(row=0, column=3)

title_reg_metrics = tk.Label(metrics_frame, text="REGRESSION LOGISTIQUE", font=("TkDefaultFont", 10, "bold"), anchor='center')
title_reg_metrics.grid(row=0, column=4)

label_reg_metrics = tk.Label(metrics_frame, text=TEXT_REG_METRICS)
label_reg_metrics.grid(row=1, column=4, padx=(PAD_X*10, 0))

metrics_frame.pack(pady=PAD_Y*2)


root_separator_2 = ttk.Separator(root, orient='horizontal')
root_separator_2.pack(fill='x', pady=PAD_Y)

# -------------- RESULT FRAME ---------------

label_result = tk.Label(root, text="Résultats : ")
label_result.pack(padx=(PAD_X, 0), pady=PAD_Y, anchor='w')

label_dt_result = tk.Label(result_frame, text=TEXT_DT_RESULT)
label_dt_result.grid(row=0, column=0, padx=(0, PAD_X*5))

result_separator_1 = ttk.Separator(result_frame, orient='vertical')
result_separator_1.grid(row=0, column=1)

label_knn_result = tk.Label(result_frame, text=TEXT_KNN_RESULT)
label_knn_result.grid(row=0, column=2)

result_separator_2 = ttk.Separator(result_frame, orient='vertical')
result_separator_2.grid(row=0, column=3)

label_reg_result = tk.Label(result_frame, text=TEXT_REG_RESULT)
label_reg_result.grid(row=0, column=40, padx=(PAD_X*5, 0))

result_frame.pack(pady=PAD_Y*2)

for child in metrics_frame.winfo_children():
        if isinstance(child, tk.Label):
            child.config(font=(child['font'], 12))

for child in result_frame.winfo_children():
        if isinstance(child, tk.Label):
            current_font = child['font']
            new_font = (current_font[0], 14) # Remplacer 16 par la taille de police souhaitée
            child.config(font=new_font)

# Lancement de la boucle principale de la fenêtre
root.mainloop()
