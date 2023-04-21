import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import csv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

TEXT_DT_METRICS = "TEXT DT METRICS"
TEXT_KNN_METRICS = "TEXT KNN METRICS"
TEXT_REG_METRICS = "TEXT REG METRICS"

TEXT_DT_RESULT = "TEXT DT RESULT"
TEXT_KNN_RESULT = "TEXT KNN RESULT"
TEXT_REG_RESULT = "TEXT REG RESULT"

PAD_X = 10
PAD_Y = 10
P_CLASSES = {"1ère classe": 1, "2ème classe": 2, "3ème classe": 3}
BOARDING_CITIES = {"Cherbourg": 'C', "Queenstown": 'Q', "Southampton": 'S'}

# Création de la frame principale
root = tk.Tk()
root.title("Formulaire Titanic")
root.geometry("1000x800")

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
    # Suppression des variables non pertinentes
    df = df.drop(["Name", "Ticket", 'Cabin', 'PassengerId'], axis=1)

    # numerisation de la colonne Sex
    df['Sex'] = df['Sex'].replace('male', 1)
    df['Sex'] = df['Sex'].replace('female', 0)

    # binarisation de Embarked
    df_embarked = pd.get_dummies(df["Embarked"])
    df = pd.concat((df, df_embarked), axis=1)
    df = df.drop(["Embarked"], axis=1)

    # Arrondir les valeurs de Fare

    df["Fare"] = round(df["Fare"], 2)

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

    dff = df.copy()
    # les nombreuse valeurs aberrantes pour Fare peuvent s'expliquer du fait que ce sont à la base des valeurs de prix.
    # et que les données tirées d'un prix ne sont pas issue de la loi normale.
    # souvent pour réduire le nombre d'outlier on peut faire un log(variable)
    dff["Fare"] = np.log(dff["Fare"])
    # ceux qui ont payé 0$ ont donc comme resultat -infinity ce qui est un problème pour la suite, je remplace donc ces valeurs par 0
    dff.replace([np.inf, -np.inf], 0, inplace=True)

    # centrer et reduire
    from sklearn.preprocessing import StandardScaler
    y = dff["Survived"]
    dff = dff.drop(["Survived"], axis=1)
    Xs = StandardScaler().fit_transform(dff)
    pd.options.display.float_format = '{:.2f}'.format
    X = pd.DataFrame(Xs, columns=dff.columns)
    X["Survived"] = y

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X.drop('Survived', axis=1), X['Survived'], test_size=0.5,
                                                        random_state=42)

    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()

    print(X_train, y_train)
    dt.fit(X_train, y_train)
    y_predDT = dt.predict(X_test)
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
    KNN = KNeighborsClassifier(n_neighbors=5)

    KNN.fit(X_train, y_train)
    y_predKN = KNN.predict(X_test)
    pred2 = y_predKN

    print(f"Taux d'accuracy des K plus proches voisins est de : {accuracy_score(y_test, y_predKN)}")
    print(f"taux de precision des K plus proches voisins est de : {precision_score(y_test, y_predKN)}")
    print(f" Taux de recall des K plus proches voisins est de : {recall_score(y_test, y_predKN)}")
    print(f"F1 Score de des K plus proches voisins est de: {f1_score(y_test, y_predKN)}")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    reg = LinearRegression()

    reg.fit(X_train, y_train)
    y_predLR = reg.predict(X_test)
    pred3 = np.where(y_predLR < 0.5, 0, 1)

    print(f"Taux d'accuracy de la régréssion linéaire est de : {accuracy_score(y_test, pred3)}")
    print(f"taux de precision de la régréssion linéaire est de : {precision_score(y_test, pred3)}")
    print(f" Taux de recall de la régréssion linéaire est de : {recall_score(y_test, pred3)}")
    print(f"F1 Score de la régréssion linéaire est de: {f1_score(y_test, pred3)}")


    return pred1, pred2, pred3, KNN, dt, reg, y_test

def valider():

    pred1, pred2, pred3, KNN, dt, reg, y_test = init()
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

    if embarked == 'C':
        cher = 1
    else:
        cher = 0

    if embarked == 'Q':
        queen = 1
    else:
        queen = 0

    if embarked == 'S':
        south = 1
    else:
        south = 0

    d = {'pclass': pclass, 'sex': sex, 'age': age, 'sibsp': sibsp, 'parch': parch, 'fare': fare, 'C': cher, 'Q': queen, 'S': south}

    array2d = np.array([[d['pclass'], d['sex'], d['age'], d['sibsp'], d['parch'], d['fare'], d['C'], d['Q'], d['S']]])

    y_pred1 = dt.predict(array2d)
    print(str(y_pred1))

    y_pred2 = KNN.predict(array2d)
    print(str(y_pred2))

    y_pred3 = reg.predict(array2d)
    y_pred3f = np.where(y_pred3 < 0.5, 0, 1)
    print(str(y_pred3f))

    global TEXT_DT_METRICS
    global TEXT_KNN_METRICS
    global TEXT_REG_METRICS

    label_dt_metrics.config(text=f"Taux d'accuracy : {accuracy_score(y_test, pred1)} \n Taux de précision : " \
                      f"{precision_score(y_test, pred1)}\n Taux de recall : {recall_score(y_test, pred1)}\n F1-score : " \
                      f"{f1_score(y_test, pred1)}")

    label_knn_metrics.config(text=f"Taux d'accuracy : {accuracy_score(y_test, pred2)}\n Taux de précision : " \
                      f"{precision_score(y_test, pred2)}\n Taux de recall : {recall_score(y_test, pred2)}\n F1-score : " \
                      f"{f1_score(y_test, pred2)}")

    label_reg_metrics.config(text=f"Taux d'accuracy : {accuracy_score(y_test, pred3)}\n Taux de précision : " \
                      f"{precision_score(y_test, pred3)}\n Taux de recall : {recall_score(y_test, pred3)}\n F1-score : " \
                      f"{f1_score(y_test, pred3)}")

    label_dt_result.config(text=f"{y_pred1}")
    label_knn_result.config(text=f"{y_pred2}")
    label_reg_result.config(text=f"{y_pred3f}")


# -------------- H1 TITLE ---------------

title_label = tk.Label(root, text="Auriez-vous survécu sur le Titanic ?", font=("Times", 24, "bold"))
title_label.pack(pady=PAD_Y)

# -------------- FORM FRAME ---------------

# LABEL PRENOM
label_firstname = tk.Label(form_frame, text="Prénom : ")
label_firstname.grid(row=0, column=0, pady=PAD_Y, sticky="E")

# INPUT PRENOM
input_firstname = tk.Entry(form_frame)
input_firstname.grid(row=0, column=1, sticky="W")

# LABEL NOM
label_lastname = tk.Label(form_frame, text="Nom : ")
label_lastname.grid(row=1, column=0, pady=PAD_Y, sticky="E")

# INPUT NOM
input_lastname = tk.Entry(form_frame)
input_lastname.grid(row=1, column=1, sticky="W")

# LABEL CLASSE
label_pclass = tk.Label(form_frame, text="Classe : ")
label_pclass.grid(row=2, column=0, pady=PAD_Y, sticky="E")

# INPUT CLASSE
input_pclass = ttk.Combobox(form_frame, values=list(P_CLASSES.keys()))
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

# LABEL AGE
label_age = tk.Label(form_frame, text="Age : ")
label_age.grid(row=4, column=0, pady=PAD_Y, sticky="E")

# INPUT AGE
input_age = tk.Entry(form_frame)
input_age.grid(row=4, column=1, sticky="W")

# LABEL SIBSP
label_sibsp = tk.Label(form_frame, text="Nombre de frères et sœurs / conjoints à bord : ")
label_sibsp.grid(row=5, column=0, pady=PAD_Y, sticky="E")

# INPUT SIBSP
input_sibsp = tk.Scale(form_frame, from_=0, to=10, orient="horizontal")
input_sibsp.grid(row=5, column=1, sticky="W")

# LABEL PARCH
label_parch = tk.Label(form_frame, text="Nombre de parents / enfants à bord : ")
label_parch.grid(row=6, column=0, pady=PAD_Y, sticky="E")

# INPUT PARCH
input_parch = tk.Scale(form_frame, from_=0, to=10, orient="horizontal")
input_parch.grid(row=6, column=1, sticky="W")

# LABEL TARIF
label_fare = tk.Label(form_frame, text="Tarif du billet : ")
label_fare.grid(row=7, column=0, pady=PAD_Y, sticky="E")

# INPUT TARIF
input_fare = tk.Entry(form_frame)
input_fare.grid(row=7, column=1, sticky="W")

# LABEL EMBARKED
label_embarked = tk.Label(form_frame, text="Port d'embarquement : ")
label_embarked.grid(row=8, column=0, pady=PAD_Y, sticky="E")

# INPUT EMBARKED
input_embarked = ttk.Combobox(form_frame, values=list(BOARDING_CITIES.keys()))
input_embarked.grid(row=8, column=1, sticky="W")

form_frame.pack(pady=PAD_Y*2)

# -------------- VALIDATION BUTTON ---------------

bouton_valider = tk.Button(root, text="Valider", command=valider)
bouton_valider.pack(pady=PAD_Y)


root_separator_1 = ttk.Separator(root, orient='horizontal')
root_separator_1.pack(fill='x', pady=PAD_Y)

# -------------- METRICS FRAME ---------------

label_metrics = tk.Label(root, text="Métriques d'évaluation : ")
label_metrics.pack(padx=(PAD_X, 0), pady=PAD_Y, anchor='w')

title_dt_metrics = tk.Label(metrics_frame, text="DECISION TREE CLASSIFIER")
title_dt_metrics.grid(row=0, column=0, padx=(0, PAD_X*5))

label_dt_metrics = tk.Label(metrics_frame, text=TEXT_DT_METRICS)
label_dt_metrics.grid(row=1, column=0, padx=(0, PAD_X*5))

metrics_separator_1 = ttk.Separator(metrics_frame, orient='vertical')
metrics_separator_1.grid(row=0, column=1)

title_knn_metrics = tk.Label(metrics_frame, text="K NEIGHBORS CLASSIFIER")
title_knn_metrics.grid(row=0, column=2, padx=(0, PAD_X*5))

label_knn_metrics = tk.Label(metrics_frame, text=TEXT_KNN_METRICS)
label_knn_metrics.grid(row=1, column=2)

metrics_separator_2 = ttk.Separator(metrics_frame, orient='vertical')
metrics_separator_2.grid(row=0, column=3)

title_reg_metrics = tk.Label(metrics_frame, text="LINEAR REGRETION")
title_reg_metrics.grid(row=0, column=4, padx=(0, PAD_X*5))

label_reg_metrics = tk.Label(metrics_frame, text=TEXT_REG_METRICS)
label_reg_metrics.grid(row=1, column=4, padx=(PAD_X*5, 0))

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

# Lancement de la boucle principale de la fenêtre
root.mainloop()
