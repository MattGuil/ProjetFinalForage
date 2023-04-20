import tkinter as tk
from tkinter import ttk

PAD_X = 10
PAD_Y = 10

# Création de la frame principale
root = tk.Tk()
root.title("Formulaire Titanic")
root.geometry("1000x800")

# Création de la frame contenant formulaire
form_frame = tk.Frame(root)

def valider():
    lastname = input_lastname.get()
    firstname = input_firstname.get()
    pclass = input_pclass.get()
    sex = input_sex.get()
    age = input_age.get()
    sibsp = input_sibsp.get()
    parch = input_parch.get()
    fare = input_fare.get()
    embarked = input_embarked.get()

    print("Identité : ", lastname + " " + firstname)
    print("Classe :", pclass)
    print("Genre :", sex)
    print("Age :", age)
    print("Nombre de frères et sœurs / conjoints à bord :", sibsp)
    print("Nombre de parents / enfants à bord :", parch)
    print("Tarif du billet :", fare)
    print("Port d'embarquement :", embarked)

# Ajouter un grand titre
title_label = tk.Label(root, text="Auriez-vous survécu sur le Titanic ?", font=("Helvetica", 24, "bold"))
title_label.pack(pady=PAD_Y)

# Ajout d'un label pour le champ "Prénom"
label_firstname = tk.Label(form_frame, text="Prénom : ")
label_firstname.grid(row=0, column=0, pady=PAD_Y)

# Ajout d'un champ de saisie pour le prénom
input_firstname = tk.Entry(form_frame)
input_firstname.grid(row=0, column=1)

# Ajout d'un label pour le champ "Nom"
label_lastname = tk.Label(form_frame, text="Nom : ")
label_lastname.grid(row=1, column=0, pady=PAD_Y)

# Ajout d'un champ de saisie pour le nom
input_lastname = tk.Entry(form_frame)
input_lastname.grid(row=1, column=1)

# Ajout d'un label pour le champ "Pclass"
label_pclass = tk.Label(form_frame, text="Classe : ")
label_pclass.grid(row=2, column=0, pady=PAD_Y)

# Ajout d'une liste déroulante pour la classe de billets
input_pclass = ttk.Combobox(form_frame, values=["1er classe", "2ème classe", "3ème classe"])
input_pclass.grid(row=2, column=1)

# Ajout d'un label pour le champ "Sex"
label_sex = tk.Label(form_frame, text="Genre : ")
label_sex.grid(row=3, column=0, pady=PAD_Y)

# Ajout de boutons radio pour le genre du passager
input_sex = tk.StringVar()
input_sex.set("Homme")
bouton_homme = tk.Radiobutton(form_frame, text="Homme", variable=input_sex, value="Homme")
bouton_homme.grid(row=3, column=1)
bouton_femme = tk.Radiobutton(form_frame, text="Femme", variable=input_sex, value="Femme")
bouton_femme.grid(row=3, column=2)

# Ajout d'un label pour le champ "Age"
label_age = tk.Label(form_frame, text="Age : ")
label_age.grid(row=4, column=0, pady=PAD_Y)

# Ajout d'un champ de saisie pour l'âge
input_age = tk.Entry(form_frame)
input_age.grid(row=4, column=1)

# Ajout d'un label pour le champ "SibSp"
label_sibsp = tk.Label(form_frame, text="Nombre de frères et sœurs / conjoints à bord : ")
label_sibsp.grid(row=5, column=0, pady=PAD_Y)

# Ajout d'un curseur pour le nombre de frères et soeurs / conjoints à bord
input_sibsp = tk.Scale(form_frame, from_=0, to=10, orient="horizontal")
input_sibsp.grid(row=5, column=1)

# Ajout d'un label pour le champ "Parch"
label_parch = tk.Label(form_frame, text="Nombre de parents / enfants à bord : ")
label_parch.grid(row=6, column=0, pady=PAD_Y)

# Ajout d'un curseur pour le nombre de parents / enfants à bord
input_parch = tk.Scale(form_frame, from_=0, to=10, orient="horizontal")
input_parch.grid(row=6, column=1)

# Ajout d'un label pour le champ "Fare"
label_fare = tk.Label(form_frame, text="Tarif du billet : ")
label_fare.grid(row=7, column=0, pady=PAD_Y)

# Ajout d'un champ de saisie pour le tarif passager
input_fare = tk.Entry(form_frame)
input_fare.grid(row=7, column=1)

# Ajout d'un label pour le champ "Embarked"
label_embarked = tk.Label(form_frame, text="Port d'embarquement : ")
label_embarked.grid(row=8, column=0, pady=PAD_Y)

# Ajout d'une liste déroulante pour la ville d'embarquement
input_embarked = ttk.Combobox(form_frame, values=["Cherbourg", "Queenstown", "Southampton"])
input_embarked.grid(row=8, column=1)

form_frame.pack(pady=PAD_Y*2)

# Ajout d'un bouton pour valider le formulaire
bouton_valider = tk.Button(root, text="Valider", command=valider)
bouton_valider.pack(pady=PAD_Y)

separator1 = ttk.Separator(root, orient='horizontal')
separator1.pack(fill='x', pady=PAD_Y)

label_metrics = tk.Label(root, text="Métriques d'évaluation : ")
label_metrics.pack(padx=(PAD_X, 0), pady=PAD_Y, anchor='w')

separator2 = ttk.Separator(root, orient='horizontal')
separator2.pack(fill='x', pady=PAD_Y)

label_results = tk.Label(root, text="Résultat : ")
label_results.pack(padx=(PAD_X, 0), pady=PAD_Y, anchor='w')

# Lancement de la boucle principale de la fenêtre
root.mainloop()
