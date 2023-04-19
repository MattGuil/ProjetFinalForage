import tkinter as tk
from tkinter import ttk

root = tk.Tk()

root.title("Formulaire Titanic")
root.geometry("1000x800")

def on_entry_click(event):
    if input_cabin.get() == "Ex: A-Z 0-999":
        input_cabin.delete(0, "end")
        input_cabin.config(fg = 'black')

def on_focusout(event):
    if input_cabin.get() == "":
        input_cabin.insert(0, "Ex: A-Z 0-999")
        input_cabin.config(fg = 'grey')

def valider():
    lastname = input_lastname.get()
    firstname = input_firstname.get()
    pclass = input_pclass.get()
    sex = input_sex.get()
    age = input_age.get()
    sibsp = input_sibsp.get()
    parch = input_parch.get()
    fare = input_fare.get()
    cabin = input_cabin.get()
    embarked = input_embarked.get()

    print("Identité : ", lastname + " " + firstname)
    print("Classe :", pclass)
    print("Genre :", sex)
    print("Age :", age)
    print("Nombre de frères et sœurs / conjoints à bord :", sibsp)
    print("Nombre de parents / enfants à bord :", parch)
    print("Tarif du billet :", fare)
    print("Cabine :", cabin)
    print("Port d'embarquement :", embarked)

# Ajout d'un label pour le champ "Prénom"
label_firstname = tk.Label(root, text="Prénom : ")
label_firstname.pack()

# Ajout d'un champ de saisie pour le prénom
input_firstname = tk.Entry(root)
input_firstname.pack()

# Ajout d'un label pour le champ "Nom"
label_lastname = tk.Label(root, text="Nom : ")
label_lastname.pack()

# Ajout d'un champ de saisie pour le nom
input_lastname = tk.Entry(root)
input_lastname.pack()

# Ajout d'un label pour le champ "Pclass"
label_pclass = tk.Label(root, text="Classe : ")
label_pclass.pack()

# Ajout d'une liste déroulante pour la classe de billets
input_pclass = ttk.Combobox(root, values=["1er classe", "2ème classe", "3ème classe"])
input_pclass.pack()

# Ajout d'un label pour le champ "Sex"
label_sex = tk.Label(root, text="Genre : ")
label_sex.pack()

# Ajout de boutons radio pour le genre du passager
input_sex = tk.StringVar()
input_sex.set("Homme")
bouton_homme = tk.Radiobutton(root, text="Homme", variable=input_sex, value="Homme")
bouton_homme.pack()
bouton_femme = tk.Radiobutton(root, text="Femme", variable=input_sex, value="Femme")
bouton_femme.pack()

# Ajout d'un label pour le champ "Age"
label_age = tk.Label(root, text="Age : ")
label_age.pack()

# Ajout d'un champ de saisie pour l'âge
input_age = tk.Entry(root)
input_age.pack()

# Ajout d'un label pour le champ "SibSp"
label_sibsp = tk.Label(root, text="Nombre de frères et sœurs / conjoints à bord : ")
label_sibsp.pack()

# Ajout d'un curseur pour le nombre de frères et soeurs / conjoints à bord
input_sibsp = tk.Scale(root, from_=0, to=10, orient="horizontal")
input_sibsp.pack()

# Ajout d'un label pour le champ "Parch"
label_parch = tk.Label(root, text="Nombre de parents / enfants à bord : ")
label_parch.pack()

# Ajout d'un curseur pour le nombre de parents / enfants à bord
input_parch = tk.Scale(root, from_=0, to=10, orient="horizontal")
input_parch.pack()

# Ajout d'un label pour le champ "Fare"
label_fare = tk.Label(root, text="Tarif du billet : ")
label_fare.pack()

# Ajout d'un champ de saisie pour le tarif passager
input_fare = tk.Entry(root)
input_fare.pack()

# Ajout d'un label pour le champ "Cabin"
label_cabin = tk.Label(root, text="Cabine : ")
label_cabin.pack()

# Ajout d'un champ de saisie pour la cabine du passager
input_cabin = tk.Entry(root)
input_cabin.insert(0, "Ex: A-Z 0-999")
input_cabin.pack()

input_cabin.bind('<FocusIn>', on_entry_click)
input_cabin.bind('<FocusOut>', on_focusout)
input_cabin.config(fg = 'grey')

# Ajout d'un label pour le champ "Embarked"
label_embarked = tk.Label(root, text="Port d'embarquement : ")
label_embarked.pack()

# Ajout d'une liste déroulante pour la ville d'embarquement
input_embarked = ttk.Combobox(root, values=["Cherbourg", "Queenstown", "Southampton"])
input_embarked.pack()

# Ajout d'un bouton pour valider le formulaire
bouton_valider = tk.Button(root, text="Valider", command=valider)
bouton_valider.pack()

separator = ttk.Separator(root, orient='horizontal')
separator.pack(fill='x')

label_metrics = tk.Label(root, text="Métriques d'évaluation : ")
label_metrics.pack()

separator = ttk.Separator(root, orient='horizontal')
separator.pack(fill='x')

label_results = tk.Label(root, text="Résultat : ")
label_results.pack()

# Lancement de la boucle principale de la fenêtre
root.mainloop()
