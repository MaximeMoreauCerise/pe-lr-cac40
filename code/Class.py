#https://mode.com/example-gallery/python_dataframe_styling/

#bibliothèque
import plotly.express as px
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.metrics import r2_score
# Fonction pour changer la couleur du texte
def couleur(valeur):
    if valeur < 0:
        return f'<font color="red">{valeur}</font>'
    else:
        return f'<font color="green">{valeur}</font>'


class portefeuille:
    def __init__(self, noms_titres, quantites, time_achat) -> None:
        self.noms_titres = noms_titres
        self.quantites = quantites
        self.time = time_achat
        #On importe les données
        self.data = []
        self.prix = []
        for i in self.noms_titres:
            path = "data-fr/Data_CAC40/dataframe_"+i+".csv"
            df = pd.read_csv(path)
            self.data.append(df)
            #print(i, "chargé")
        #On récupère le prix de fermeture des actions à l'instant t = time_achat
        self.recuperer_prix(self.time)
        #Création du dataframe de notre portefeuille
        self.complete_actions_df()
    

    def __str__(self) -> str:
        return f'{self.actions}'
    
    def get_noms_titres(self):
        return self.noms_titres

    def recuperer_prix(self, t):
        for i in self.data:
            self.prix.append(i['Close'][t])

            
    def get_data(self):
        return self.data
    
    def complete_actions_df(self):
        self.actions_temp = {'Prix achat à t = ' + str(self.time) :self.prix,
                        'Quantite': self.quantites}
        self.actions = pd.DataFrame(self.actions_temp, index=self.noms_titres)
        self.actions['Valeur des actions en $'] = self.actions.apply(lambda row: row['Prix achat à t = ' + str(self.time)] * row['Quantite'], axis=1)

    def add_predic(self, t, mod):
            predict = []
            for i in mod.models:
                predict.append(i.predict(np.array([t]).reshape(1,1))[0])
            self.actions['Évolution du prix en %'] = 100*(predict-self.actions['Prix achat à t = ' + str(self.time)])/self.actions['Prix achat à t = ' + str(self.time)]
            self.actions['Estimation du prix de unitaire à t = '+str(t)] = predict
            self.actions['Estimation de la valeur des actions à t = '+str(t)] = self.actions.apply(lambda row: row['Estimation du prix de unitaire à t = '+str(t)] * row['Quantite'], axis=1)
            #self.couleur_evolution()
            self.actions['Coefficient de determination'] = self.scores

    def afficher_resultats(self):
        return self.actions.head(len(self.actions))
    
    def couleur_evolution(self):
        self.actions['Évolution du prix en %'] = self.actions['Évolution du prix en %'].apply(lambda x: couleur(x))

    def get_r2_mean(self):
        return self.actions['Coefficient de determination'].mean()
    
    def get_r2_var(self):
        return self.actions['Coefficient de determination'].var()

    def set_scores(self, scores):
        self.scores = scores

class lr_models:
    #period = période d'apprentissage => inputs
    def __init__(self, portefeuille, period, learning_rate) -> None:
        self.mon_portefeuille = portefeuille 
        self.noms_titres = portefeuille.get_noms_titres()
        self.m = period
        self.learning_rate = learning_rate
        self.scores = []

    def create_models(self):
        self.models = []
        for i in self.mon_portefeuille.get_data():
            self.models.append(self.get_model(i))


    def get_model(self, df_titre):
        df = df_titre.head(self.m)
        self.Y = df['Close']
        self.X = np.arange(self.m).reshape(self.m,1)
        model = LinearRegression()
        model.fit(self.X,self.Y)
        self.scores.append(model.score(self.X,self.Y))
        self.mon_portefeuille.set_scores(self.scores)
        #print(model.score(self.X, self.Y))
        return model
