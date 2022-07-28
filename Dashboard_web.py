import joblib
import streamlit as st
import pandas as pd
import datetime
import xgboost

st.write('''
# APPLICATION DE PREDICTION
Cette application sert à predire les prix de vente de biens immobiliers:Maison et Appartement
Auteurs: ATOUKOUVI & DROH, étudiants AS_1
''')


st.sidebar.header("les paramètres d'entrés")

def user_input():
    date=st.sidebar.date_input('date',datetime.date.today())
    adresse_code_voie=st.sidebar.number_input("l'adresse code voie encodée")
    code_postal = st.sidebar.number_input("entrer le code postal")
    nom_commune = st.sidebar.number_input('nom de la commune encodé)')
    type_local_Appartement = st.sidebar.number_input('l immobilier est-il un appartement ? 1 si oui, 0 sinon')
    type_local_Maison = st.sidebar.number_input('l immobilier est-il une maison ? 1 si oui, 0 sinon')
    surface_reelle_bati=st.sidebar.number_input('surface relle batie')
    surface_terrain=st.sidebar.number_input('surface du terrain')
    nombre_pieces_principales=st.sidebar.number_input('nombre de pièces principales')
    longitude=st.sidebar.number_input('la longitude')
    latitude=st.sidebar.number_input('la latitude')
    departement =st.sidebar.number_input('le departement (encodé)')
    year=st.sidebar.number_input("l'année")
    data={
          'type_local_Appartement': type_local_Appartement,
          'type_local_Maison': type_local_Maison,
          'adresse_code_voie':adresse_code_voie,
          'code_postal': code_postal,
          'nom_commune':nom_commune,
          'surface_reelle_bati':surface_reelle_bati,
          'surface_terrain':surface_terrain,
          'nombre_pieces_principales':nombre_pieces_principales,
          'longitude':longitude,
          'latitude':latitude,
          'departement':departement,
          'year': year
    }
    parametre = pd.DataFrame(data,index=[0])
    return parametre
kl=user_input()
st.subheader("on veut estimer le prix des biens dont les caracteristiques sont les suivants:")
st.write(kl)


st.subheader("le prix estimé du bien est:")

Lgbm= joblib.load('C:\\Users\\DELL\\OneDrive - ENSEA\Bureau\\GEEK CHALLENGE\\Lgbm.sav')
Cbst = joblib.load('C:\\Users\\DELL\\OneDrive - ENSEA\Bureau\\GEEK CHALLENGE\\Cbst.sav')
#Xgbst = xgboost.Booster().load_model('C:\\Users\\DELL\\OneDrive - ENSEA\Bureau\\GEEK CHALLENGE\\Xgbst.sav')


def model_predict(data):
    return 0.6*Lgbm.predict(data)+ 0.4*Cbst.predict(data)

prediction = model_predict(kl)

st.write(prediction)

