#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import xgboost
from xgboost import plot_importance
import sklearn
from sklearn.datasets import make_regression
import lightgbm as lgb
from catboost import CatBoostRegressor, FeaturesData, Pool
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


# # DATA PRE-PROCESSING

# ## Import of the train dataset

# In[20]:


df = pd.read_csv('C:\\Users\\DELL\\OneDrive - ENSEA\\Bureau\\GEEK CHALLENGE\\data\\train.csv')
df


# ## Checking for missing values

# In[154]:


df.isnull().sum()


# ### Nous remarquons qu'il n'y a pas de valeur manquante, alors nous continuons.

# ## Outliers checking

# ### cette analyse concerne les variables continues. Nous allons  réaliser une boite à moustache pour vérifier si ces variables contiennent des Outliers

# In[91]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharey=True, figsize= (20, 5))
sns.boxplot(x= df.surface_reelle_bati, ax= ax1)
sns.boxplot(x= df.surface_terrain, ax= ax2)
sns.boxplot(x= df.longitude, ax= ax3)
sns.boxplot(x= df.latitude, ax= ax4)
plt.show()


# ## Outliers cleaning

# ### Les boxplots réalisées, nous pouvons voir que seules les variables 'surface_relle_bati' et 'surface_terrain' contiennent des outliers. Nous allons tenter de les nettoyer.
# ### Nous optons pour le remplacement par la valeur du quantile d'ordre 0.95

# In[21]:


a = df['surface_reelle_bati'].quantile(.95)
b = df['surface_terrain'].quantile(.95)


# In[22]:


df["surface_reelle_bati"] = df["surface_reelle_bati"].apply(lambda x: a if x >a else x)
df["surface_terrain"] = df["surface_terrain"].apply(lambda x: b if x >b else x)


# # FEATURES ENGINEERING

# In[23]:


df = df.drop(['code_commune'], axis=1)


# In[51]:


df.info()


# In[52]:


features = df.drop(['id', 'estimation_prix', 'date_evaluation', 'type_local'], axis= 1)


# In[33]:


type_local = df.drop(['adresse_code_voie', 'code_postal', 'nom_commune', 'surface_reelle_bati', 'surface_terrain', 'nombre_pieces_principales', 'longitude', 'latitude', 'departement', 'year', 'id', 'estimation_prix', 'date_evaluation'], axis=1)
type_local = pd.get_dummies(data= type_local, columns= type_local.columns)
df_final = pd.concat([df.estimation_prix, type_local, features], axis=1)
df_final


# ## Split dataset in train / test / validation

# In[34]:


x_train, x_test, y_train, y_test = train_test_split(df_final.drop("estimation_prix", axis=1), df_final["estimation_prix"],
                                                    train_size=0.7, shuffle=True, random_state=42)


# In[35]:


test_set = pd.concat([y_test, x_test], axis= 1)
x_validation, x_test, y_validation, y_test =train_test_split(test_set.drop('estimation_prix', axis =1), test_set['estimation_prix'],
                                                            test_size=0.6, shuffle=True, random_state=42)


# # LINEAR CORRELATIONS ANALYSIS
# ### Cette analyse nous permettra de remarquer une dépendance (ou indépendance) linéaire entre les numerics features et le target, afin de savoir le modèle adéquat.

# In[36]:


(df_final).corr(method= 'pearson')['estimation_prix']


# ### Après analyse, nous remarquons à travers les chiffres ci-dessus qu'il n'y a pas de forte correlations linéaire entre les features et les targets.
# ### Du coup, nous allons nous orienter vers des algorithmes de regression outre la regression linéaire.

# # FIRST MODEL
# ## Xgboost

# In[40]:


Xgbst = xgboost.XGBRegressor(n_estimators= 700, seed = 100, learning_rate=0.2,max_deph= 70, gamma=1, reg_alpha=0.75, reg_gamma= 0.52, subsample=0.5)
Xgbst.fit(x_train, y_train)


# In[41]:


pred_1 = Xgbst.predict(x_test)
pred_1


# In[42]:


print('MAPE SCORE:', np.round(100*mape(y_test, pred_1),3),'%')


# ### Après entraînement du modèle, nous l'avons tester sur le test_set, ce qui nous donne une Mean Absolute Percentage Error de 26,45 % .

# # SECOND MODEL
# ## LighGbm

# In[37]:


Lgbm= lgb.LGBMRegressor(objective= 'regression',boostig_type = 'gbdt' ,metric= 'mean_absolute_error', n_estimators= 1000)
Lgbm.fit(x_train, y_train)


# In[38]:


pred_2 = Lgbm.predict(x_test)
pred_2


# In[39]:


print('MAPE SCORE:', np.round(100*mape(y_test, pred_2),4),'%')


# ### Après avoir entraîner notre modèle, nous l'avons tester sur le test_set, ce qui nous donne une Mean Absolute Percentage Error de 26, 71 %

# # THIRD MODEL
# ## Catboost

# In[43]:


Cbst = CatBoostRegressor()
Cbst.fit(x_train, y_train)


# In[44]:


pred_3 = Cbst.predict(x_test)
pred_3


# In[45]:


print('MAPE SCORE:', np.round(100*mape(y_test, pred_3),3),'%')


# ### Le Catboost nous donne un score de 27,1 %

# ## Mean of the three predictions

# ### Nous remarquons que nos 3 modèles ont à peu près la même perfomance. Pour ajuster les résultats, nous allons effectuer une moyenne pondérée des résultats de chaque modèle en effectant un poids élevé au modèles les plus perfomants.

# In[46]:


mean_preds= (0.5*pred_1)+(0.3*pred_2)+(0.2*pred_3)
mean_preds


# In[47]:


print('MAPE SCORE:', np.round(100*mape(mean_preds, y_test),3),'%')


# ### La moyenne des prédictions nous donne une MAPE de 22,4 % , ce qui représente une amélioration des perfomances de notre prédiction.
# ### Finalement, nous avons un modèle de prédiction qui, testé sur notre test_set nous donne une MAPE de 22,4  %

# # FINAL MODEL
# ## Weigh average of the three models

# In[48]:


def model_predict(data):
    return 0.5*Xgbst.predict(data)+ 0.3*Lgbm.predict(data)+ 0.2*Cbst.predict(data)


# # VALIDATION PHASE

# In[49]:


model_predict(x_validation)


# In[50]:


print('MAPE SCORE:', np.round(100*mape(model_predict(x_validation), y_validation),3),'%')


# # PREDICTION ON THE TEST DATASET

# In[106]:


test_data = pd.read_csv('C:\\Users\\DELL\\OneDrive - ENSEA\\Bureau\\GEEK CHALLENGE\\data\\test.csv')
id = test_data['id']
test_data= test_data.drop(['id'], axis=1)


# In[99]:


###fonctions traitement de la base de données
def treatment(data_frame):
    data_frame= data_frame.drop(['code_commune'], axis=1)
    a = data_frame['surface_reelle_bati'].quantile(.95)
    b = data_frame['surface_terrain'].quantile(.95)
    data_frame["surface_reelle_bati"] = data_frame["surface_reelle_bati"].apply(lambda x: a if x >a else x)
    data_frame["surface_terrain"] = data_frame["surface_terrain"].apply(lambda x: b if x >b else x)
    feat = data_frame.drop(['date_evaluation', 'type_local'], axis= 1)
    type_loc = data_frame.drop(['adresse_code_voie', 'code_postal', 'nom_commune', 'surface_reelle_bati', 'surface_terrain', 'nombre_pieces_principales', 'longitude', 'latitude', 'departement', 'year', 'date_evaluation'], axis=1)
    type_loc = pd.get_dummies(data= type_loc, columns= type_loc.columns)
    data_final = pd.concat([type_loc, feat], axis=1)
    return data_final


# In[107]:


test_data = treatment(test_data)


# In[108]:


PREDICTIONS = model_predict(test_data)
PREDICTIONS


# # SUBMISSION

# In[112]:


submission = pd.DataFrame({'id': id, 'estimation_prix': PREDICTIONS})
submission


# In[116]:


submission.to_csv('C:\\Users\\DELL\\OneDrive - ENSEA\Bureau\\GEEK CHALLENGE\\ATOUKOUVI_DROH.csv')

