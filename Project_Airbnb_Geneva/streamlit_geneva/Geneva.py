#Librerías
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import plotly_express as px
#import folium_static

import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

#to make the plotly graphs
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode

#text mining
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud





#Configuración de página, puede ser "centered" o "wide"

st.set_page_config(page_title="Airbnb Geneva", layout="wide",page_icon="🇨🇭")

#insercción de imagen de fondo


st.image(r'/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/streamlit_geneva/jetdeau.jpg')
st.text('referencia de la imagen de fondo: https://planetofhotels.com/guide/es/suiza/ginebra/geneva-water-fountains-jet-deau')
#insercción de texto, imágenes y datos introductorios

st.title("GINEBRA: Un destino muy versátil")



col1,col2 = st.columns(2)

with col1:
    st.write('**Suiza es un país que esta territorialmente estructurado en 26 cantones, los cuales se dividen en comunas. El cantón de Ginebra está compuesto por 45 comunas. Como este cantón es de los más pequeños de Suiza en comparación con los demás Airbnb considera el cantón completo como una ciudad y sus comunas como vecindarios.**', 
unsafe_allow_html=True)
    
with col2:
    st.image(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/streamlit_geneva/suizamap.gif", width = 500, caption=st.write("Mapa de los 26 cantones suizos ([link](https://planetofhotels.com/guide/es/suiza/ginebra/geneva-water-fountains-jet-deau))"))




st.write('**PREPROCESAMIENTO:**', unsafe_allow_html=True)
st.write('**0. Unión de la información por columnas que nos interesa de cada dataset (listings y listings_details)**', unsafe_allow_html=True)
st.write('**1. Eliminación de columnas inútiles por tener todos sus valores nulos como: grupo de vecindario o licencia.**', unsafe_allow_html=True)
st.write('**2. Cambio de tipo de variable para la velocidad de respuesta de objeto a numérica después de eliminar el símbolo del porcentaje.**', unsafe_allow_html=True)
st.write('**3. Los valores nulos de las valoraciones los sustituyo por la media de todos los valores de cada columna.**', unsafe_allow_html=True)
st.write('**4. Creación de nuevas columnas como legality, crime_rate and security_rate (a partir de los últimos datos publicados por el gobierno suizo (estadísticas de 2022): http://www.citypopulation.de/en/switzerland/geneve/, https://www.ge.ch/document/statistique-policiere-criminalite-2022-commune, https://www.swissinfo.ch/fre/societe/ins%C3%A9curit%C3%A9-en-suisse_la-criminalit%C3%A9-d-un-des-pays-les-plus-s%C3%BBrs-du-monde/46255492).**', unsafe_allow_html=True)
st.write('**5. Los valores atípicos de la columna precio se sustituyeron por la mediana para poder trabajar con esta variable.**', unsafe_allow_html=True)
    

col1,col2 = st.columns(2)

with col1:
    st.text('LISTINGS DATASET')
    listings = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/listings.csv")
    st.dataframe(listings.head(11))

with col2:
    st.text('LISTINGS_DETAILS DATASET')
    listings_details = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/listings_details.csv")
    target_columns = ["id","property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "instant_bookable", "neighborhood_overview", "host_identity_verified", "bathrooms_text", "bedrooms"]
    listings_ready = pd.merge(listings, listings_details[target_columns], on='id', how='left')
    st.dataframe(listings_ready.head(11))

listings_ready = listings_ready.drop(columns=['neighbourhood_group','license'])
listings_ready['price'].mask(listings_ready['price'] >278.5 , listings_ready['price'].median(), inplace=True)
listings_ready['host_response_rate'] = pd.to_numeric(listings_ready['host_response_rate'].str.strip('%'))
listings_ready['review_scores_location'].fillna(listings_ready['review_scores_location'].mean(), inplace=True)
listings_ready['review_scores_communication'].fillna(listings_ready['review_scores_communication'].mean(), inplace=True)
listings_ready['review_scores_cleanliness'].fillna(listings_ready['review_scores_cleanliness'].mean(), inplace=True)
listings_ready['review_scores_accuracy'].fillna(listings_ready['review_scores_accuracy'].mean(), inplace=True)
listings_ready['review_scores_checkin'].fillna(listings_ready['review_scores_checkin'].mean(), inplace=True)
listings_ready['host_is_superhost'] = listings_ready['host_is_superhost'].replace({"f": "Anfitrión estándar", "t": "Superanfitrión"})
dic_crime = {'Commune de Genève':2907, 'Versoix':55, 'Avully':9, 'Vernier':200, 'Plan-les-Ouates':104, 'Chêne-Bougeries':140, 'Grand-Saconnex':44, 'Carouge':324,
       'Genthod':16, 'Onex':100, 'Veyrier':90, 'Meyrin':206, 'Chêne-Bourg':133,
       'Vandoeuvres':7, 'Pregny-Chambésy':15, 'Lancy':290, 'Confignon':17, 'Cologny':95,
       'Collonge-Bellerive':36, 'Bellevue':24, 'Bardonnex':14, 'Thônex':169, 'Satigny':28,
       'Laconnex':2, 'Bernex':73, 'Troinex':14, 'Hermance':6, 'Anières':8, 'Puplinge':22,
       'Céligny':4, 'Gy':3, 'Russin':1, 'Soral':10, 'Dardagny':5, 'Corsier':1,
       'Meinier':10, 'Presinge':2, 'Perly-Certoux':26, 'Collex-Bossy':6, 'Jussy':6,
       'Chancy':11}
listings_ready['crime_vol'] =  listings_ready['neighbourhood'].map(dic_crime)

dic_pop = {'Commune de Genève':203757, 'Versoix':13332, 'Avully':1709, 'Vernier':36563, 'Plan-les-Ouates':12088, 'Chêne-Bougeries':13256, 'Grand-Saconnex':12603, 'Carouge':22160,
       'Genthod':2882, 'Onex':18765, 'Veyrier':11897, 'Meyrin':26507, 'Chêne-Bourg':8833,
       'Vandoeuvres':2852, 'Pregny-Chambésy':3991, 'Lancy':34645, 'Confignon':4594, 'Cologny':5971,
       'Collonge-Bellerive':8493, 'Bellevue':4071, 'Bardonnex':2530, 'Thônex':16113, 'Satigny':4449,
       'Laconnex':703, 'Bernex':10250, 'Troinex':2600, 'Hermance':1189, 'Anières':2417, 'Puplinge':2526,
       'Céligny':845, 'Gy':449, 'Russin':536, 'Soral':963, 'Dardagny':1800, 'Corsier':2265,
       'Meinier':2068, 'Presinge':737, 'Perly-Certoux':3139, 'Collex-Bossy':1727, 'Jussy':1193,
       'Chancy':1671}
listings_ready['pop_vol'] =  listings_ready['neighbourhood'].map(dic_pop)

listings_ready['crime_rate'] =  listings_ready['crime_vol']/listings_ready['pop_vol']*100000 

st.text('CONJUNTO DE DATOS PREPROCESADO')
st.dataframe(listings_ready.head(11))

