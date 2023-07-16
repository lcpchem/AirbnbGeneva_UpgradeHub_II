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
from streamlit_folium import st_folium
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

st.set_page_config(page_title="Consejos al Gobierno cantonal de Ginebra", layout="wide",page_icon="🇨🇭")

#Análisis general de los datos
st.title("ALOJAMIENTOS POTENCIALMENTE ILEGALES")

listings = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/listings.csv")
listings_details = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/listings_details.csv")
target_columns = ["id","property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "instant_bookable", "neighborhood_overview", "host_identity_verified", "bathrooms_text", "bedrooms"]
listings_ready = pd.merge(listings, listings_details[target_columns], on='id', how='left')
listings_ready = listings_ready.drop(columns=['neighbourhood_group','license'])
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


def legality(maximum_nights):
    if maximum_nights <= 90:
        return 'legal'
    else :
        return 'illegal'
    
entireplace = listings_ready.loc[listings_ready['room_type'] == 'Entire home/apt']
entireplace['legality'] = entireplace['maximum_nights'].apply(legality)


st.markdown("<h5 style=>Alojamientos con un máximo de noches para alojarse mayor al permitido por el gobierno (90 noches)</h5>", unsafe_allow_html=True)
st.write('Esta ley se aplica al alquiler temporal de alojamientos enteros')
feq4=entireplace[['legality','maximum_nights']].value_counts().reset_index()
fig0 = px.scatter(entireplace, x=feq4['maximum_nights'], y=feq4[0], color= feq4['legality'], labels={'y':'Número de alojamientos', 'x':'Número máximo de noches para alojarse'}, title='Número máximo de noches para alojarse en alojamientos enteros',width=1000, height=400)
st.plotly_chart(fig0, theme = 'streamlit', use_container_width=True)


st.markdown("<h3 style=>Identificación de empresas infiltradas en Airbnb</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h5 style=>Anfitriones con más de 3 habitaciones privadas</h5>", unsafe_allow_html=True)
    private = listings_ready[listings_ready['room_type'] == "Private room"]
    host_private = private.groupby(['host_id', 'host_name']).size().reset_index(name='private_rooms').sort_values(by=['private_rooms'], ascending=False)
    st.dataframe(host_private[host_private['private_rooms']>=3])
    

with col2:
    st.write('Ejemplo 1')
    erik = private[private['host_id']== 12766121]
    erik = erik[['name','host_id', 'host_name', 'latitude', 'longitude']]
    erik.index.name = "listing_id"
    st.dataframe(erik)

col1, col2 = st.columns(2)

with col1:
    st.write('Ejemplo 2')
    valentino = private[private['host_id']== 166151549]
    valentino = valentino[['name','host_id', 'host_name', 'latitude', 'longitude']]
    valentino.index.name = "listing_id"
    st.dataframe(valentino)

with col2:
    st.write('Ejemplo 3')
    marie = private[private['host_id']== 44666245]
    marie = marie[['name','host_id', 'host_name', 'latitude', 'longitude']]
    marie.index.name = "listing_id"
    st.dataframe(marie)




st.markdown("<h5 style=>Anfitriones con palabras clave incluidas en su nombre:</h5>", unsafe_allow_html=True)
st.markdown("<h6 style=>'hotel' y 'hôtel'</h6>", unsafe_allow_html=True)
st.dataframe(host_private[host_private['host_name'].str.contains('hotel|hôtel', case=False)])


col1, col2 = st.columns(2)

with col1:
    st.markdown("<h5 style=>Anfitriones con 10 o más alojamientos incluidos en airbnb:</h5>", unsafe_allow_html=True)
    freq1 = listings_ready.groupby(['host_id', 'host_name', 'host_about','host_is_superhost']).size().reset_index(name='num_host_listings')
    freq1 = freq1.sort_values(by=['num_host_listings'], ascending=False)
    freq1 = freq1[freq1['num_host_listings'] >= 10]
    freq1

with col2:
    st.markdown("<h5 style=>Anfitriones con palabras clave incluidas en su descripción:</h5>", unsafe_allow_html=True)
    st.markdown("<h6 style=>'company' y 'management'</h6>", unsafe_allow_html=True)
    st.dataframe(freq1[freq1['host_about'].str.contains('company|management', case=False)])


listings_ready['legality'] = listings_ready['maximum_nights'].apply(legality)
listings_ready['legality'].mask((listings_ready['host_name'].str.contains('hotel|hôtel', case=False)) | (listings_ready['host_about'].str.contains('company|management', case=False)), 'illegal', inplace=True)

m = folium.Map(location=[46.204391, 6.143158],width="%100",height="%100")
color_map = {'legal': 'green', 'illegal': 'red'}
for index, row in listings_ready.iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']],radius=1, color=color_map[row['legality']]).add_to(m)
st_data = st_folium(m,  width=1000)