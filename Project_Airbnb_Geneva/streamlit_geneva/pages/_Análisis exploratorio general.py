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

st.set_page_config(page_title="Análisis exploratorio general", layout="wide",page_icon="📊")

#Análisis general de los datos
st.title("ALOJAMIENTOS EN GINEBRA")

listings = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/listings.csv")
listings_details = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/listings_details.csv")
target_columns = ["id","property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "instant_bookable", "neighborhood_overview", "host_identity_verified", "bathrooms_text", "bedrooms"]
listings_ready = pd.merge(listings, listings_details[target_columns], on='id', how='left')
listings_ready = listings_ready.drop(columns=['neighbourhood_group','license'])
listings_ready['price'].mask(listings_ready['price'] >278.5 , listings_ready['price'].median(), inplace=True)
listings_ready['host_response_rate'] = pd.to_numeric(listings_ready['host_response_rate'].str.strip('%'))
listings_ready['review_scores_location'].fillna(listings_ready['review_scores_location'].mean(), inplace=True)
listings_ready['review_scores_communication'].fillna(listings_ready['review_scores_communication'].mean(), inplace=True)
listings_ready['review_scores_cleanliness'].fillna(listings_ready['review_scores_cleanliness'].mean(), inplace=True)
listings_ready['review_scores_accuracy'].fillna(listings_ready['review_scores_accuracy'].mean(), inplace=True)
listings_ready['review_scores_checkin'].fillna(listings_ready['review_scores_checkin'].mean(), inplace=True)
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


options = st.sidebar.selectbox(
    'Municipio',
    listings_ready['neighbourhood'].unique())

if options:
    listmunic=listings_ready.loc[listings_ready['neighbourhood']==options]
else:
    listmunic=listings_ready

col1,col2 = st.columns(2)

with col1:
    
    feq0=listings_ready['neighbourhood'].value_counts().sort_values(ascending=True)

    fig0 = px.bar(listings_ready, x=feq0.values, y=feq0.index, orientation='h', labels={'y':'Municipio', 'x':'Número de alojamientos'}, title='Número de alojamientos por municipio')
    st.plotly_chart(fig0, theme = 'streamlit', use_container_width=True)
    
with col2:
    lats2023 = listings_ready['latitude'].tolist()
    lons2023 = listings_ready['longitude'].tolist()
    locations = list(zip(lats2023, lons2023))

    map1 = folium.Map(location=[46.204391, 6.143158], zoom_start=12)
    FastMarkerCluster(data=locations).add_to(map1)
    
    st_data = st_folium(map1, width=500)
            

#listings_ready['neighbourhood'].unique()
#listings_ready_filter0 = listings_ready.loc[listings_ready['neighbourhood'] == '']

col1, col2 = st.columns(2)

with col1:

    lats2023 = listmunic['latitude'].tolist()
    lons2023 = listmunic['longitude'].tolist()
    locations = list(zip(lats2023, lons2023))

    map0 = folium.Map(location=[46.204391, 6.143158], zoom_start=12)
    FastMarkerCluster(data=locations).add_to(map0)
        
    st_data = st_folium(map0,  width=1000)


with col2:

    feq1=listmunic['room_type'].value_counts().sort_values(ascending=True)

    fig1 = px.bar(feq1, x=feq1.values, y=feq1.index, orientation='h', labels={'y':'Tipo de alojamiento', 'x':'Número de alojamientos'}, title='Tipos de alojamiento')
    st.plotly_chart(fig1, theme = 'streamlit', use_container_width=True)

         


prop = listmunic.groupby(['property_type','room_type']).room_type.count()
prop = prop.unstack()
prop['total'] = prop.iloc[:,0:3].sum(axis = 1)
prop = prop.sort_values(by=['total']) 
prop = prop[prop['total']>=5]
prop = prop.drop(columns=['total'])
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=prop['Entire home/apt'], y=prop.index, name='Entire home/apt', orientation = 'h'))
fig3.add_trace(go.Bar(x=prop['Private room'], y=prop.index, name='Private room', orientation = 'h'))
fig3.update_layout(barmode= 'stack', title = 'Tipos de propiedades en Ginebra')
st.plotly_chart(fig3, theme = 'streamlit', use_container_width=True)


feq3=listmunic['accommodates'].value_counts().sort_index()

fig4 = px.bar(listings_ready, x=feq3.index, y=feq3.values, labels={'y':'Número de alojamientos', 'x':'Tipo de alojamiento'}, title='Tipos de alojamiento',width=700, height=400)
st.plotly_chart(fig4, theme = 'streamlit', use_container_width=True)

fig5=px.box(listings_ready,x = 'neighbourhood', y = 'price', template="plotly_white")
st.plotly_chart(fig5, theme = 'streamlit', use_container_width=True)



