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

st.set_page_config(page_title="Disponibilidad y Precio en el tiempo", layout="wide",page_icon="⏳")

#Análisis general de los datos


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


calendar = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/calendar.csv", parse_dates=['date'], index_col=['listing_id'])
reviews = pd.read_csv(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/reviews_details.csv", parse_dates=['date'])

calendar.price = calendar.price.str.replace(",","")
calendar['price'] = pd.to_numeric(calendar['price'].str.strip('$'))
calendar = calendar[calendar.date < '2024-03-28']

listings_ready.set_index('id', inplace=True)
listings_ready.index.name = "listing_id"
calendar = pd.merge(calendar, listings_ready[['accommodates']], on = "listing_id", how = "left")


st.markdown("<h5 style=>Disponibilidad de alojamientos en Ginebra en el tiempo según el tipo de alojamiento</h5>", unsafe_allow_html=True)

sum_available1 = calendar[(calendar.available == "t") & (calendar.accommodates == 1)].groupby(['date']).size().to_frame(name= 'available').reset_index()
sum_available1['weekday'] = sum_available1['date'].dt.day_name()
sum_available1 = sum_available1.set_index('date')

sum_available2 = calendar[(calendar.available == "t") & (calendar.accommodates == 2)].groupby(['date']).size().to_frame(name= 'available').reset_index()
sum_available2['weekday'] = sum_available2['date'].dt.day_name()
sum_available2 = sum_available2.set_index('date')

sum_available3 = calendar[(calendar.available == "t") & (calendar.accommodates >= 3)].groupby(['date']).size().to_frame(name= 'available').reset_index()
sum_available3['weekday'] = sum_available3['date'].dt.day_name()
sum_available3 = sum_available3.set_index('date')

fig0 = go.Figure()
fig0.add_trace(go.Scatter(x=sum_available1.index, y=sum_available1['available'].values, mode='lines',text=sum_available1['weekday'].values, name='Alojamientos de 1 persona'))
fig0.add_trace(go.Scatter(x=sum_available2.index, y=sum_available2['available'].values, mode='lines',text=sum_available2['weekday'].values, name='Alojamientos de 2 personas'))
fig0.add_trace(go.Scatter(x=sum_available3.index, y=sum_available3['available'].values, mode='lines',text=sum_available3['weekday'].values, name='Alojamientos de 3 o más personas'))
fig0.update_layout(title='Disponibilidad de alojamientos por fecha', xaxis_title='Fecha', yaxis_title='Número de alojamientos disponibles')

st.plotly_chart(fig0, theme='streamlit', use_container_width=True)

st.write('36ª sesión especial del Consejo de Derechos Humanos sobre el impacto en los derechos humanos del actual conflicto en Sudán - 11 de mayo de 2023 (https://www.ohchr.org/es/hr-bodies/hrc/special-sessions/session36/36-special-session)')
st.write('VITAFOODS EUROPE 2023 MAY 9-11, 2023: empresas dedicadas a la nutracéutica se reunen para estudiar sus negocios y problemas con otros expertos en el sector (https://palexpo.ch/en/evenement/en-vitafoods-europe-2023/)')



st.markdown("<h5 style=>Precio medio de alojamientos en Ginebra en el tiempo según el tipo de alojamiento</h5>", unsafe_allow_html=True)


average_price1 = calendar[(calendar.available == "t") & (calendar.accommodates == 1)].groupby(['date']).mean().astype(np.int64).reset_index()
average_price1['weekday'] = average_price1['date'].dt.day_name()
average_price1 = average_price1.set_index('date')

average_price2 = calendar[(calendar.available == "t") & (calendar.accommodates == 2)].groupby(['date']).mean().astype(np.int64).reset_index()
average_price2['weekday'] = average_price2['date'].dt.day_name()
average_price2 = average_price2.set_index('date')

average_price3 = calendar[(calendar.available == "t") & (calendar.accommodates >= 3)].groupby(['date']).mean().astype(np.int64).reset_index()
average_price3['weekday'] = average_price3['date'].dt.day_name()
average_price3 = average_price3.set_index('date')


fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=average_price1.index, y=average_price1['price'].values, mode='lines',text=average_price1['weekday'].values, name='Alojamientos de 1 persona'))
fig1.add_trace(go.Scatter(x=average_price2.index, y=average_price2['price'].values, mode='lines',text=average_price2['weekday'].values, name='Alojamientos de 2 personas'))
fig1.add_trace(go.Scatter(x=average_price3.index, y=average_price3['price'].values, mode='lines',text=average_price3['weekday'].values, name='Alojamientos de 3 personas'))
fig1.update_layout(title='Precio medio de alojamientos por fecha', xaxis_title='Fecha', yaxis_title='Precio medio',width=1000)

st.plotly_chart(fig1, theme='streamlit', use_container_width=True)