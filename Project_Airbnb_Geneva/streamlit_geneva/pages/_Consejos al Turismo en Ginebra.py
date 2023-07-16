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

st.set_page_config(page_title="Consejos al Turismo en Ginebra", layout="wide",page_icon="🛩")

#Análisis general de los datos


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

st.title("ALOJAMIENTO A MEDIDA")

tab1, tab2= st.tabs(["Seguridad en Ginebra","Viajes a Ginebra según tamaño del grupo"])

with tab1:
    st.header("Seguridad en Ginebra")
    # mapa de criminalidad
    feq = pd.DataFrame([listings_ready.groupby('neighbourhood')['crime_rate'].mean().sort_values(ascending=True)])

    adam = gpd.read_file(r"/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/neighbourhoods.geojson")
    #feq = pd.DataFrame([feq])
    feq = feq.transpose()
    adam = pd.merge(adam, feq, on='neighbourhood', how='inner')
    adam.rename(columns={'crime_rate': 'average_crime_rate'}, inplace=True)
    adam.average_crime_rate = adam.average_crime_rate.round(decimals=0)

    print('Índice mínimo criminal: {}'.format(adam.average_crime_rate.min()))
    print('Índice máximo criminal: {}'.format(adam.average_crime_rate.max()))

    # Conseguimos colores para nuestras casas 
    map_dict = adam.set_index('neighbourhood')['average_crime_rate'].to_dict()
    color_scale = LinearColormap(['green','pink','purple'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

    def get_color(feature):
        value = map_dict.get(feature['properties']['neighbourhood'])
        return color_scale(value)


    # Hacemos el mapa
    map3 = folium.Map(location=[46.204391, 6.143158], zoom_start=11)
    folium.GeoJson(data=adam,
                name='Amsterdam',
                tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_crime_rate'],
                                                        labels=True,
                                                        sticky=True),
                style_function= lambda feature: {
                    'fillColor': get_color(feature),
                    'color': 'black',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity':0.9
                    },
                highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}
                ).add_to(map3)
    color_scale.caption = 'Índice de criminalidad'
    map3.add_child(color_scale)
    st_data = st_folium(map3,  width=1000)

    #escalar el crime_rate paara que me quede en un rango de 0-5, con 5 saltos. [0,1,2,3,4,5]

    listings_ready['crime_rate'] = listings_ready['crime_rate'].round(decimals=0)
    security = np.linspace(listings_ready['crime_rate'].min(),listings_ready['crime_rate'].max(),7,endpoint=True)

    def recode_crime_rate(rate):
        if  security[0] <= rate < security[1]:
            return 6
        elif security[1] <= rate < security[2]:
            return 5
        elif security[2] <= rate < security[3]:
            return 4
        elif security[3] <= rate < security[4]:
            return 3
        elif security[4] <= rate <= security[5]:
            return 2
        else:
            return 1
        
    listings_ready['security_rate'] = listings_ready['crime_rate'].apply(recode_crime_rate)

    feq = listings_ready[listings_ready['number_of_reviews']>=10]

    feq3=feq.groupby('neighbourhood')['review_scores_rating'].mean()
    feq5=feq.groupby('neighbourhood')['review_scores_location'].mean()
    feq4=feq.groupby('neighbourhood')['security_rate'].mean()

    fig1 = make_subplots(
        rows=1, cols=2, shared_yaxes=True, subplot_titles=('',''))

    fig1.add_trace(go.Bar(x=feq3.values, y=feq3.index,orientation = 'h', name='General'), row=1, col=1)
    fig1.add_trace(go.Bar(x=feq5.values, y=feq5.index,orientation = 'h', name='Localización'), row=1, col=1)
    fig1.update_xaxes(title_text="Valoración media por municipio", row=1, col=1)

    fig1.add_trace(go.Bar(x=feq4.values, y=feq4.index,orientation = 'h'), row=1, col=2)
    fig1.update_xaxes(title_text="Seguridad por municipio", row=1, col=2)

    fig1.update_layout(width=1000, height=700,
                    title_text='Valoraciones: general, ubicación y seguridad', showlegend=False)


    st.plotly_chart(fig1, use_container_width = True)

    data = listings_ready[['review_scores_rating','review_scores_location','review_scores_communication','review_scores_accuracy','review_scores_checkin','review_scores_cleanliness','security_rate','price']]
    corr = data.corr(method = 'spearman')

    fig2=px.imshow(corr.iloc[0:8,0:8], labels =dict(x="", y="", color="Coeficiente de correlación de Spearman"), x=corr.iloc[0:8,0:8].index,
                    y=corr.iloc[0:8,0:8].index, text_auto = '.2f', range_color = [-1, 1], color_continuous_scale="viridis",template="plotly_dark", height=600, title="Correlaciones entre valoraciones y precio")
    st.plotly_chart(fig2, use_container_width = True)

    from scipy.stats import spearmanr
    data1 = listings_ready['price'].values
    data2 = listings_ready['security_rate'].values
    stat, p = spearmanr(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        st.write('**El precio y la seguridad probablemente sean independientes**')
    else:
        st.write('**El precio y la seguridad probablemente sean dependientes**')

    from scipy.stats import spearmanr
    data1 = listings_ready['security_rate'].values
    data2 = listings_ready['review_scores_location'].values
    stat, p = spearmanr(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        st.write('**La seguridad y la localización probablemente sean independientes**')
    else:
        st.write('**La seguridad y la localización probablemente sean dependientes**')

    st.write('Estas correlaciones entre el precio y la localización y el precio inversas aunque débiles se pueden deber a que el índice general de criminalidad en Suiza es bastante bajo (https://es.numbeo.com/criminalidad/pa%25C3%25ADs/Suiza) y por tanto en las zonas más concurridas como el centro de Ginebra son en los que tienen lugar más actos delictivos (especialmente robos, https://es.numbeo.com/criminalidad/ciudad/Ginebra) donde, a su vez, los precios son más altos.')


with tab2:
    st.header("Encuentra tus mejores opciones")
    
    def triptype_recode(accomod):
        if accomod == 1:
            return 'Viaje de Negocios'
        elif accomod == 2:
            return 'Viaje en Pareja'
        else:
            return 'Viaje Familiar o en Grupo'
    
    listings_ready['trip_type'] = listings_ready['accommodates'].apply(triptype_recode)
    
    alojados = st.sidebar.selectbox('Tipos de viajes', listings_ready['trip_type'].unique())

    if alojados:
        listdf = listings_ready.loc[listings_ready['trip_type'] == alojados]

    #st.header("Viaje de Negocios: Alojamientos para 1 persona")

    col1, col2= st.columns(2)

    with col1:
        lats2023_1 = listdf['latitude'].tolist()
        lons2023_1 = listdf['longitude'].tolist()
        locations_1 = list(zip(lats2023_1, lons2023_1))
        airport = {'latitude': 46.23810, 'longitude':6.10895}
        train_station = {'latitude': 46.21023, 'longitude': 6.14262}
        ONU = {'latitude':46.22659,'longitude':6.1406}
        MBlanc = {'latitude':46.2075,'longitude':6.1484} 
        JD = {'latitude':46.20753,'longitude':6.15595} 
        map4 = folium.Map(location=[46.204391, 6.143158], zoom_start=10)
        FastMarkerCluster(data=locations_1).add_to(map4)
        folium.Marker([airport['latitude'], airport['longitude']],
             icon=folium.Icon(color="red", icon="plane"),).add_to(map4)
        folium.Marker([train_station['latitude'], train_station['longitude']],
                    icon=folium.Icon(color="red", icon="info-sign"),).add_to(map4)
        folium.Marker([ONU['latitude'], ONU['longitude']],
                    icon=folium.Icon(color="green", icon="info-sign"),).add_to(map4)
        folium.Marker([MBlanc['latitude'], MBlanc['longitude']],
                    icon=folium.Icon(color="green", icon="info-sign"),).add_to(map4)
        folium.Marker([JD['latitude'], JD['longitude']],
                    icon=folium.Icon(color="green", icon="info-sign"),).add_to(map4)
        st_folium(map4, width=1000)
    
    with col2:
        feq = pd.DataFrame([listdf.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)])
        feq = feq.transpose()
        adam = pd.merge(adam, feq, on='neighbourhood', how='inner')
        adam.rename(columns={'price': 'average_price'}, inplace=True)
        adam.average_price = adam.average_price.round(decimals=0)

        # Conseguimos colores para nuestras casas 
        map_dict = adam.set_index('neighbourhood')['average_price'].to_dict()
        color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

        def get_color(feature):
            value = map_dict.get(feature['properties']['neighbourhood'])
            return color_scale(value)

        
        # Hacemos el mapa
        map4B = folium.Map(location=[46.204391, 6.143158], zoom_start=11)
        folium.GeoJson(data=adam,
                    name='Geneva',
                    tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                                            labels=True,
                                                            sticky=True),
                    style_function= lambda feature: {
                        'fillColor': get_color(feature),
                        'color': 'black',
                        'weight': 1,
                        'dashArray': '5, 5',
                        'fillOpacity':0.9
                        },
                    highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}
                    ).add_to(map4B)
        st_folium(map4B, width=1000)

    st.write('El precio medio mínimo de los alojamientos disponibles en Ginebra para una persona es de {}$ y se encuentra en el municipio {}'.format(adam['average_price'].min(),adam[adam['average_price'] == adam['average_price'].min()]['neighbourhood'].values[0]))
    st.write('El precio medio máximo de los alojamientos disponibles en Ginebra para una persona es de {}$ y se encuentra en el municipio {}'.format(adam['average_price'].max(),adam[adam['average_price'] == adam['average_price'].max()]['neighbourhood'].values[0]))

    st.markdown('<h5 style=>Comunicación por tranvía en Ginebra</h5>', unsafe_allow_html=True)
    st.image(r'/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/streamlit_geneva/tramway_geneva.png')

    st.markdown('<h5 style=>¿Qué aspecto preocupa más de los alojamientos en función de sus valoraciones?</h5>', unsafe_allow_html=True)
    
    if alojados == 'Viaje de Negocios':
        st.image(r'/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/streamlit_geneva/polargraph_negocios.png')
    elif alojados == 'Viaje en Pareja':
        st.image(r'/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/streamlit_geneva/polargraph_pareja.png')
    else:
        st.image(r'/Users/laura/Desktop/Upgrade Hub/DataAnalytics/Contenidos/Modulo2/Project_Airbnb_Geneva/streamlit_geneva/polargraph_grupo.png')
    

    
    superhost = listdf[listdf['host_is_superhost'] == 'Superanfitrión']
    standardhost = listdf[listdf['host_is_superhost'] != 'Superanfitrión']
    
    st.markdown("<h5 style=>¿Realmente los superanfitriones marcan la diferencia en Ginebra?</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=superhost['host_response_rate'].value_counts().index, y=superhost['host_response_rate'].value_counts().values, name='Superanfitrión'))
        fig4.update_xaxes(title_text="Porcentaje de respuesta (%)")
        fig4.update_yaxes(title_text="Número de alojamientos")
        fig4.add_trace(go.Bar(x=standardhost['host_response_rate'].value_counts().index, y=standardhost['host_response_rate'].value_counts().values, name='Anfitrión estándar'))
        fig4.update_xaxes(title_text="Porcentaje de respuesta (%)")
        fig4.update_yaxes(title_text="Número de alojamientos")
        fig4.update_layout(barmode = 'stack', title_text='', showlegend=True)
        st.plotly_chart(fig4, theme='streamlit', use_container_width = True)

    with col2:
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=superhost['host_response_time'].value_counts().index, y=superhost['host_response_time'].value_counts().values, name='Superanfitrión'))
        fig5.update_xaxes(title_text="Tiempo de respuesta")
        fig5.update_yaxes(title_text="Número de alojamientos")
        fig5.add_trace(go.Bar(x=standardhost['host_response_time'].value_counts().index, y=standardhost['host_response_time'].value_counts().values, name='Anfitrión estándar'))
        fig5.update_xaxes(title_text="Tiempo de respuesta")
        fig5.update_yaxes(title_text="Número de alojamientos")
        fig5.update_layout(barmode = 'stack', title_text='', showlegend=True)
        st.plotly_chart(fig5, theme='streamlit', use_container_width = True)

    st.write('**Comparación entre los precios de alojamientos ofertados por superanfitriones y anfitriones estándar**')
    
    from scipy.stats import mannwhitneyu
    data1 = listdf[listdf['host_is_superhost']=='Superanfitrión']['price']
    data2 = listdf[listdf['host_is_superhost']=='Anfitrión estándar']['price']
    stat, p = mannwhitneyu(data1, data2)
    if p > 0.05:
        st.write('El precio de los alojamientos de anfitriones estándar y superanfitriones PROBABLEMENTE tengan la MISMA DISTRIBUCIÓN: Mann Whitney test')
    else:
        st.write('El precio de los alojamientos de anfitriones estándar y superanfitriones PROBABLEMENTE tengan la DISTINTA DISTRIBUCIÓN: Mann Whitney test')


    st.write('Después de ver estos resultados, aconsejaría a Airbnb ajustar la escala de las valoraciones o considerar otro tipo de aspectos a valorar de los alojamientos para ver realmente lo que diferencia un alojamiento de otro y saber realmente a qué se puede deber esa ligera diferencia en los precios de los alojamientos')
