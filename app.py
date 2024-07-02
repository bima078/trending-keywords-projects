# import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb

# import modelling libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def main():
    # configure the page information
    st.set_page_config(
        page_title= 'Trending Keywords Predictions by Bima!',
        page_icon= ':fire:'
    )
    
    # initialize and load the data
    loaded_data = pd.read_csv('Include/trending-keywords-datasets.csv', delimiter=';', decimal=',')
    loaded_data['CPC (IDR)'] = loaded_data['CPC (IDR)'].astype('float')
    loaded_data["Competitive Density"] = loaded_data["Competitive Density"].astype('float')
    model = jb.load('Include/knn_model.sav')

    # configure the title
    st.header('Trending Keywords Predictions by Bima! :fire:', anchor = False)
    st.divider()

    st.header('Input values', anchor = False)
    col1, col2 = st.columns([2, 2]) # create two columns

    with col1:
        # keyword input
        def keywords_choice(option):
            temp = loaded_data[loaded_data['Keyword'] == option].iloc[0]
            temp = temp['idx']
            return int(temp)

        v_keywords = st.selectbox(label='Keywords',
                                  options=list(loaded_data['Keyword']))
        
        # st.write(keywords_choice(v_keywords))
        
        # volume input
        v_volume = st.number_input(label = 'Volume',
                                   min_value = float(loaded_data['Volume'].min()),
                                   max_value = float(loaded_data['Volume'].max()))
                                   
        # april input
        v_Apr = st.number_input(label = 'Trend April 2023',
                                  min_value = float(loaded_data["Trend April'23"].min()),
                                  max_value = float(loaded_data["Trend April'23"].max()))
        
        # may input
        v_May = st.number_input(label = 'Trend May 2023',
                                  min_value = float(loaded_data["Trend Mei'23"].min()),
                                  max_value = float(loaded_data["Trend Mei'23"].max()))
        
        # juni input
        v_Jun = st.number_input(label = 'Trend Juny 2023',
                                  min_value = float(loaded_data["Trend Juni'23"].min()),
                                  max_value = float(loaded_data["Trend Juni'23"].max()))

        # juli input
        v_Jul = st.number_input(label = 'Trend July 2023',
                                  min_value = float(loaded_data["Trend Juli'23"].min()),
                                  max_value = float(loaded_data["Trend Juli'23"].max()))
                                   
        # august input
        v_Aug = st.number_input(label = 'Trend August 2023',
                                  min_value = float(loaded_data["Trend Agu'23"].min()),
                                  max_value = float(loaded_data["Trend Agu'23"].max()))
        
        # september input
        v_Sept = st.number_input(label = 'Trend September 2023',
                                  min_value = float(loaded_data["Trend Sep'23"].min()),
                                  max_value = float(loaded_data["Trend Sep'23"].max()))
    
    with col2:
        # october input
        v_Oct = st.number_input(label = 'Trend October 2023',
                                  min_value = float(loaded_data["Trend Okt'23"].min()),
                                  max_value = float(loaded_data["Trend Okt'23"].max()))
        
        # november input
        v_Nov = st.number_input(label = 'Trend November 2023',
                                  min_value = float(loaded_data["Trend Nov'23"].min()),
                                  max_value = float(loaded_data["Trend Nov'23"].max()))
        
        # desember input
        v_Dec = st.number_input(label = 'Trend December 2023',
                                  min_value = float(loaded_data["Trend Dec'23"].min()),
                                  max_value = float(loaded_data["Trend Dec'23"].max()))

        # january input
        v_Jan = st.number_input(label = 'Trend January 2024',
                                  min_value = float(loaded_data["Trend Jan'24"].min()),
                                  max_value = float(loaded_data["Trend Jan'24"].max()))
                                   
        # august input
        v_Feb = st.number_input(label = 'Trend February 2024',
                                  min_value = float(loaded_data["Trend Feb'24"].min()),
                                  max_value = float(loaded_data["Trend Feb'24"].max()))
        
        # september input
        v_Mar = st.number_input(label = 'Trend March 2024',
                                  min_value = float(loaded_data["Trend Mar'24"].min()),
                                  max_value = float(loaded_data["Trend Mar'24"].max()))

        # cpc input
        v_cpc = st.number_input(label = 'CPC',
                                  min_value = float(loaded_data["CPC (IDR)"].min()),
                                  max_value = float(loaded_data["CPC (IDR)"].max()))
        
        # competitive density input
        v_compe = st.number_input(label = 'Competitive Density',
                                  min_value = float(loaded_data["Competitive Density"].min()),
                                  max_value = float(loaded_data["Competitive Density"].max()))
    
    # do the prediction
    but_predict = st.button("Predict!", type='primary', key='pred1')
    if but_predict:
        st.divider()
        feature = [keywords_choice(v_keywords), v_volume, v_Apr, v_May, v_Jun, v_Jul, v_Aug, v_Sept,
                   v_Oct, v_Nov, v_Dec, v_Jan, v_Feb, v_Mar, v_cpc, v_compe]
        feature = np.reshape(feature, (1, -1))
        pred = model.predict([feature[0]])[0]
        
        # give the prediction information
        if pred == 0.0:
            st.header('Prediction results: :orange[Not trending!]', anchor = False)
            st.write('The following keyword/s is/are not trending!')
        elif pred == 1.0:
            st.header('Prediction results: :green[Trending!]', anchor = False)
            st.write('The following keyword/s is/are trending!')
    
    # copyright claim
    st.divider()
    st.caption('*Copyright (c) Bima 2024*')
    
if __name__ == '__main__':
    main()