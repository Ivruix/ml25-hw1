import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    with open('ridge_model.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()

st.title('Прогнозирование цен на автомобили')
tab1, tab2, tab3 = st.tabs(['EDA анализ', 'Прогнозирование', 'Важность признаков'])

with tab1:
    st.header('Анализ данных (EDA)')
    
    st.subheader('Попарные распределения числовых признаков')
    st.image('resources/pairs.png')

    st.subheader('Корреляционная матрица')
    st.image('resources/corr.png')

with tab2:
    st.header('Прогнозирование цены автомобиля')
    st.subheader('Введите параметры автомобиля')
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input('Год выпуска', min_value=1983, value=2020)
        km_driven = st.number_input('Пробег (км)', min_value=0, value=50000)
        mileage = st.number_input('Расход топлива (км/л)', min_value=0.0, value=20.0)
        engine = st.number_input('Объем двигателя (CC)', min_value=0, value=1500)
        max_power = st.number_input('Мощность (bhp)', min_value=0.0, value=100.0)

    with col2:
        fuel = st.selectbox('Тип топлива', ['Diesel', 'Petrol', 'LPG', 'CNG'])
        seller_type = st.selectbox('Тип продавца', ['Individual', 'Dealer', 'Trustmark Dealer'])
        transmission = st.selectbox('Трансмиссия', ['Manual', 'Automatic'])
        owner = st.selectbox('Владелец', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
        seats = st.selectbox('Количество сидений', [2, 4, 5, 6, 7, 8, 9, 10, 14])
    
    if st.button('Предсказать цену', type='primary'):
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner]
        })
        
        cat_data = pipeline['encoder'].transform(input_data[pipeline['categorical_features']])
        cat_df = pd.DataFrame(cat_data, columns=pipeline['encoder'].get_feature_names_out(pipeline['categorical_features']))
        
        numeric_data = input_data[pipeline['numeric_features']]
        final_data = pd.concat([numeric_data.reset_index(drop=True), cat_df], axis=1)

        prediction = pipeline['model'].predict(final_data)
        
        st.success(f'Предсказанная цена автомобиля: {prediction[0]:,.2f}')

with tab3:
    st.header('Важность признаков')

    coef_df = pd.DataFrame({
        'Признак': pipeline['feature_names'],
        'Коэффициент': pipeline['model'].coef_
    })
    coef_df['Абсолютная важность'] = np.abs(coef_df['Коэффициент'])
    coef_df = coef_df.sort_values('Абсолютная важность', ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    top_20 = coef_df.head(20)
    colors = ['red' if x < 0 else 'blue' for x in top_20['Коэффициент']]
    
    ax.bar(top_20['Признак'], top_20['Коэффициент'], color=colors)
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Значение коэффициента')
    ax.set_title('Топ 20 самых важных признаков')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.subheader('Таблица коэффициентов модели')
    st.dataframe(coef_df)