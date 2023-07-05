import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import tensorflow as tf




st.write("""
<style>
h1 {
    font-size: 70px;
}
h2 {
    font-size: 30px;
}
</style>
""", unsafe_allow_html=True)



st.write("""
# Предсказание дефолта по кредиту

## Данная программа предсказывает то, будет ли совершен дефолт по кредиту при помощи градиентного бустинга и нейронной сети
""", unsafe_allow_html=True)


st.sidebar.header('Файл для прогнозирования дефолта по кредиту')

st.sidebar.markdown(f"""
[Пример входного CSV-файла](https://drive.google.com/file/d/1kdQmZFEP1SL3httgm-CD_mHcB-m_mjrJ/view?usp=sharing)
""")

 
# Add a file uploader widget
uploaded_file = st.sidebar.file_uploader("Выберите файл формата CSV", type="csv")

if uploaded_file is not None:
    # Read the CSV file using pandas
    data = pd.read_csv(uploaded_file)
    # Display the loaded data
    st.write("## Загруженные данные:")
    data1 = pd.read_csv('C:\\Users\\Tseh\\Desktop\\Credit scoring\\Tests\\testing_file_60.csv')

    styled_data = data1.style.set_properties(**{'font-size': '30px'})

    # Increase font size for headers
    styles = [
        {'selector': 'th',
         'props': [('font-size', '18px')]}
    ]
    styled_data = styled_data.set_table_styles(styles)

    st.table(styled_data)



else:
    data = pd.read_csv('C:\\Users\\Tseh\\Desktop\\Credit scoring\\Tests\\testing_file_458.csv')
    #data = data.drop(['id', 'flag'], axis=1)


# Reads in saved classification model
model_gb = joblib.load('C:\\Users\\Tseh\\Desktop\\Credit scoring\\Models\\model_gb.joblib')
model_nn = tf.keras.models.load_model('C:\\Users\\Tseh\\Desktop\\Credit scoring\\Models\\model_nn.h5')


# Apply model to make predictions
prediction_1 = model_gb.predict(data)
prediction_proba_1 = model_gb.predict_proba(data)

# Apply model to make predictions
prediction_2 = model_nn.predict(data)
predicted_class = prediction_2[0, 0]
predicted_class = format(predicted_class, ".3f")
col1, col2 = st.columns(2)

with col1:
    st.subheader('Прогноз модели градиентного бустинга: ')

    if prediction_1[0] == 0:
        st.markdown("<p style='background-color: green; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'>КРЕДИТ ОДОБРЕН</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='background-color: red; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'>ОТКАЗАНО В КРЕДИТЕ</p>", unsafe_allow_html=True)

    st.subheader('Значение прогноза:')
    st.markdown("<div style='background-color: red; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'><span style='color: white; font-weight: bold;'>{:.3f}</span></div>".format(round(prediction_proba_1[0, 1], 3)), unsafe_allow_html=True)

with col2:
    st.subheader('Прогноз модели нейронной сети: ')

    if prediction_2[0, 0] <= 0.5:
        st.markdown("<p style='background-color: green; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'>КРЕДИТ ОДОБРЕН</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='background-color: red; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'>ОТКАЗАНО В КРЕДИТЕ</p>", unsafe_allow_html=True)

    st.subheader('Значение прогноза:')
    
    if prediction_2[0, 0] <= 0.5:
        st.markdown("<div style='background-color: green; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'><span style='color: white; font-weight: bold;'>{} </span></div>".format(predicted_class), unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: red; font-size: 20px; display: inline; padding: 0.2em 0.5em; border-radius: 0.5em;'><span style='color: white; font-weight: bold;'>{} </span></div>".format(predicted_class), unsafe_allow_html=True)
