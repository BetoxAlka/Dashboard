# ------------------------------------------------------------ Importar Librerias ------------------------------------------------------------
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# -------------------------------------------------- Configuracion Inicial del Dashboard --------------------------------------------------
st.set_page_config(page_title='Analisis de Datos del Socio Formador', layout='wide')

# ------------------------------------------------------------ Funciones ------------------------------------------------------------
# Funcion para cargar Datos
@st.cache_data
def load_data(filename):
    # Cargar archivo
    file_path = filename
    df = pd.read_csv(file_path)

    return df

# Funcion para crear Sumatoria de Variables
def count_sum(data):
    # Generar Sumatoria de variables
    numeric_cols = data.select_dtypes(include=['number']) # Unicamente columnas numericas
    n_data = numeric_cols.replace({-1:0}) # Eliminar Numeros Negativos
    columns_sums = n_data.drop('time',axis=1).sum() # Sumatoria de las Variables
    df_sums = pd.DataFrame(columns_sums, columns=['Sumatory']) # Convertir a DataFrame

    return df_sums

# Funcion para crear Cruces en Cero
def count_zeros(data):
    # Generar Cruces en Cero
    # Filtramos solo las columnas numéricas para estandarizar
    numeric_cols = data.select_dtypes(include=['number']).columns

    # Estandarización de las columnas numéricas
    scaler = StandardScaler()
    my_data_normalized = pd.DataFrame(scaler.fit_transform(data[numeric_cols]), columns=numeric_cols)

    # Contar cruces por cero para cada columna
    cross_zero_counts = {}
    for col in my_data_normalized.columns:
        # Contar cruces por cero (un cambio de signo entre valores consecutivos)
        cross_zero_counts[col] = ((my_data_normalized[col].shift(1) * my_data_normalized[col]) < 0).sum()

    # Convertir el resultado en un DataFrame
    df_cross_zero_counts = pd.DataFrame(list(cross_zero_counts.items()), columns=['Column', 'Zero_Crossings'])

    return df_cross_zero_counts

# Funcion para obtener intervalos de emociones
def emotion_intervals(session):
    # Identificar las columnas de sentimientos
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    # Lista para tiempos maximos
    max_intervals_session = []
    # Graficar cada sentimiento
    for i, col in enumerate(sentiment_columns):
        # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
        intervals = []
        start = None
        for j in range(len(session)):
            if session[col].iloc[j] == 1:
                if start is None:
                    start = session['time'].iloc[j]
            else:
                if start is not None:
                    end = session['time'].iloc[j - 1]
                    intervals.append((start, end))
                    start = None
        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervals.append((start, session['time'].iloc[-1]))

        # Dibujar barras horizontales para los intervalos
        #for (start, end) in intervals:
        #    ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

        # Calcular intervalos con el tiempo maximo
        time = 0
        for i in intervals:
            inter = i[1] - i[0]
            if inter > time:
                time = inter
        max_intervals_session.append(time)

    return max_intervals_session

# Funcion para obtener columnas numericas
def numeric_cols(data):
    cols = data.select_dtypes(include=['number']).columns

    return cols

# Funcion para la carga de datos de la Regresion Lineal Simple
@st.cache_data
def regresion(data):
    #Tomar valores numericos
    numeric = data.select_dtypes(include=['number'])

    # Coeficientes de correlacion entre las variables
    corr_f = numeric.corr()

    # Valor absoluto de todas las correlaciones entre las variables
    corr_abs = abs(corr_f)

    return corr_f, corr_abs

# Funcion para generar Line Chart
@st.cache_data
def line_graph(session1,session2,session3,session4,selected_var):
            # Generador de grafica
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

            # Crear la gráfica de barras
            plt.style.use('Solarize_Light2')  
            # Gráfica de barras
            colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

            # Rotar las etiquetas del eje X
            plt.xticks(rotation=45)

            # Sesion 1
            session1.plot(x='time',y=selected_var, ax=axs[0,0], color='skyblue')        
            # Añadir etiquetas y título
            axs[0,0].set_xlabel("Tiempo", fontsize=12)
            axs[0,0].set_ylabel(f"{selected_var}", fontsize=12)
            axs[0,0].set_title(f'{selected_var} VS Tiempo - SESION 1', fontsize=16)

            # Sesion 2
            session2.plot(x='time',y=selected_var, ax=axs[0,1], color='skyblue')        
            # Añadir etiquetas y título
            axs[0,1].set_xlabel("Tiempo", fontsize=12)
            axs[0,1].set_ylabel(f"{selected_var}", fontsize=12)
            axs[0,1].set_title(f'{selected_var} VS Tiempo - SESION 2', fontsize=16)

            # Sesion 3
            session3.plot(x='time',y=selected_var, ax=axs[1,0], color='skyblue')        
            # Añadir etiquetas y título
            axs[1,0].set_xlabel("Tiempo", fontsize=12)
            axs[1,0].set_ylabel(f"{selected_var}", fontsize=12)
            axs[1,0].set_title(f'{selected_var} VS Tiempo - SESION 3', fontsize=16)

            # Sesion 4
            session4.plot(x='time',y=selected_var, ax=axs[1,1], color='skyblue')        
            # Añadir etiquetas y título
            axs[1,1].set_xlabel("Tiempo", fontsize=12)
            axs[1,1].set_ylabel(f"{selected_var}", fontsize=12)
            axs[1,1].set_title(f'{selected_var} VS Tiempo - SESION 4', fontsize=16)

            plt.tight_layout()

            st.pyplot(fig)

# Funcion para generar Bar Chart
@st.cache_data
def bar_graph(time_var_max,option):
            fig, ax = plt.subplots(figsize=(20,10))

            # Crear la gráfica de barras
            plt.style.use('Solarize_Light2')  

            # Rotar las etiquetas del eje X
            plt.xticks(rotation=45)

            if option == 'TimeMax':
                bars = plt.bar(time_var_max.index, time_var_max['TimeForMax'], color='skyblue')
                # Añadir etiquetas y título
                plt.xlabel("Sesiones", fontsize=12)
                plt.ylabel(f"Tiempo para {selected_var} Maxima", fontsize=12)
                plt.title(f"Tiempo que transcurre en cada Sesion para alcanzar la {selected_var} Maxima - {paciente_seleccionado}", fontsize=20)

            elif option == 'DistMax':
                bars = plt.bar(dists_max.index, dists_max['DistMax'], color='skyblue')
                # Añadir etiquetas y título
                plt.xlabel("Sesiones", fontsize=12)
                plt.ylabel("Distancia Maxima", fontsize=12)
                plt.title(f"Distancias Maximas de cada Sesion - {paciente_seleccionado}", fontsize=16)

            # Mostrar el valor encima de cada barra
            for bar in bars:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{bar.get_height():.1f}',
                    ha='center', va='bottom'
                )

            plt.tight_layout()

            st.pyplot(fig)

# Funcion para generar Histograma
@st.cache_data
def histogram(session1,session2,session3,session4):
    # Generador de Graficas
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

    # Crear la gráfica de barras
    plt.style.use('Solarize_Light2')  
    # Gráfica de barras
    colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

    # Sesion 1
    axs[0,0].hist(session1['Distancia'], alpha=0.5, edgecolor='black', bins=10, color='skyblue')
    # Anaidir Etiquetas
    axs[0,0].set_title('Histograma Distancia - Sesion 1', fontsize=16)
    axs[0,0].set_xlabel('Distancia', fontsize=12)
    axs[0,0].set_ylabel('Frecuencia', fontsize=12)

    # Sesion 2
    axs[0,1].hist(session2['Distancia'], alpha=0.5, edgecolor='black', bins=10, color='skyblue')
    # Anaidir Etiquetas
    axs[0,1].set_title('Histograma Distancia - Sesion 2', fontsize=16)
    axs[0,1].set_xlabel('Distancia', fontsize=12)
    axs[0,1].set_ylabel('Frecuencia', fontsize=12)

    # Sesion 3
    axs[1,0].hist(session3['Distancia'], alpha=0.5, edgecolor='black', bins=10, color='skyblue')
    # Anaidir Etiquetas
    axs[1,0].set_title('Histograma Distancia - Sesion 3', fontsize=16)
    axs[1,0].set_xlabel('Distancia', fontsize=12)
    axs[1,0].set_ylabel('Frecuencia', fontsize=12)

    # Sesion 4
    axs[1,1].hist(session4['Distancia'], alpha=0.5, edgecolor='black', bins=10, color='skyblue')
    # Anaidir Etiquetas
    axs[1,1].set_title('Histograma Distancia - Sesion 4', fontsize=16)
    axs[1,1].set_xlabel('Distancia', fontsize=12)
    axs[1,1].set_ylabel('Frecuencia', fontsize=12)

    plt.tight_layout()

    st.pyplot(fig)

# Funcion para generar Density Chart
@st.cache_data
def density(session1,session2,session3,session4):
    # Generador de Graficas
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

    # Crear la gráfica de barras
    plt.style.use('Solarize_Light2')  
    # Gráfica de barras
    colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

    # Sesion 1
    session1['Distancia'].plot.density(ax=axs[0,0])
    # Anaidir Etiquetas
    axs[0,0].set_title('Gráfica Densidad - Sesion 1', fontsize=16)
    axs[0,0].set_xlabel('Intervalo', fontsize=12)
    axs[0,0].set_ylabel('Frecuencia', fontsize=12)

    # Sesion 2
    session2['Distancia'].plot.density(ax=axs[0,1])
    # Anaidir Etiquetas
    axs[0,1].set_title('Gráfica Densidad - Sesion 2', fontsize=16)
    axs[0,1].set_xlabel('Intervalo', fontsize=12)
    axs[0,1].set_ylabel('Frecuencia', fontsize=12)

    # Sesion 3
    session3['Distancia'].plot.density(ax=axs[1,0])
    # Anaidir Etiquetas
    axs[1,0].set_title('Gráfica Densidad - Sesion 3', fontsize=16)
    axs[1,0].set_xlabel('Intervalo', fontsize=12)
    axs[1,0].set_ylabel('Frecuencia', fontsize=12)

    # Sesion 4
    session4['Distancia'].plot.density(ax=axs[1,1])
    # Anaidir Etiquetas
    axs[1,1].set_title('Gráfica Densidad - Sesion 4', fontsize=16)
    axs[1,1].set_xlabel('Intervalo', fontsize=12)
    axs[1,1].set_ylabel('Frecuencia', fontsize=12)

    plt.tight_layout()

    st.pyplot(fig)

# Funcion para generar Emotions Chart
@st.cache_data
def emotions_dist(sentiment_columns,session1,session2,session3,session4):
    # Crear la figura y los ejes
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20,14))

    # Crear la gráfica de barras
    plt.style.use('Solarize_Light2')  
    
    # Lista de Ejes
    axis = [0,1,2,3]

    # Graficar cada eje
    num_session = 1
    for ax in axis:
        # Establecer dataset por iteracion
        if num_session == 1:
            data = session1
        elif num_session == 2:
            data = session2
        elif num_session == 3:
            data = session3
        elif num_session == 4:
            data = session4

        # Graficar cada sentimiento
        for i, col in enumerate(sentiment_columns):
            # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
            intervals = []
            start = None
            for j in range(len(data)):
                if data[col].iloc[j] == 1:
                    if start is None:
                        start = data['time'].iloc[j]
                else:
                    if start is not None:
                        end = data['time'].iloc[j - 1]
                        intervals.append((start, end))
                        start = None
            # Capturar el último intervalo si termina en el último valor de la columna
            if start is not None:
                intervals.append((start, data['time'].iloc[-1]))

            # Dibujar barras horizontales para los intervalos
            for (start, end) in intervals:
                axs[ax].hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

        # Iterar en sesion
        num_session += 1

    # Etiquetas y formato
    axs[0].set_yticks(range(len(sentiment_columns)))
    axs[0].set_yticklabels(sentiment_columns)
    axs[0].set_xlabel("Tiempo",fontsize=12)
    axs[0].set_ylabel("Emociones",fontsize=12)
    axs[0].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 1",fontsize=16)

    # Etiquetas y formato
    axs[1].set_yticks(range(len(sentiment_columns)))
    axs[1].set_yticklabels(sentiment_columns)
    axs[1].set_xlabel("Tiempo",fontsize=12)
    axs[1].set_ylabel("Emociones",fontsize=12)
    axs[1].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 2",fontsize=16)

    # Etiquetas y formato
    axs[2].set_yticks(range(len(sentiment_columns)))
    axs[2].set_yticklabels(sentiment_columns)
    axs[2].set_xlabel("Tiempo",fontsize=12)
    axs[2].set_ylabel("Emociones",fontsize=12)
    axs[2].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 3",fontsize=16)

    # Etiquetas y formato
    axs[3].set_yticks(range(len(sentiment_columns)))
    axs[3].set_yticklabels(sentiment_columns)
    axs[3].set_xlabel("Tiempo",fontsize=12)
    axs[3].set_ylabel("Emociones",fontsize=12)
    axs[3].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 4",fontsize=16)

    plt.tight_layout()

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    st.pyplot(fig)

# Funcion para generar Max Emotion Chart
@st.cache_data
def maxtemp_emotion(max_time_emotion):
    # Número de variables/categorías
    num_vars = len(max_time_emotion)

    # Ángulos de cada categoría en el radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Repetimos el primer valor al final para cerrar el gráfico
    emotion1_values = max_time_emotion['01_C'].tolist()
    emotion1_values += emotion1_values[:1]

    emotion2_values = max_time_emotion['02_A'].tolist()
    emotion2_values += emotion2_values[:1]

    emotion3_values = max_time_emotion['03_D'].tolist()
    emotion3_values += emotion3_values[:1]

    emotion4_values = max_time_emotion['04_M'].tolist()
    emotion4_values += emotion4_values[:1]

    angles += angles[:1]

    # Inicializar el gráfico en coordenadas polares
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(20, 10), subplot_kw=dict(polar=True))

    # Crear la gráfica de barras
    plt.style.use('Solarize_Light2')  
    # Gráfica de barras
    colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

    # Dibujar la línea del gráfico
    axs[0,0].plot(angles, emotion1_values, color='skyblue', linewidth=2, linestyle='solid')
    # Rellenar el área
    axs[0,0].fill(angles, emotion1_values, color='skyblue', alpha=0.4)
    # Añadir etiquetas para cada categoría
    axs[0,0].set_xticks(angles[:-1])
    axs[0,0].set_xticklabels(max_time_emotion.index,fontsize=10)
    # Ajustar los límites del eje radial y el título
    axs[0,0].set_ylim(0, max(emotion1_values))
    axs[0,0].set_title('                                                        CANSANCIO', fontsize=16)

    # Dibujar la línea del gráfico
    axs[0,1].plot(angles, emotion2_values, color='skyblue', linewidth=2, linestyle='solid')
    # Rellenar el área
    axs[0,1].fill(angles, emotion2_values, color='skyblue', alpha=0.4)
    # Añadir etiquetas para cada categoría
    axs[0,1].set_xticks(angles[:-1])
    axs[0,1].set_xticklabels(max_time_emotion.index,fontsize=10)
    # Ajustar los límites del eje radial y el título
    axs[0,1].set_ylim(0, max(emotion2_values))
    axs[0,1].set_title('                                                        ANSIEDAD', fontsize=16)

    # Dibujar la línea del gráfico
    axs[1,0].plot(angles, emotion3_values, color='skyblue', linewidth=2, linestyle='solid')
    # Rellenar el área
    axs[1,0].fill(angles, emotion3_values, color='skyblue', alpha=0.4)
    # Añadir etiquetas para cada categoría
    axs[1,0].set_xticks(angles[:-1])
    axs[1,0].set_xticklabels(max_time_emotion.index,fontsize=10)
    # Ajustar los límites del eje radial y el título
    axs[1,0].set_ylim(0, max(emotion3_values))
    axs[1,0].set_title('                                                        DOLOR', fontsize=16)

    # Dibujar la línea del gráfico
    axs[1,1].plot(angles, emotion4_values, color='skyblue', linewidth=2, linestyle='solid')
    # Rellenar el área
    axs[1,1].fill(angles, emotion4_values, color='skyblue', alpha=0.4)
    # Añadir etiquetas para cada categoría
    axs[1,1].set_xticks(angles[:-1])
    axs[1,1].set_xticklabels(max_time_emotion.index,fontsize=10)
    # Ajustar los límites del eje radial y el título
    axs[1,1].set_ylim(0, max(emotion4_values))
    axs[1,1].set_title('                                                        MOTIVACION', fontsize=16)

    plt.tight_layout()

    st.pyplot(fig)

# Funcion para generar Scatter Chart
@st.cache_data
def scatter_graph(x_var,y_var,session1,session2,session3,session4):
    # Sesion 1
    # Declaramos las variables dependientes e independientes para la regresion lineal
    vars_indep_session1 = session1[[x_var]]
    var_dep_session1 = session1[y_var]
    # Generamos el Modelo Lineal
    model_session1 = LinearRegression()
    # Ajustamos el modelo con las variables antes declaradas
    model_session1.fit(X=vars_indep_session1, y=var_dep_session1)
    # Predecimos los valores a partir de la variable
    y_pred_session1 = model_session1.predict(X=session1[[x_var]])
    # Insertamos la columna de predicciones en el DataFrame
    session1_copy = session1.copy()
    session1_copy.insert(0,'predicciones',y_pred_session1)
    # Corrobaremos cual es el coeficiente de Determinacion de nuestro modelo (R_Cuadrada)
    coef_deter_session1 = model_session1.score(X=vars_indep_session1, y=var_dep_session1)
    # Corroboramos cual es el coeficiente de correlacion de nuestro modelo (R)
    coef_correl_session1 = np.sqrt(coef_deter_session1)

    # Sesion 2
    # Declaramos las variables dependientes e independientes para la regresion lineal
    vars_indep_session2 = session2[[x_var]]
    var_dep_session2 = session2[y_var]
    # Generamos el Modelo Lineal
    model_session2 = LinearRegression()
    # Ajustamos el modelo con las variables antes declaradas
    model_session2.fit(X=vars_indep_session2, y=var_dep_session2)
    # Predecimos los valores a partir de la variable
    y_pred_session2 = model_session2.predict(X=session2[[x_var]])
    # Insertamos la columna de predicciones en el DataFrame
    session2_copy = session2.copy()
    session2_copy.insert(0,'predicciones',y_pred_session2)
    # Corrobaremos cual es el coeficiente de Determinacion de nuestro modelo (R_Cuadrada)
    coef_deter_session2 = model_session2.score(X=vars_indep_session2, y=var_dep_session2)
    # Corroboramos cual es el coeficiente de correlacion de nuestro modelo (R)
    coef_correl_session2 = np.sqrt(coef_deter_session2)

    # Sesion 3
    # Declaramos las variables dependientes e independientes para la regresion lineal
    vars_indep_session3 = session3[[x_var]]
    var_dep_session3 = session3[y_var]
    # Generamos el Modelo Lineal
    model_session3 = LinearRegression()
    # Ajustamos el modelo con las variables antes declaradas
    model_session3.fit(X=vars_indep_session3, y=var_dep_session3)
    # Predecimos los valores a partir de la variable
    y_pred_session3 = model_session3.predict(X=session3[[x_var]])
    # Insertamos la columna de predicciones en el DataFrame
    session3_copy = session3.copy()
    session3_copy.insert(0,'predicciones',y_pred_session3)
    # Corrobaremos cual es el coeficiente de Determinacion de nuestro modelo (R_Cuadrada)
    coef_deter_session3 = model_session3.score(X=vars_indep_session3, y=var_dep_session3)
    # Corroboramos cual es el coeficiente de correlacion de nuestro modelo (R)
    coef_correl_session3 = np.sqrt(coef_deter_session3)

    # Sesion 4
    # Declaramos las variables dependientes e independientes para la regresion lineal
    vars_indep_session4 = session4[[x_var]]
    var_dep_session4 = session4[y_var]
    # Generamos el Modelo Lineal
    model_session4 = LinearRegression()
    # Ajustamos el modelo con las variables antes declaradas
    model_session4.fit(X=vars_indep_session4, y=var_dep_session4)
    # Predecimos los valores a partir de la variable
    y_pred_session4 = model_session4.predict(X=session4[[x_var]])
    # Insertamos la columna de predicciones en el DataFrame
    session4_copy = session4.copy()
    session4_copy.insert(0,'predicciones',y_pred_session4)
    # Corrobaremos cual es el coeficiente de Determinacion de nuestro modelo (R_Cuadrada)
    coef_deter_session4 = model_session4.score(X=vars_indep_session4, y=var_dep_session4)
    # Corroboramos cual es el coeficiente de correlacion de nuestro modelo (R)
    coef_correl_session4 = np.sqrt(coef_deter_session4)

    st.subheader(f'Regresión Lineal - {y_var} VS {x_var}')

    # Mostrar Coeficientes 
    coef1, coef2, coef3, coef4 = st.columns(4)

    with coef1:
        st.text('Coef Sesion 1:')
        st.text(coef_correl_session1)

    with coef2:
        st.text('Coef Sesion 2:')
        st.text(coef_correl_session2)

    with coef3:
        st.text('Coef Sesion 3:')
        st.text(coef_correl_session3)

    with coef4:
        st.text('Coef Sesion 4:')
        st.text(coef_correl_session4)

    # Inicializar el gráfico en coordenadas polares
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(20, 10))

    # Sesion 1
    axs[0,0].scatter(session1_copy[x_var], session1_copy[y_var], color='skyblue',label='Valores')
    axs[0,0].scatter(session1_copy[x_var], session1_copy['predicciones'], label='Predicciones')
    axs[0,0].set_title(f'{y_var} vs {x_var} - Sesión 1',fontsize=16)
    axs[0,0].set_xlabel(f'{x_var}',fontsize=12)
    axs[0,0].set_ylabel(f'{y_var}',fontsize=12)
    axs[0,0].legend()

    # Sesion 2
    axs[0,1].scatter(session2_copy[x_var], session2_copy[y_var], color='skyblue', label='Valores')
    axs[0,1].scatter(session2_copy[x_var], session2_copy['predicciones'], label='Predicciones')
    axs[0,1].set_title(f'{y_var} vs {x_var} - Sesión 2',fontsize=16)
    axs[0,1].set_xlabel(f'{x_var}',fontsize=12)
    axs[0,1].set_ylabel(f'{y_var}',fontsize=12)
    axs[0,1].legend()

    # Sesion 3
    axs[1,0].scatter(session3_copy[x_var], session3_copy[y_var], color='skyblue', label='Valores')
    axs[1,0].scatter(session3_copy[x_var], session3_copy['predicciones'], label='Predicciones')
    axs[1,0].set_title(f'{y_var} vs {x_var} -  Sesión 3',fontsize=16)
    axs[1,0].set_xlabel(f'{x_var}',fontsize=12)
    axs[1,0].set_ylabel(f'{y_var}',fontsize=12)
    axs[1,0].legend()

    # Sesion 4
    axs[1,1].scatter(session4_copy[x_var], session4_copy[y_var], color='skyblue', label='Valores')
    axs[1,1].scatter(session4_copy[x_var], session4_copy['predicciones'], label='Predicciones')
    axs[1,1].set_title(f'{y_var} vs {x_var} - Sesión 4',fontsize=16)
    axs[1,1].set_xlabel(f'{x_var}',fontsize=12)
    axs[1,1].set_ylabel(f'{y_var}',fontsize=12)
    axs[1,1].legend()

    st.pyplot(fig)

# ------------------------------------------------------------ DASHBOARD ------------------------------------------------------------

# -------------------- Sidebar --------------------
# Configuracion del Sidebar
st.markdown(  
    """  
    <style>  
        section[data-testid="stSidebar"] {  
            width:230px !important; /* Set the width to your desired value */  
        }  
    </style>  
    """,  
    unsafe_allow_html=True,  
)  

# Creacion del Sidebar
with st.sidebar:
    st.title(':gear: Configuración')
    # Creacion del SelectBox para el Paciente
    paciente_seleccionado = st.sidebar.selectbox(":hospital: Seleccionar paciente", ["Paciente 7", "Paciente 8", "Paciente 11"])

# ---------- Metodo para obtener los datos respecto al paciente ----------
# Modificar los nombres del archivo segun los que se encuentren en la carpeta 'data'
if paciente_seleccionado == 'Paciente 7':
    session1 = load_data('data/P07/12dic03/P07_2018_12dic03_v07_atribMOVPREFAE_etiq.csv') 
    session2 = load_data('data/P07/12dic10/P07_2018_12dic10_v07_atribMOVPREFAE_etiq.csv')
    session3 = load_data('data/P07/12dic17/P07_2018_12dic17_v07_atribMOVPREFAE_etiq.csv')
    session4 = load_data('data/P07/01ene02/P07_2019_01ene02_v07_atribMOVPREFAE_etiq.csv')

elif paciente_seleccionado == 'Paciente 8':
    session1 = load_data('data/P08/01ene02/P08_2019_01ene02_v07_atribMOVPREFAE_etiq.csv')
    session2 = load_data('data/P08/01ene09/P08_2019_01ene09_v07_atribMOVPREFAE_etiq.csv')
    session3 = load_data('data/P08/01ene18/P08_2019_01ene18_v07_atribMOVPREFAE_etiq.csv')
    session4 = load_data('data/P08/01ene21/P08_2019_01ene21_v07_atribMOVPREFAE_etiq.csv')

elif paciente_seleccionado == 'Paciente 11':
    session1 = load_data('data/P11/04abr11/P11_2019_04abr11_v07_atribMOVPREFAE_etiq.csv')
    session2 = load_data('data/P11/04abr17/P11_2019_04abr17_v07_atribMOVPREFAE_etiq.csv')
    session3 = load_data('data/P11/04abr29/P11_2019_04abr29_v07_atribMOVPREFAE_etiq.csv')
    session4 = load_data('data/P11/05may13/P11_2019_05may13_v07_atribMOVPREFAE_etiq.csv')

# -------------------- Header --------------------
# Columnas para el Header
h1, h2, h3 = st.columns([0.80,0.10,0.10])

# Titulo | Imagen1 | Imagen2
with h1:
    st.title('Rehabilitación Virtual')
with h2:
    st.image('resources/TEC.png',width=70)
with h3:
    st.image('resources/INAOE.png',width=70)

# Línea divisoria visual ----------------------------------------
#st.markdown('---')
st.markdown('<hr style="border: 1px solid #96c2db; background-color: #96c2db; height: 3px;">', unsafe_allow_html=True)

# Generador de Pestanias 
tab1, tab2, tab3, tab4, tab5 = st.tabs([':chart_with_upwards_trend: Variables VS Tiempo',':signal_strength: Distancias Recorridas',':bar_chart: Cruces en Cero / Sumatorias',':no_mouth: Distribución de Emociones',':twisted_rightwards_arrows: Regresión y Correlaciones'])

# ---------- Pestania 1: Variables VS Tiempo ----------
with tab1:
    # Columnas para mostrar SelecBox pequenio
    varcol, space1, space2 = st.columns(3)

    with varcol:
        # Selector de Variable a graficar
        variables = session1.drop('time',axis=1).select_dtypes(include=['number']).columns
        selected_var = st.selectbox('Selecciona una variable para graficar', options=variables)

    # Checkbox para mostrar datos descriptivos de variable
    desc = st.checkbox(':clipboard: Mostrar Datos Descriptivos de la Variable')

    # Mostrar Datos Estadisticos
    if desc:
        st.subheader(f"Análisis Estadístico de {selected_var} ")

        # Sesion 1
        st.text('Sesión 1')
        stats1 = session1[[selected_var]].describe().T
        st.table(stats1)

        # Sesion 2
        st.text('Sesión 1')
        stats2 = session2[[selected_var]].describe().T
        st.table(stats2)

        # Sesion 3
        st.text('Sesión 3')
        stats3 = session3[[selected_var]].describe().T
        st.table(stats3)

        # Sesion 4
        st.text('Sesión 4')
        stats4 = session4[[selected_var]].describe().T
        st.table(stats4)

    # Columnas de Pestania
    tab1_colA, tab1_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab1_colA:
        
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header(f'Variable VS Tiempo - {paciente_seleccionado}')

        st.subheader(f'{selected_var} VS Tiempo')

        # Graficar Grafico de Lineas
        line_graph(session1,session2,session3,session4,selected_var)

    # ---------- Columna B ----------
    with tab1_colB:

        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)
        
        st.header(f'Tiempo para Alcanzar Máximo - {paciente_seleccionado}')

        st.subheader(f'Tiempo para alcanzar {selected_var} Max')

        time_var_max = pd.DataFrame([session1['time'].iloc[session1[selected_var].idxmax()],session2['time'].iloc[session2[selected_var].idxmax()],session3['time'].iloc[session3[selected_var].idxmax()],session4['time'].iloc[session4[selected_var].idxmax()]],columns=['TimeForMax'],index=['session1','session2','session3','session4'])

        # Graficar Grafico de Barras
        bar_graph(time_var_max,'TimeMax')

# ---------- Pestania 2: Distancias Recorridas ----------
with tab2:
    # Selector de Grafico a visualizar
    graph_selected = st.radio('Seleccione el Tipo de Grafico que visualizar:',options=[':bookmark_tabs: Histograma',':scooter: Densidad'])

    # Collumnas de Pestania
    tab2_colA, tab2_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab2_colA:

        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header(f'Distancias - {paciente_seleccionado}')

        # Condicional de Graficas
        # ---------- Histograma ----------
        if graph_selected == ':bookmark_tabs: Histograma':
            st.subheader(f'Histograma de Distancias por Sesión')

            # Graficar Histograma
            histogram(session1,session2,session3,session4)

        # ---------- Densidad ----------
        elif graph_selected == ':scooter: Densidad':
            st.subheader(f'Densidad de Distancias por Sesión')
            
            # Graficar Grafica de Densidad
            density(session1,session2,session3,session4)

    # ---------- Columna B ----------
    with tab2_colB:

        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header(f'Distancias Maximas - {paciente_seleccionado}')

        st.subheader('Distancias Maximas Alcanzadas por Sesión')

        # Obtener Distancias Maximas por Sesion
        dists_max = pd.DataFrame([session1['Distancia'].sum(),session2['Distancia'].sum(),session3['Distancia'].sum(),session4['Distancia'].sum()],columns=['DistMax'],index=['session1','session2','session3','session4'])

        # Graficar Grafico de barras
        bar_graph(dists_max,'DistMax')

# ---------- Pestania 3: Cruces en Cero / Sumatorias ----------
with tab3:
    # Seleccionar opcion a Graficar
    selected_option = st.radio('Seleccione una Opción a Mostrar',options=[':zero: Cruces en Cero',':heavy_plus_sign: Sumatoria'])

    # ---------- Cruces en Cero ----------
    if selected_option == ':zero: Cruces en Cero':
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.subheader(f'Cruces en Cero - {paciente_seleccionado}')

        # Dataframes de cruces en cero
        session1_zero = count_zeros(session1)
        session2_zero = count_zeros(session2)
        session3_zero = count_zeros(session3)
        session4_zero = count_zeros(session4)

        # Ordenar los datos en orden descendente por 'Zero_Crossings'
        sums1_sorted = session1_zero.sort_values(by='Zero_Crossings', ascending=False).reset_index(drop=True)
        sums2_sorted = session2_zero.sort_values(by='Zero_Crossings', ascending=False).reset_index(drop=True)
        sums3_sorted = session3_zero.sort_values(by='Zero_Crossings', ascending=False).reset_index(drop=True)
        sums4_sorted = session4_zero.sort_values(by='Zero_Crossings', ascending=False).reset_index(drop=True)

        # Calcular el porcentaje acumulativo
        sums1_sorted['Cumulative_Percentage'] = sums1_sorted['Zero_Crossings'].cumsum() / sums1_sorted['Zero_Crossings'].sum() * 100
        sums2_sorted['Cumulative_Percentage'] = sums2_sorted['Zero_Crossings'].cumsum() / sums2_sorted['Zero_Crossings'].sum() * 100
        sums3_sorted['Cumulative_Percentage'] = sums3_sorted['Zero_Crossings'].cumsum() / sums3_sorted['Zero_Crossings'].sum() * 100
        sums4_sorted['Cumulative_Percentage'] = sums4_sorted['Zero_Crossings'].cumsum() / sums4_sorted['Zero_Crossings'].sum() * 100

        # Crear la gráfica de Pareto
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

        # Crear la gráfica de barras
        plt.style.use('Solarize_Light2')  
        # Gráfica de barras
        colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

        # Sesion 1
        bars1 = axs[0,0].bar(sums1_sorted['Column'], sums1_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[0,0].set_xlabel('Column',fontsize=10)
        axs[0,0].set_ylabel('Zero Crossings', color='skyblue', fontsize=10)
        axs[0,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,0].set_xticks(sums1_sorted['Column'])
        axs[0,0].set_xticklabels(sums1_sorted['Column'], rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,0].plot(sums1_sorted['Column'], sums1_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,0].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[0,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,0].set_title('Cruces en Cero - Sesión 1', fontsize=14)

        # Sesion 2
        bars2 = axs[0,1].bar(sums2_sorted['Column'], sums2_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[0,1].set_xlabel('Column',fontsize=10)
        axs[0,1].set_ylabel('Zero Crossings', color='skyblue',fontsize=10)
        axs[0,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,1].set_xticks(sums1_sorted['Column'])
        axs[0,1].set_xticklabels(sums2_sorted['Column'], rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,1].plot(sums2_sorted['Column'], sums2_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,1].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[0,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,1].set_title('Cruces en Cero - Sesión 2',fontsize=14)

        # Sesion 3
        bars3 = axs[1,0].bar(sums3_sorted['Column'], sums3_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[1,0].set_xlabel('Column',fontsize=10)
        axs[1,0].set_ylabel('Zero Crossings', color='skyblue',fontsize=10)
        axs[1,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,0].set_xticks(sums1_sorted['Column'])
        axs[1,0].set_xticklabels(sums3_sorted['Column'], rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,0].plot(sums3_sorted['Column'], sums3_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,0].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[1,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,0].set_title('Cruces en Cero - Sesion 3',fontsize=14)

        # Sesion 4
        bars4 = axs[1,1].bar(sums4_sorted['Column'], sums4_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[1,1].set_xlabel('Column',fontsize=10)
        axs[1,1].set_ylabel('Zero Crossings', color='skyblue',fontsize=10)
        axs[1,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,1].set_xticks(sums1_sorted['Column'])
        axs[1,1].set_xticklabels(sums4_sorted['Column'], rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,1].plot(sums4_sorted['Column'], sums4_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,1].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[1,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,1].set_title('Cruces en Cero - Sesion 4',fontsize=14)

        plt.tight_layout()

        st.pyplot(fig)

    # ---------- Sumatoria ----------
    elif selected_option == ':heavy_plus_sign: Sumatoria':
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.subheader(f'Sumatoria de Variables - {paciente_seleccionado}')

        # Dataframes de sumatoria de variables
        session1_sums = count_sum(session1)
        session2_sums = count_sum(session2)
        session3_sums = count_sum(session3)
        session4_sums = count_sum(session4)

        # Ordenar los datos en orden descendente por 'Sumatory'
        sums1_sorted = session1_sums.sort_values(by='Sumatory', ascending=False)
        sums2_sorted = session2_sums.sort_values(by='Sumatory', ascending=False)
        sums3_sorted = session3_sums.sort_values(by='Sumatory', ascending=False)
        sums4_sorted = session4_sums.sort_values(by='Sumatory', ascending=False)

        # Calcular el porcentaje acumulativo
        sums1_sorted['Cumulative_Percentage'] = sums1_sorted['Sumatory'].cumsum() / sums1_sorted['Sumatory'].sum() * 100
        sums2_sorted['Cumulative_Percentage'] = sums2_sorted['Sumatory'].cumsum() / sums2_sorted['Sumatory'].sum() * 100
        sums3_sorted['Cumulative_Percentage'] = sums3_sorted['Sumatory'].cumsum() / sums3_sorted['Sumatory'].sum() * 100
        sums4_sorted['Cumulative_Percentage'] = sums4_sorted['Sumatory'].cumsum() / sums4_sorted['Sumatory'].sum() * 100

        # Crear la gráfica de Pareto
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

        # Crear la gráfica de barras
        plt.style.use('Solarize_Light2')  
        # Gráfica de barras
        colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

        # Sesion 1
        bars1 = axs[0,0].bar(sums1_sorted.index, sums1_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[0,0].set_xlabel('Column',fontsize=10)
        axs[0,0].set_ylabel('Sumatory', color='skyblue',fontsize=10)
        axs[0,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,0].set_xticks(sums1_sorted.index)
        axs[0,0].set_xticklabels(sums1_sorted.index, rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,0].plot(sums1_sorted.index, sums1_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,0].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[0,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,0].set_title('Sumatoria - Sesión 1',fontsize=14)

        # Sesion 2
        bars2 = axs[0,1].bar(sums2_sorted.index, sums2_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[0,1].set_xlabel('Column',fontsize=10)
        axs[0,1].set_ylabel('Sumatory', color='skyblue',fontsize=10)
        axs[0,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,1].set_xticks(sums1_sorted.index)
        axs[0,1].set_xticklabels(sums2_sorted.index, rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,1].plot(sums2_sorted.index, sums2_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,1].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[0,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,1].set_title('Sumatoria - Sesión 2',fontsize=14)

        # Sesion 3
        bars3 = axs[1,0].bar(sums3_sorted.index, sums3_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[1,0].set_xlabel('Column',fontsize=10)
        axs[1,0].set_ylabel('Sumatory', color='skyblue',fontsize=10)
        axs[1,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,0].set_xticks(sums1_sorted.index)
        axs[1,0].set_xticklabels(sums3_sorted.index, rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,0].plot(sums3_sorted.index, sums3_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,0].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[1,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,0].set_title('Sumatoria - Sesion 3',fontsize=14)

        # Sesion 4
        bars4 = axs[1,1].bar(sums4_sorted.index, sums4_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[1,1].set_xlabel('Column',fontsize=10)
        axs[1,1].set_ylabel('Sumatory', color='skyblue',fontsize=10)
        axs[1,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,1].set_xticks(sums1_sorted.index)
        axs[1,1].set_xticklabels(sums4_sorted.index, rotation=45, ha='right',fontsize=8)
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,1].plot(sums4_sorted.index, sums4_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,1].set_ylabel('Cumulative Percentage (%)', color='orange',fontsize=10)
        axs[1,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,1].set_title('Sumatoria - Sesion 4',fontsize=14)

        plt.tight_layout()

        st.pyplot(fig)

# ---------- Pestania 4: Distribucion de Emociones ----------
with tab4:
    # Columnas de Pestania
    tab4_colA, tab4_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab4_colA:
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header(f'Emociones a traves del Tiempo - {paciente_seleccionado}')

        # Identificar las columnas de sentimientos
        sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

        # Graficar Grafico de Distribucion de Emociones
        emotions_dist(sentiment_columns,session1,session2,session3,session4)

    # ---------- Columna B ----------
    with tab4_colB:
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header(f'Tiempos Maximos por Emocion - {paciente_seleccionado}')
        
        # Obtener intervalos de emociones
        max_emotions_session1 = emotion_intervals(session1)
        max_emotions_session2 = emotion_intervals(session2)
        max_emotions_session3 = emotion_intervals(session3)
        max_emotions_session4 = emotion_intervals(session4)

        max_time_emotion = pd.DataFrame([max_emotions_session1,max_emotions_session2,max_emotions_session3,max_emotions_session4],columns=['01_C','02_A','03_D','04_M'],index=['Session1','Session2','Session3','Session4'])

        # Graficar Tiempos Maximos de Cada Emocion
        maxtemp_emotion(max_time_emotion)

# ---------- Pestania 5: Regresion y Correlaciones ----------
with tab5:
    # Columnas de Pestania
    tab5_colA, tab5_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab5_colA:
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header('Regresiones')

        numeric_columns = numeric_cols(session1)

        # Columnas para seleccionar variables
        var_option1, var_option2 = st.columns(2)

        with var_option1:
            y_var = st.selectbox('Variable Dependiente (Y)', options=numeric_columns)

        with var_option2:
            x_var = st.selectbox('Variable Independiente (X)', options=numeric_columns)

        # Espaciamineto para alineacion
        st.text('')
        st.text('')
        st.text('')
        st.text('')

        scatter_graph(x_var,y_var,session1,session2,session3,session4)

    # ---------- Columna B ----------
    with tab5_colB:
        # CSS para asegurar alturas consistentes de los encabezados
        st.markdown("""
            <style>
                h2 {  
                    height: 5px; /* Altura fija para los encabezados */
                
                    font-size: 2.5em; /* Size of the main title */  
                    color: #FFFFFF; /* White color for main title */  
                    margin-bottom: 80px; /* Margin below the title */  

                    text-align: center;
                }  

                h3 {  
                    height: 30px; /* Altura fija para los encabezados */

                    font-size: 1.5em; /* Size of secondary headers */  
                    color: #FFFFFF; /* White color for secondary headers */  
                    margin-bottom: 10px; /* Margin below the title */  

                    text-align: center;
                } 
            </style>
        """, unsafe_allow_html=True)

        st.header('Correlaciones')
    
        # Opciones para Mostrar
        option1, option2 = st.columns(2)
        with option1:
            selected_session = st.radio('Seleccione la Sesion',options=['Sesión 1','Sesión 2','Sesión 3','Sesión 4'])

        with option2:
            selected_type = st.radio('Seleccione como visualizar las Correlaciones',options=[':date: DataFrame',':iphone: HeatMap'])

        # Creacion de Correlaciones
        if selected_session == 'Sesión 1':
            corr, corr_abs = regresion(session1)
        elif selected_session == 'Sesión 2':
            corr, corr_abs = regresion(session2)
        elif selected_session == 'Sesión 3':
            corr, corr_abs = regresion(session3)
        elif selected_session == 'Sesión 4':
            corr, corr_abs = regresion(session4)

        if selected_type == ':date: DataFrame':
            st.subheader(f'Correlaciones Lineales - {selected_session}')
            #Mostramos el dataset
            st.text('')
            st.text('')
            st.write(corr)

        elif selected_type == ':iphone: HeatMap':
            st.subheader(f'Heat Map - {selected_session}')
            #Mostramos el Heat Map
            heatmap = px.imshow(corr_abs)
            st.plotly_chart(heatmap)  
