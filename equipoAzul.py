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

# Funcion para la carga de datos de la Regresion Lineal Simple
def load_data_regresion(data):
    # Coeficientes de correlacion entre las variables
    corr_f = data.corr()

    # Valor absoluto de todas las correlaciones entre las variables
    corr_abs = abs(corr_f)

    return corr_f, corr_abs

# ------------------------------------------------------------ DASHBOARD ------------------------------------------------------------

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title(':gear: Configuración')
    paciente_seleccionado = st.sidebar.selectbox(":hospital: Seleccionar paciente", ["Paciente 7", "Paciente 8", "Paciente 11"])

# ---------- Metodo para obtener los datos respecto al paciente ----------
if paciente_seleccionado == 'Paciente 7':
    session1 = load_data('data/P07/P07_2018_12dic03_v07_atribMOVPREFAE_etiq.csv')
    session2 = load_data('data/P07/P07_2018_12dic10_v07_atribMOVPREFAE_etiq.csv')
    session3 = load_data('data/P07/P07_2018_12dic17_v07_atribMOVPREFAE_etiq.csv')
    session4 = load_data('data/P07/P07_2019_01ene02_v07_atribMOVPREFAE_etiq.csv')

elif paciente_seleccionado == 'Paciente 8':
    session1 = load_data('data/P08/P08_2019_01ene02_v07_atribMOVPREFAE_etiq.csv')
    session2 = load_data('data/P08/P08_2019_01ene09_v07_atribMOVPREFAE_etiq.csv')
    session3 = load_data('data/P08/P08_2019_01ene18_v07_atribMOVPREFAE_etiq.csv')
    session4 = load_data('data/P08/P08_2019_01ene21_v07_atribMOVPREFAE_etiq.csv')

elif paciente_seleccionado == 'Paciente 11':
    session1 = load_data('data/P11/P11_2019_04abr11_v07_atribMOVPREFAE_etiq.csv')
    session2 = load_data('data/P11/P11_2019_04abr17_v07_atribMOVPREFAE_etiq.csv')
    session3 = load_data('data/P11/P11_2019_04abr29_v07_atribMOVPREFAE_etiq.csv')
    session4 = load_data('data/P11/P11_2019_05may13_v07_atribMOVPREFAE_etiq.csv')

# -------------------- Header --------------------
# Columnas para el Header
h1, h2, h3 = st.columns([0.70,0.15,0.15])

# Titulo | Imagen1 | Imagen2
with h1:
    st.title('Rehabilitación Virtual')
with h2:
    st.image('resources/TEC.png')
with h3:
    st.image('resources/INAOE.png')

# Generador de Pestanias 
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Variables VS Tiempo','Distancias Recorridas','Cruces en Cero / Sumatorias','Distribución de Emociones','Regresión y Correlaciones'])

# ---------- Pestania 1: Variables VS Tiempo ----------
with tab1:
    # Selector de Variable a graficar
    variables = session1.drop('time',axis=1).select_dtypes(include=['number']).columns
    selected_var = st.selectbox('Selecciona una variable para graficar', options=variables)

    # Columnas de Pestania
    tab1_colA, tab1_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab1_colA:
        st.header(f'Variable VS Tiempo - {paciente_seleccionado}')

        st.subheader(f'{selected_var} VS Tiempo')

        # Generador de grafica
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

        # Rotar las etiquetas del eje X
        plt.xticks(rotation=45)

        # Sesion 1
        session1.plot(x='time',y=selected_var, ax=axs[0,0])        

        # Añadir etiquetas y título
        axs[0,0].set_xlabel("Tiempo", fontsize=12)
        axs[0,0].set_ylabel(f"{selected_var}", fontsize=12)
        axs[0,0].set_title(f'{selected_var} VS Tiempo - SESION 1', fontsize=14)

        # Sesion 2
        session2.plot(x='time',y=selected_var, ax=axs[0,1])        

        # Añadir etiquetas y título
        axs[0,1].set_xlabel("Tiempo", fontsize=12)
        axs[0,1].set_ylabel(f"{selected_var}", fontsize=12)
        axs[0,1].set_title(f'{selected_var} VS Tiempo - SESION 2', fontsize=14)

        # Sesion 3
        session3.plot(x='time',y=selected_var, ax=axs[1,0])        

        # Añadir etiquetas y título
        axs[1,0].set_xlabel("Tiempo", fontsize=12)
        axs[1,0].set_ylabel(f"{selected_var}", fontsize=12)
        axs[1,0].set_title(f'{selected_var} VS Tiempo - SESION 3', fontsize=14)

        # Sesion 4
        session4.plot(x='time',y=selected_var, ax=axs[1,1])        

        # Añadir etiquetas y título
        axs[1,1].set_xlabel("Tiempo", fontsize=12)
        axs[1,1].set_ylabel(f"{selected_var}", fontsize=12)
        axs[1,1].set_title(f'{selected_var} VS Tiempo - SESION 4', fontsize=14)

        st.pyplot(fig)

    # ---------- Columna B ----------
    with tab1_colB:
        st.header(f'Tiempo para Alcanzar Máximo - {paciente_seleccionado}')

        st.subheader(f'Tiempo para alcanzar {selected_var} Max')

        time_var_max = pd.DataFrame([session1['time'].iloc[session1[selected_var].idxmax()],session2['time'].iloc[session2[selected_var].idxmax()],session3['time'].iloc[session3[selected_var].idxmax()],session4['time'].iloc[session4[selected_var].idxmax()]],columns=['TimeForMax'],index=['session1','session2','session3','session4'])

        fig, ax = plt.subplots(figsize=(20,10))

        # Crear la gráfica de barras
        plt.style.use('classic')  
        # Gráfica de barras
        colors = plt.get_cmap("Set2").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

        bars = plt.bar(time_var_max.index, time_var_max['TimeForMax'], color=colors[:len(time_var_max)])
        # Rotar las etiquetas del eje X
        plt.xticks(rotation=45)

        # Añadir etiquetas y título
        plt.xlabel("Sesiones", fontsize=12)
        plt.ylabel(f"Tiempo para {selected_var} Maxima", fontsize=12)
        plt.title(f"Tiempo que transcurre en cada Sesion para alcanzar la {selected_var} Maxima - {paciente_seleccionado}", fontsize=20)

        # Mostrar el valor encima de cada barra
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{bar.get_height():.1f}',
                ha='center', va='bottom'
            )

        st.pyplot(fig)

# ---------- Pestania 2: Distancias Recorridas ----------
with tab2:
    # Selector de Grafico a visualizar
    graph_selected = st.radio('Seleccione el Tipo de Grafico que visualizar:',options=['Histograma','Densidad'])

    # Collumnas de Pestania
    tab2_colA, tab2_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab2_colA:
        st.header(f'Distancias - {paciente_seleccionado}')

        # Condicional de Graficas
        # ---------- Histograma ----------
        if graph_selected == 'Histograma':
            st.subheader(f'{graph_selected} de Distancias por Sesión')
            
            # Generador de Graficas
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

            # Sesion 1
            axs[0,0].hist(session1['Distancia'], alpha=0.5, edgecolor='black', bins=10)
            # Anaidir Etiquetas
            axs[0,0].set_title('Histograma Distancia - Sesion 1', fontsize=14)
            axs[0,0].set_xlabel('Distancia', fontsize=12)
            axs[0,0].set_ylabel('Frecuencia', fontsize=12)

            # Sesion 2
            axs[0,1].hist(session2['Distancia'], alpha=0.5, edgecolor='black', bins=10)
            # Anaidir Etiquetas
            axs[0,1].set_title('Histograma Distancia - Sesion 2', fontsize=14)
            axs[0,1].set_xlabel('Distancia', fontsize=12)
            axs[0,1].set_ylabel('Frecuencia', fontsize=12)

            # Sesion 3
            axs[1,0].hist(session3['Distancia'], alpha=0.5, edgecolor='black', bins=10)
            # Anaidir Etiquetas
            axs[1,0].set_title('Histograma Distancia - Sesion 3', fontsize=14)
            axs[1,0].set_xlabel('Distancia', fontsize=12)
            axs[1,0].set_ylabel('Frecuencia', fontsize=12)

            # Sesion 4
            axs[1,1].hist(session4['Distancia'], alpha=0.5, edgecolor='black', bins=10)
            # Anaidir Etiquetas
            axs[1,1].set_title('Histograma Distancia - Sesion 4', fontsize=14)
            axs[1,1].set_xlabel('Distancia', fontsize=12)
            axs[1,1].set_ylabel('Frecuencia', fontsize=12)

            st.pyplot(fig)

        # ---------- Densidad ----------
        elif graph_selected == 'Densidad':
            st.subheader(f'{graph_selected} de Distancias por Sesión')

            # Generador de Graficas
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

            # Sesion 1
            session1['Distancia'].plot.density(ax=axs[0,0])
            # Anaidir Etiquetas
            axs[0,0].set_title('Gráfica Densidad - Sesion 1', fontsize=14)
            axs[0,0].set_xlabel('Intervalo', fontsize=12)
            axs[0,0].set_ylabel('Frecuencia', fontsize=12)

            # Sesion 2
            session2['Distancia'].plot.density(ax=axs[0,1])
            # Anaidir Etiquetas
            axs[0,1].set_title('Gráfica Densidad - Sesion 2', fontsize=14)
            axs[0,1].set_xlabel('Intervalo', fontsize=12)
            axs[0,1].set_ylabel('Frecuencia', fontsize=12)

            # Sesion 3
            session3['Distancia'].plot.density(ax=axs[1,0])
            # Anaidir Etiquetas
            axs[1,0].set_title('Gráfica Densidad - Sesion 3', fontsize=14)
            axs[1,0].set_xlabel('Intervalo', fontsize=12)
            axs[1,0].set_ylabel('Frecuencia', fontsize=12)

            # Sesion 4
            session4['Distancia'].plot.density(ax=axs[1,1])
            # Anaidir Etiquetas
            axs[1,1].set_title('Gráfica Densidad - Sesion 4', fontsize=14)
            axs[1,1].set_xlabel('Intervalo', fontsize=12)
            axs[1,1].set_ylabel('Frecuencia', fontsize=12)

            st.pyplot(fig)

    # ---------- Columna B ----------
    with tab2_colB:
        st.header(f'Distancias Maximas -  {paciente_seleccionado}')

        st.subheader('Distancias Maximas Alcanzadas por Sesión')

        # Obtener Distancias Maximas por Sesion
        dists_max = pd.DataFrame([session1['Distancia'].sum(),session2['Distancia'].sum(),session3['Distancia'].sum(),session4['Distancia'].sum()],columns=['DistMax'],index=['session1','session2','session3','session4'])

        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(10, 3))

        # Crear la gráfica de barras
        plt.style.use('classic')  
        # Gráfica de barras
        colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

        bars = plt.bar(dists_max.index, dists_max['DistMax'], color=colors[:len(dists_max)])
        # Rotar las etiquetas del eje X
        plt.xticks(rotation=45)

        # Añadir etiquetas y título
        plt.xlabel("Sesiones", fontsize=12)
        plt.ylabel("Distancia Maxima", fontsize=12)
        plt.title(f"Distancias Maximas de cada Sesion - {paciente_seleccionado}", fontsize=14)

        # Mostrar el valor encima de cada barra
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{bar.get_height():.1f}',
                ha='center', va='bottom'
            )

        st.pyplot(fig)

# ---------- Pestania 3: Cruces en Cero / Sumatorias ----------
with tab3:
    # Seleccionar opcion a Graficar
    selected_option = st.radio('Seleccione una Opción a Mostrar',options=['Cruces en Cero','Sumatoria'])

    # ---------- Cruces en Cero ----------
    if selected_option == 'Cruces en Cero':
        st.header(f'Cruces en Cero - {paciente_seleccionado}')

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

        # Sesion 1
        bars1 = axs[0,0].bar(sums1_sorted['Column'], sums1_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[0,0].set_xlabel('Column')
        axs[0,0].set_ylabel('Zero Crossings', color='skyblue')
        axs[0,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,0].set_xticklabels(sums1_sorted['Column'], rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,0].plot(sums1_sorted['Column'], sums1_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,0].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[0,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,0].set_title('Cruces en Cero - Sesión 1')

        # Sesion 2
        bars2 = axs[0,1].bar(sums2_sorted['Column'], sums2_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[0,1].set_xlabel('Column')
        axs[0,1].set_ylabel('Zero Crossings', color='skyblue')
        axs[0,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,1].set_xticklabels(sums2_sorted['Column'], rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,1].plot(sums2_sorted['Column'], sums2_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,1].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[0,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,1].set_title('Cruces en Cero - Sesión 2')

        # Sesion 3
        bars3 = axs[1,0].bar(sums3_sorted['Column'], sums3_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[1,0].set_xlabel('Column')
        axs[1,0].set_ylabel('Zero Crossings', color='skyblue')
        axs[1,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,0].set_xticklabels(sums3_sorted['Column'], rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,0].plot(sums3_sorted['Column'], sums3_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,0].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[1,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,0].set_title('Cruces en Cero - Sesion 3')

        # Sesion 4
        bars4 = axs[1,1].bar(sums4_sorted['Column'], sums4_sorted['Zero_Crossings'], color='skyblue', label='Sumatory')
        axs[1,1].set_xlabel('Column')
        axs[1,1].set_ylabel('Zero Crossings', color='skyblue')
        axs[1,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,1].set_xticklabels(sums4_sorted['Column'], rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,1].plot(sums4_sorted['Column'], sums4_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,1].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[1,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,1].set_title('Cruces en Cero - Sesion 4')

        st.pyplot(fig)

    # ---------- Sumatoria ----------
    elif selected_option == 'Sumatoria':
        st.header(f'Sumatoria de Variables - {paciente_seleccionado}')

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

        # Sesion 1
        bars1 = axs[0,0].bar(sums1_sorted.index, sums1_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[0,0].set_xlabel('Column')
        axs[0,0].set_ylabel('Sumatory', color='skyblue')
        axs[0,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,0].set_xticklabels(sums1_sorted.index, rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,0].plot(sums1_sorted.index, sums1_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,0].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[0,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,0].set_title('Sumatoria - Sesión 1')

        # Sesion 2
        bars2 = axs[0,1].bar(sums2_sorted.index, sums2_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[0,1].set_xlabel('Column')
        axs[0,1].set_ylabel('Sumatory', color='skyblue')
        axs[0,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[0,1].set_xticklabels(sums2_sorted.index, rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[0,1].plot(sums2_sorted.index, sums2_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[0,1].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[0,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[0,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[0,1].set_title('Sumatoria - Sesión 2')

        # Sesion 3
        bars3 = axs[1,0].bar(sums3_sorted.index, sums3_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[1,0].set_xlabel('Column')
        axs[1,0].set_ylabel('Sumatory', color='skyblue')
        axs[1,0].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,0].set_xticklabels(sums3_sorted.index, rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,0].plot(sums3_sorted.index, sums3_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,0].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[1,0].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,0].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,0].set_title('Sumatoria - Sesion 3')

        # Sesion 4
        bars4 = axs[1,1].bar(sums4_sorted.index, sums4_sorted['Sumatory'], color='skyblue', label='Sumatory')
        axs[1,1].set_xlabel('Column')
        axs[1,1].set_ylabel('Sumatory', color='skyblue')
        axs[1,1].tick_params(axis='y', labelcolor='skyblue')
        # Rotar las etiquetas del eje X
        axs[1,1].set_xticklabels(sums4_sorted.index, rotation=45, ha='right')
        # Crear un segundo eje Y para la línea de porcentaje acumulado
        axs[1,1].plot(sums4_sorted.index, sums4_sorted['Cumulative_Percentage'], color='orange', marker='o', linestyle='-', linewidth=2, label='Cumulative Percentage')
        axs[1,1].set_ylabel('Cumulative Percentage (%)', color='orange')
        axs[1,1].tick_params(axis='y', labelcolor='orange')
        # Agregar línea de referencia en 80%
        axs[1,1].axhline(80, color='grey', linestyle='--', linewidth=0.8)
        axs[1,1].set_title('Sumatoria - Sesion 4')

        st.pyplot(fig)

# ---------- Pestania 4: Distribucion de Emociones ----------
with tab4:
    # Columnas de Pestania
    tab4_colA, tab4_colB = st.columns(2)

    # ---------- Columna A ----------
    with tab4_colA:
        st.header(f'Emociones a traves del Tiempo - {paciente_seleccionado}')

        # Identificar las columnas de sentimientos
        sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

        # Crear la figura y los ejes
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20,10))

        # Lista de Ejes
        axis = [0,1,2,3]

        # Graficar cada eje
        for ax in axis:
            # Graficar cada sentimiento
            for i, col in enumerate(sentiment_columns):
                # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
                intervals = []
                start = None
                for j in range(len(session4)):
                    if session4[col].iloc[j] == 1:
                        if start is None:
                            start = session4['time'].iloc[j]
                    else:
                        if start is not None:
                            end = session4['time'].iloc[j - 1]
                            intervals.append((start, end))
                            start = None
                # Capturar el último intervalo si termina en el último valor de la columna
                if start is not None:
                    intervals.append((start, session4['time'].iloc[-1]))

                # Dibujar barras horizontales para los intervalos
                for (start, end) in intervals:
                    axs[ax].hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

        # Etiquetas y formato
        axs[0].set_yticks(range(len(sentiment_columns)))
        axs[0].set_yticklabels(sentiment_columns)
        axs[0].set_xlabel("Tiempo")
        axs[0].set_ylabel("Emociones")
        axs[0].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 1")

        # Etiquetas y formato
        axs[1].set_yticks(range(len(sentiment_columns)))
        axs[1].set_yticklabels(sentiment_columns)
        axs[1].set_xlabel("Tiempo")
        axs[1].set_ylabel("Emociones")
        axs[1].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 2")

        # Etiquetas y formato
        axs[2].set_yticks(range(len(sentiment_columns)))
        axs[2].set_yticklabels(sentiment_columns)
        axs[2].set_xlabel("Tiempo")
        axs[2].set_ylabel("Emociones")
        axs[2].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 3")

        # Etiquetas y formato
        axs[3].set_yticks(range(len(sentiment_columns)))
        axs[3].set_yticklabels(sentiment_columns)
        axs[3].set_xlabel("Tiempo")
        axs[3].set_ylabel("Emociones")
        axs[3].set_title("Presencia de sentimientos en intervalos de tiempo - Sesion 4")

        plt.tight_layout()

        plt.grid(axis='x', linestyle='--', alpha=0.7)

        st.pyplot(fig)

    # ---------- Columna B ----------
    with tab4_colB:
        st.header(f'Tiempos Maximos por Emocion - {paciente_seleccionado}')
        
        # Obtener intervalos de emociones
        max_emotions_session1 = emotion_intervals(session1)
        max_emotions_session2 = emotion_intervals(session2)
        max_emotions_session3 = emotion_intervals(session3)
        max_emotions_session4 = emotion_intervals(session4)

        max_time_emotion = pd.DataFrame([max_emotions_session1,max_emotions_session2,max_emotions_session3,max_emotions_session4],columns=['01_C','02_A','03_D','04_M'],index=['Session1','Session2','Session3','Session4'])

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

        # Dibujar la línea del gráfico
        axs[0,0].plot(angles, emotion1_values, color='skyblue', linewidth=2, linestyle='solid')
        # Rellenar el área
        axs[0,0].fill(angles, emotion1_values, color='skyblue', alpha=0.4)
        # Añadir etiquetas para cada categoría
        axs[0,0].set_xticks(angles[:-1])
        axs[0,0].set_xticklabels(max_time_emotion.index)
        # Ajustar los límites del eje radial y el título
        axs[0,0].set_ylim(0, max(emotion1_values))
        axs[0,0].set_title('                                                        CANSANCIO')

        # Dibujar la línea del gráfico
        axs[0,1].plot(angles, emotion2_values, color='skyblue', linewidth=2, linestyle='solid')
        # Rellenar el área
        axs[0,1].fill(angles, emotion2_values, color='skyblue', alpha=0.4)
        # Añadir etiquetas para cada categoría
        axs[0,1].set_xticks(angles[:-1])
        axs[0,1].set_xticklabels(max_time_emotion.index)
        # Ajustar los límites del eje radial y el título
        axs[0,1].set_ylim(0, max(emotion2_values))
        axs[0,1].set_title('                                                        ANSIEDAD')

        # Dibujar la línea del gráfico
        axs[1,0].plot(angles, emotion3_values, color='skyblue', linewidth=2, linestyle='solid')
        # Rellenar el área
        axs[1,0].fill(angles, emotion3_values, color='skyblue', alpha=0.4)
        # Añadir etiquetas para cada categoría
        axs[1,0].set_xticks(angles[:-1])
        axs[1,0].set_xticklabels(max_time_emotion.index)
        # Ajustar los límites del eje radial y el título
        axs[1,0].set_ylim(0, max(emotion3_values))
        axs[1,0].set_title('                                                        DOLOR')

        # Dibujar la línea del gráfico
        axs[1,1].plot(angles, emotion4_values, color='skyblue', linewidth=2, linestyle='solid')
        # Rellenar el área
        axs[1,1].fill(angles, emotion4_values, color='skyblue', alpha=0.4)
        # Añadir etiquetas para cada categoría
        axs[1,1].set_xticks(angles[:-1])
        axs[1,1].set_xticklabels(max_time_emotion.index)
        # Ajustar los límites del eje radial y el título
        axs[1,1].set_ylim(0, max(emotion4_values))
        axs[1,1].set_title('                                                        MOTIVACION')

        plt.tight_layout()

        st.pyplot(fig)

# ---------- Pestania 5: Regresion y Correlaciones ----------
with tab5:
    # Columnas de Pestania
    tab5_colA, tab5_colB = st.columns(2)

    # ---------- Columna A ----------
    

    # ---------- Columna B ----------
