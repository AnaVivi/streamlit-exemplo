# Deploy de Aplicações Preditivas com Streamlit

# Imports
import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

##### Programando a Barra Superior da Aplicação Web #####

# Título
st.write("*Formação Engenheiro de Machine Learning*")
st.write("*Deploy de Modelos de Machine Learning*")
st.write("*Deploy de Aplicações Preditivas com Streamlit*")
st.title("Regressão Logística")

##### Programando a Barra Lateral de Navegação da Aplicação Web #####

# Cabeçalho lateral
st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Selecione o Dataset Desejado**""")
Dataset = st.sidebar.selectbox('Dataset',('Iris', 'Wine', 'Breast Cancer'))
Split = st.sidebar.slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 70/30):', 0.1, 0.9, 0.70)
st.sidebar.markdown("""**Selecione os Hiperparâmetros Para o Modelo de Regressão Logística**""")
Solver = st.sidebar.selectbox('Algoritmo', ('lbfgs', 'newton-cg', 'liblinear', 'sag'))
Penality = st.sidebar.radio("Regularização:", ('none', 'l1', 'l2', 'elasticnet'))
Tol = st.sidebar.text_input("Tolerância Para Critério de Parada (default = 1e-4):", "1e-4")
Max_Iteration = st.sidebar.text_input("Número de Iterações (default = 50):", "50")

# Dicionário Para os Hiperparâmetros
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
parameters = { 'Penality':Penality, 'Tol':Tol, 'Max_Iteration':Max_Iteration, 'Solver':Solver }
       
##### Funções Para Carregar e Preparar os Dados #####

# Função para carregar o dataset
def carrega_dataset(dataset):

    # Carrega o dataset
    if dataset == 'Iris':
        dados = sklearn.datasets.load_iris()
    elif dataset == 'Wine':
         dados = sklearn.datasets.load_wine()
    elif dataset == 'Breast Cancer':
         dados = sklearn.datasets.load_breast_cancer()
    
    return dados

# Função para preparar os dados e fazer a divisão em treino e teste
def prepara_dados(dados, split):

    # Divide os dados de acordo com o valor de split definido pelo usuário
    X_treino, X_teste, y_treino, y_teste = train_test_split(dados.data, dados.target, test_size = float(split), random_state = 42)

    # Prepara o scaler para padronização
    scaler = MinMaxScaler()

    # Fit e transform nos dados de treino
    X_treino = scaler.fit_transform(X_treino)

    # Apenas transform nos dados de teste
    X_teste = scaler.transform(X_teste)

    return (X_treino, X_teste, y_treino, y_teste)

##### Função Para o Modelo de Machine Learning #####  

# Função para o modelo
def cria_modelo(parameters):
    
    # Extrai os dados de treino e teste
    X_treino, X_teste, y_treino, y_teste = prepara_dados(Data, Split) 

    # Cria o modelo
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = LogisticRegression(penalty = parameters['Penality'], 
                             solver = parameters['Solver'], 
                             max_iter = int(parameters['Max_Iteration']), 
                             tol = float(parameters['Tol']))

    # Treina o modelo
    clf = clf.fit(X_treino, y_treino)

    # Faz previsões
    prediction = clf.predict(X_teste)
    
    # Calcula a acurácia
    accuracy = sklearn.metrics.accuracy_score(y_teste, prediction)

    # Calcula a confusion matrix
    cm = confusion_matrix(y_teste, prediction)

    # Dicionário com os resultados
    dict_value = {"modelo":clf, "acuracia": accuracy, "previsao":prediction, "y_real": y_teste, "Metricas":cm, "X_teste": X_teste }
       
    return(dict_value)
    
    return(X_treino, X_teste, y_treino, y_teste)

##### Programando o Corpo da Aplicação Web ##### 

# Resumo dos dados
st.markdown("""Resumo dos Dados""")
st.write("Nome do Dataset:", Dataset)

# Carrega o dataset escolhido pelo usuário
Data = carrega_dataset(Dataset)

# Extrai a variável alvo
targets = Data.target_names

# Prepara o dataframe com os dados
Dataframe = pd.DataFrame (Data.data, columns = Data.feature_names)
Dataframe['target'] = pd.Series(Data.target)
Dataframe['target labels'] = pd.Series(targets[i] for i in Data.target)

# Mostra o dataset selecionado pelo usuário
st.write("Visão Geral dos Atributos:")
st.write(Dataframe)

##### Programando o Botão de Ação ##### 

if(st.sidebar.button("Clique Para Treinar o Modelo de Regressão Logística")):
    
    # Barra de progressão
    with st.spinner('Carregando o Dataset...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Dataset Carregado!")
    
    # Cria e treina o modelo
    modelo = cria_modelo(parameters) 
    
    # Barra de progressão
    my_bar = st.progress(0)

    # Mostra a barra de progressão com percentual de conclusão
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    # Info para o usuário
    with st.spinner('Treinando o Modelo...'):
        time.sleep(1)

    # Info de sucesso
    st.success("Modelo Treinado") 

    # Extrai os labels reais
    labels_reais = [targets[i] for i in modelo["y_real"]]

    # Extrai os labels previstos
    labels_previstos = [targets[i] for i in modelo["previsao"]]

    # Sub título
    st.subheader("Previsões do Modelo nos Dados de Teste")

    # Mostra o resultado
    st.write(pd.DataFrame({"Valor Real" : modelo["y_real"], 
                           "Label Real" : labels_reais, 
                           "Valor Previsto" : modelo["previsao"], 
                           "Label Previsto" :  labels_previstos,}))
    
    # Extrai as métricas
    matriz = modelo["Metricas"]

    # Sub título
    st.subheader("Matriz de Confusão nos Dados de Teste")

    # Mostra a matriz de confusão
    st.write(matriz)

    # Mostra a acurácia
    st.write("Acurácia do Modelo:", modelo["acuracia"])

    # Obrigado
    st.write("Obrigado por usar esta app do Streamlit!")




