# Preparação dos dados

# Modelo Árvore de Decisão - Decision Tree

![image](https://github.com/user-attachments/assets/03d01f1b-d7b5-4947-bd1c-5bf9c5cd47ff)

![image](https://github.com/user-attachments/assets/f12ce3c5-50b4-473d-896e-065e7109421d)

![image](https://github.com/user-attachments/assets/b0522d26-32bf-4b51-bc9a-3d08803caccd)

![image](https://github.com/user-attachments/assets/790c593f-8d48-457c-a6b2-0a629726c0da)

![image](https://github.com/user-attachments/assets/ae4d8c4e-4311-410c-8c1e-e636b3359e74)

![image](https://github.com/user-attachments/assets/41d5e1cd-2ece-4237-bb25-ad064c9f04a6)

![image](https://github.com/user-attachments/assets/120fe0cb-ed1a-4538-be15-b39995c15b94)

![image](https://github.com/user-attachments/assets/8770c97d-8934-4c07-8cca-a68a77c4668e)

![image](https://github.com/user-attachments/assets/edcf24ee-f332-4bde-b029-c5cdd28d74fe)

![image](https://github.com/user-attachments/assets/9bb4dc7d-13e3-4841-93b8-323c85dcd55f)



# Modelo - Classificador Naive Bayes

## Importando Bibliotecas Necessárias

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt # for data visualization purposes
    import seaborn as sns # for statistical data visualization
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.impute import SimpleImputer
    import warnings

    warnings.filterwarnings('ignore')
    %matplotlib inline
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        import pandas as pd

## Colocando Dataset em uma variável

    df = pd.read_csv('/content/sample_data/heart_attack_dataset.csv')

## Mapeando, convertando e aplicando colunas categórias para númericas do tipo float, padrão americano

    Gender_map = {'Male': 1, 'Female': 0}
    Has_Diabetes_map = {'Yes': 1, 'No': 0}
    Smoking_Status_map = {' Never': 0, 'Former': 1, 'Current': 2}
    Chest_Pain_Type_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    Treatment_map = {'Lifestyle Changes': 0, 'Angioplasty': 1, 'Coronary Artery Bypass Graft (CABG)': 2, 'Medication': 3}

    df['Gender'] = df['Gender'].map(Gender_map).astype(float)
    df['Has Diabetes'] = df['Has Diabetes'].map(Has_Diabetes_map).astype(float)
    df['Smoking Status'] = df['Smoking Status'].map(Smoking_Status_map).astype(float)
    df['Chest Pain Type'] = df['Chest Pain Type'].map(Chest_Pain_Type_map).astype(float)
    df['Treatment'] = df['Treatment'].map(Treatment_map).astype(float)


## Categorizando colunas

    num_cols = ['Age', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)']
    cat_cols = ['Gender', 'Has Diabetes', 'Smoking Status', 'Chest Pain Type']

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


## Mostrando informações do sobre os dados


- ![image](/src/images/df.info().PNG)

_Fonte: Envolvidos do Projeto do Eixo 7_

## Definindo Colunas 

    col_names = ['Gender', 'Age', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)', 'Has Diabetes', 'Smoking Status', 'Chest Pain Type', 'Treatment']

    df.columns = col_names
    df.columns


## Encontrando Variáveis Categóricas

    categorical = [var for var in df.columns if df[var].dtype=='O']

    print('Há {} variáveis categóricas\n'.format(len(categorical)))

    print('As variáveis categóricas são :\n\n', categorical)

## Resultado 

- ![image](/src/images/VarCat.png)

_Fonte: Envolvidos do Projeto do Eixo 7_

## Declarando Var. Destino e Vetor de Características

    X = df.drop(['Treatment'], axis=1)

    y = df['Treatment']

## Declarando Var. Destino e Vetor de Características

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

## Resultado tamanho dos set de treinamento e de teste

    X_train.shape, X_test.shape

![image](/src/images/set.png)

_Fonte: Envolvidos do Projeto do Eixo 7_


## Como foi executado no início do código o mapeamento das variáveis, não há valores categóricos 

    categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

    categorical


    numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

    numerical


- ![image](/src/images/Tipos%20Dados.png)

_Fonte: Envolvidos do Projeto do Eixo 7_



## Feature Scaling(Determinando volume de recurso para treinametno)

    cols = X_train.columns


## Começando o procedimento para criação do modelo 

    from sklearn.naive_bayes import GaussianNB

## Instanciando o modelo

    gnb = GaussianNB()

## Ajustando Modelo 

    gnb.fit(X_train, y_train)


## Prevendo o resultado

    y_pred = gnb.predict(X_test) # Cada número é o tipo de treinamento realizado

##### Treatment_map ={'Lifestyle Changes': 1, 'Angioplasty': 2, 'Coronary Artery Bypass Graft (CABG)': 3, 'Medication': 4}

    y_pred

- ![image](/src/images/Prevendo%20Resultado.png)

_Fonte: Envolvidos do Projeto do Eixo 7_


## Importando métrica de acurácia e demonstrando resultado da acurácia

- ![image](/src/images/Acurácia.png)

_Fonte: Envolvidos do Projeto do Eixo 7_

## Treinamento de Y 

    y_pred_train = gnb.predict(X_train)

    y_pred_train

## Resultado dos treinamntos

    print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

    print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

![image](/src/images/Treinamento.png)

_Fonte: Envolvidos do Projeto do Eixo 7_

## Checando o score de acurácia nula


    null_accuracy = (7407/(7407+2362))

    print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


![image](/src/images/AcuráciaNula.png)

_Fonte: Envolvidos do Projeto do Eixo 7_

## Criando e plotando a matriz de confusão 

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)


    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=Treatment_map.keys(),
                yticklabels=Treatment_map.keys())
    plt.title("Matriz de confusão do modelo Naive Bayes")
    plt.xlabel("Predicted Treatment")
    plt.ylabel("Actual Treatment")
    plt.savefig("knn_matriz_confusão.png")  
    plt.show()

## Resultado 

![image](/src/images/MatrizConfusão%20NB.png)

_Fonte: Envolvidos do Projeto do Eixo 7_

# MODELO RANDOM FOREST CLASSFIER (Floreta Aleatória)

## Importando as bibliotecas

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.linear_model import LogisticRegression

## Importação e leitura do dataset AtaqueCardiaco
df = pd.read_csv('/content/datasetAtaqueCardiaco.csv')

df.head()

## Verificar se há dados faltantes
df.isnull().sum()

## Substituir valores ausentes ou eliminando colunas com dados ausentes
df = df.dropna()

## Codificar variáveis categóricas com LabelEncoder
label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male=1, Female=0
df['Has Diabetes'] = label_encoder.fit_transform(df['Has Diabetes'])  # Yes=1, No=0
df['Smoking Status'] = label_encoder.fit_transform(df['Smoking Status'])  # Never=0, Current=1, etc.
df['Chest Pain Type'] = label_encoder.fit_transform(df['Chest Pain Type'])  # Codificando tipos de dor
df['Treatment'] = label_encoder.fit_transform(df['Treatment'])  # Tratamentos como variável alvo

## Visualizando as primeiras linhas após o pré-processamento
df.head()

## Definindo as variáveis independentes (X) e dependente (y)
X = df.drop('Treatment', axis=1)  # Removendo a coluna 'Treatment' para ser a variável alvo
y = df['Treatment']  # Variável alvo

## Dividindir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Visualizando as dimensões dos dados de treino e teste
print(f"Tamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")

## Criar o modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

## Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

## Fazer previsões com os dados de teste
y_pred = model.predict(X_test)

## Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

## Relatório de classificação
print(classification_report(y_test, y_pred))

## Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

![image](https://github.com/user-attachments/assets/feb5d87c-5400-48e9-8230-d9066fb5142d)
![image](https://github.com/user-attachments/assets/c459df4f-6f66-40e2-983c-7d994447605d)

Fonte: Envolvidos do Projeto do Eixo 7



# Descrição dos modelos

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas.

Como parte da comprovação de construção dos modelos, um vídeo de demonstração com todas as etapas de pré-processamento e de execução dos modelos deverá ser entregue. Este vídeo poderá ser do tipo _screencast_ e é imprescindível a narração contemplando a demonstração de todas as etapas realizadas.

# Avaliação dos modelos criados

## Métricas utilizadas

Essas foram as métricas utilizadas:

Acurácia: é obtida pela divisão de quantidade de acertos e o total de entradas. Essa métrica foi utilizada na Árvore de Decisão, KNN, Naive Bayes e o Random Forest.
F1-score: Média calculada com base na precisão e revocação. Essa métrica foi utilizada na Árvore de Decisão.
Recall: Esta métrica avalia a capacidade do método detectar com sucesso resultados classificados como positivos. Foi utilizado para avaliar o modelo KNN.

A fonte de definicação está anexado na etapa 1.


## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos pelos modelos construídos, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecendo relação com os objetivos previamente propostos. 

# Pipeline de pesquisa e análise de dados

Em pesquisa e experimentação em sistemas de informação, um pipeline de pesquisa e análise de dados refere-se a um conjunto organizado de processos e etapas que um profissional segue para realizar a coleta, preparação, análise e interpretação de dados durante a fase de pesquisa e desenvolvimento de modelos. Esse pipeline é essencial para extrair _insights_ significativos, entender a natureza dos dados e, construir modelos de aprendizado de máquina eficazes. 

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar  os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".

Além disso, deverá ser entregue um vídeo onde deverão ser descritas todas as etapas realizadas. O vídeo, que não tem limite de tempo, deverá ser apresentado por **todos os integrantes da equipe**, de forma que, cada integrante tenha oportunidade de apresentar o que desenvolveu e as  percepções obtidas.


















