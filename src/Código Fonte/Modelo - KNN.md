# Modelo - KNN

## importando e instalando bibliotecas necessárias
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix


## carregando dataset
    df = pd.read_csv('/content/sample_data/heart_attack_dataset.csv')


## Processando os dados

## Mapeamento das features do tipo categóricas para numéricas, do tipo float, padrão americano

    Gender_map = {'Male': 1, 'Female': 0}
    Has_Diabetes_map = {'Yes': 1, 'No': 0}
    Smoking_Status_map = {' Never': 0, 'Former': 1, 'Current': 2}
    Chest_Pain_Type_map = {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 4}
    Treatment_map = {'Lifestyle Changes': 1, 'Angioplasty': 2, 'Coronary Artery Bypass Graft (CABG)': 3, 'Medication': 4}
    df['Gender'] = df['Gender'].map(Gender_map).astype(float)
    df['Has Diabetes'] = df['Has Diabetes'].map(Has_Diabetes_map).astype(float)
    df['Smoking Status'] = df['Smoking Status'].map(Smoking_Status_map).astype(float)
    df['Chest Pain Type'] = df['Chest Pain Type'].map(Chest_Pain_Type_map).astype(float)
    df['Treatment'] = df['Treatment'].map(Treatment_map).astype(float)

## Imputando valores NaN usando SimpleImputer

## Para este conjunto de dados, usamos 'median' para numérico e 'most_frequent' para categóricos

    num_cols = ['Age', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)']
    cat_cols = ['Gender', 'Has Diabetes', 'Smoking Status', 'Chest Pain Type']
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


## Dividindo os dados em conjuntos de treinamento e teste

    X = df[['Gender', 'Age', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)', 'Has Diabetes', 'Smoking Status', 'Chest Pain Type']]
    y = df['Treatment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


## Utilizando Standard Scaler para colocar os dados numa mesma escala
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

## Criando e treinando o modelo KNN

    knn_model = KNeighborsClassifier(n_neighbors=30)  # nesta linha poderemos ajustar o vizinhos se for necessário
    knn_model.fit(X_train, y_train)

## Predição do modelo KNN criado

    y_pred = knn_model.predict(X_test)

## Avaliação do KNN model criado com as métricas: acurácia, recall_score e f1_score

## acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the KNN model: {accuracy}")

## recall_score
    recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class
    print(f"Recall of the KNN model: {recall}")

## f1_score
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class
    print(f"F1-score of the KNN model: {f1}")

- ![image](/src/images/Resultado.PNG)


## Criando a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)


## Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=Treatment_map.keys(),
            yticklabels=Treatment_map.keys())
plt.title("Matriz de confusão do modelo KNN")
plt.xlabel("Predicted Treatment")
plt.ylabel("Actual Treatment")
plt.savefig("knn_matriz_confusão.png")  # poderemos salvar a figura gerada e jogar noutro software
plt.show()

- ![image](/src/images/MKNN.png)