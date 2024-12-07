# Introdução

As doenças cardíacas permanecem como uma das principais causas de mortalidade tanto no Brasil quanto no mundo, representando um desafio significativo para os sistemas de saúde globais. Dados recentes indicam um aumento contínuo no número de óbitos relacionados a doenças cardiovasculares, atribuídos a fatores como envelhecimento populacional, hábitos de vida inadequados e exposição a poluentes ambientais. Este cenário reforça a necessidade de estratégias inovadoras e eficazes para a prevenção, diagnóstico e tratamento dessas condições de saúde.

Nesse contexto, o aprendizado de máquina surge como uma ferramenta promissora para aprimorar a detecção e o manejo de doenças cardíacas. Ao analisar grandes volumes de dados clínicos, algoritmos inteligentes podem identificar padrões complexos e prever com maior precisão o risco de ocorrência de eventos cardiovasculares em indivíduos, considerando uma ampla gama de fatores de risco e características pessoais. Essa abordagem permite uma intervenção médica mais precoce e personalizada, aumentando as chances de sucesso no tratamento e redução da mortalidade.

O projeto em questão tem como objetivo principal desenvolver um modelo preditivo utilizando técnicas avançadas de aprendizado de máquina para estimar o risco de doenças coronarianas. A partir da análise de um conjunto abrangente de dados públicos, busca-se identificar os principais fatores que influenciam o desenvolvimento dessas doenças e, consequentemente, recomendar os tratamentos mais adequados para cada perfil de paciente. Essa ferramenta pretende auxiliar profissionais de saúde na tomada de decisões clínicas mais informadas e precisas.

A justificativa para o desenvolvimento deste projeto baseia-se na urgência de soluções eficazes frente à crescente carga das doenças cardiovasculares na sociedade. A implementação de modelos preditivos avançados pode não apenas melhorar os desfechos clínicos dos pacientes, mas também otimizar a alocação de recursos no sistema de saúde, promovendo uma assistência mais eficiente e sustentável. Além disso, a adoção de tecnologias de inteligência artificial na área médica representa um passo significativo rumo à modernização e inovação dos cuidados em saúde.

O público-alvo principal desta iniciativa são os médicos e profissionais de saúde que atuam diretamente no diagnóstico e tratamento de doenças cardíacas. Ao fornecer uma ferramenta de suporte baseada em evidências e dados robustos, espera-se facilitar o processo de decisão clínica e melhorar a qualidade da assistência prestada aos pacientes. Indiretamente, pacientes e instituições de saúde também se beneficiam através de tratamentos mais eficazes e da potencial redução de custos associados ao manejo das doenças cardiovasculares.

## Problema

Mesmo com o avanço das tecnologias de aprendizado de máquina e das técnincas de análise de dados, ainda existem poucas ferramentas que auxiliam especialistas em doenças coronarianas na tomadas de decisão relacionadas ao tratamento mais adequado e individualizado para pacientes baseado em seus dados específicos de estilo de vida.


## Questão de pesquisa

Com qual confiabilidade é possível, a partir de um conjunto de atributos e fatores relacionados a doenças cardíacas construir um modelo que recomende o tratamento mais adequado?"

## Objetivos preliminares

Desenvolver um modelo preditivo utilizando técnicas de aprendizado de máquina para estimar com alguma precisão o risco de ocorrência de um doenças coronárias em indivíduos com alguma enfermidade prévia.
Haverá análise dos fatores de risco presentes em um dataset público, identificando quais variáveis influenciam mais o desenvolvimento de doenças coronarianas. 

Definido os fatores e suas respectivas grandezas, será elaborado um modelo preditivo, de modo a contribuir para a tomada de decisões em relação ao tratamento mais adequado por parte dos profissionais de saúde e promoção da saúde cardiovascular dos pacientes.
 

## Justificativa 

As doenças cardíacas continuam sendo uma das principais causas de morte no Brasil e no mundo. De acordo com dados globais recentes, as doenças cardiovasculares (DCV) representaram aproximadamente 19,8 milhões de mortes em 2022, um aumento significativo em relação aos 12,4 milhões registrados em 1990. Esse aumento é atribuído ao crescimento populacional, ao envelhecimento da população e à prevalência de fatores de risco como hipertensão, colesterol elevado, dieta inadequada e poluição do ar ​(Our World in Data, IHME).

No Brasil, as doenças cardiovasculares são a principal causa de mortes, responsáveis por aproximadamente 380 mil óbitos por ano, o que equivale a cerca de 30% de todas as mortes no país. O Infarto Agudo do Miocárdio (IAM), popularmente conhecido como ataque cardíaco, e o acidente vascular cerebral (AVC) são as manifestações mais comuns e letais dessas doenças​ (Our World in Data).

Globalmente, a doença arterial coronariana (DAC) é a forma mais comum de DCV, e em 2022, causou 371.506 mortes nos Estados Unidos, refletindo a gravidade da condição. Estima-se que uma em cada cinco mortes por DCV ocorra em pessoas com menos de 65 anos, destacando a importância de intervenções preventivas em populações mais jovens ​(CDC).

Os dados também revelam disparidades regionais significativas na mortalidade por DCV. Regiões como a Europa Oriental e a Ásia Central apresentam as taxas mais altas de mortalidade ajustadas por idade, enquanto a Australásia apresenta as taxas mais baixas. Essas variações refletem diferenças nos sistemas de saúde, nos estilos de vida e na exposição a fatores de risco em diferentes partes do mundo (IHME).

Em termos de fatores de risco, a hipertensão sistólica foi o maior contribuinte para a carga de doenças cardiovasculares, seguido por riscos dietéticos e poluição por partículas ambientais. Esses fatores destacam a necessidade de políticas de saúde pública focadas na redução desses riscos através de campanhas de conscientização e intervenções de saúde preventiva​ (IHME).

A utilização de técnicas de machine learning para avaliar e melhorar o tratamento de pacientes com doenças cardíacas é de importância crucial devido ao impacto massivo dessas doenças na saúde global. Dado que as doenças cardiovasculares são a principal causa de mortes tanto no Brasil quanto no mundo, qualquer avanço que permita personalizar e otimizar o tratamento para diferentes perfis de pacientes pode ter efeitos profundamente positivos na saúde pública, servindo como ferramenta de apoio para médicos em suas tomadas de decisão.

Machine learning oferece uma abordagem poderosa para a análise de grandes volumes de dados médicos, permitindo a identificação de padrões complexos que podem não ser evidentes para os métodos tradicionais. Por exemplo, ao considerar o estilo de vida, histórico médico e condições biológicas individuais, algoritmos de machine learning podem prever quais tratamentos são mais eficazes para diferentes subgrupos de pacientes. Isso é especialmente relevante para doenças como as cardiovasculares, onde os fatores de risco são múltiplos e interdependentes, variando de fatores genéticos a hábitos de vida como dieta e atividade física​ (CDC, IHME).

Além disso, modelos de machine learning podem ser usados para identificar pacientes com maior risco de complicações, ajudando os médicos a tomar decisões informadas sobre intervenções precoces. Por exemplo, um paciente com hipertensão e histórico familiar de doenças cardíacas pode ser monitorado mais de perto ou tratado com regimes terapêuticos mais agressivos, baseados em predições específicas ao seu perfil​ (Our World in Data).

Em resumo, o uso de machine learning em estudos acadêmicos focados na prevenção e tratamento de doenças cardíacas não só permite uma abordagem mais personalizada, mas também pode ajudar a reduzir a mortalidade e melhorar a qualidade de vida dos pacientes, ao garantir que cada indivíduo receba o tratamento mais adequado às suas necessidades específicas. Isso é particularmente vital em um cenário global onde a carga de doenças cardiovasculares continua a crescer, exigindo estratégias inovadoras e baseadas em dados para enfrentar esse desafio​ (IHME).

## Público-Alvo 

O público-alvo são médicos que desejam possuir uma ferramenta de apoio para tomadas de decisões relacionadas à melhor forma de tratamento para pacientes que possuem algum problema cardíaco já diagnosticado.

Mapa de stakeholders:

![image](https://github.com/user-attachments/assets/78f57562-acf9-44ef-a38f-6aac275c6c1e)

Descrição:

1° Os Beneficiários Diretos (Primários) da ferramenta de apoio, serão os médicos.

2° Os Beneficiários Indiretos (Secundários) da ferramenta de apoio, serão os pacientes.

3° Os Beneficiários Indiretos (Terciários) da ferramentade apoio, serão o Governo, o Sistema Único de Saúde, e os demais orgãos públicos e privados ligados a saúde.

## Estado da arte

### Heart Disease Prediction Using Machine Learning (International Journal of Innovative Research in Computer and Communication Engineering, 2024)

#### 1. Introdução ao Problema
As doenças cardíacas são uma das principais causas de mortalidade no mundo, e a previsão precisa dessas condições pode salvar inúmeras vidas. O artigo "Heart Disease Prediction Using Machine Learning" de Lowlesh Yadav investiga a aplicação de técnicas de machine learning para prever a probabilidade de um indivíduo desenvolver doenças cardíacas. O objetivo principal do estudo é aprimorar a precisão das previsões ao empregar diferentes algoritmos de machine learning, comparando seu desempenho e selecionando a melhor abordagem para este problema crítico de saúde pública.

#### 2. Características do Dataset Utilizado
O estudo utiliza um dataset bem conhecido no domínio da previsão de doenças cardíacas, geralmente derivado do Cleveland Heart Disease dataset. Este conjunto de dados inclui diversas variáveis relevantes para o diagnóstico de doenças cardíacas, tais como:

- Idade: Idade do paciente.
- Sexo: Gênero do paciente.
- Pressão arterial em repouso: Medida da pressão arterial em repouso (mm Hg).
- Colesterol sérico: Níveis de colesterol sérico (mg/dL).
- Glicemia em jejum: Indicador se a glicemia em jejum é maior que 120 mg/dL.
- Resultados do eletrocardiograma (ECG): Inclui anormalidades específicas.
- Frequência cardíaca máxima atingida: Durante exercício físico.
- Angina induzida por exercício: Se o paciente apresenta dor no peito induzida por esforço.
- Depressão do segmento ST: Comparação entre o repouso e o exercício.
- Pico do segmento ST: Tipo de anomalia no ECG.
- Número de vasos principais coloridos por fluoroscopia: Indicador de bloqueios nas artérias.
- Talassemia: Presença ou ausência de um distúrbio genético.

Este conjunto de dados foi escolhido devido à sua relevância clínica e sua ampla utilização em pesquisas anteriores, o que permite comparações diretas com outros estudos na área.

#### 3. Abordagens e Algoritmos Utilizados
O autor implementa vários algoritmos de machine learning para prever doenças cardíacas. As principais abordagens exploradas no artigo incluem:

- Regressão Logística: Um modelo de regressão linear aplicado a problemas de classificação binária.
  - Parâmetros utilizados: Regularização L2, taxa de aprendizado ajustada.

- K-Nearest Neighbors (KNN): Algoritmo baseado em instâncias que classifica os dados com base na proximidade dos pontos de dados.
  - Parâmetros utilizados: Número de vizinhos (k=5), distância euclidiana.

- Support Vector Machine (SVM): Algoritmo que encontra o hiperplano que melhor separa as classes.
  - Parâmetros utilizados: Kernel radial basis function (RBF), parâmetro de regularização (C=1), parâmetro do kernel (gamma).

- Árvore de Decisão: Algoritmo de aprendizado supervisionado que particiona o espaço de dados em sub-regiões baseadas em features.
  - Parâmetros utilizados: Critério de entropia, profundidade máxima da árvore.

- Random Forest: Conjunto de árvores de decisão para melhorar a robustez e reduzir o overfitting.
  - Parâmetros utilizados: Número de árvores (100), profundidade máxima, critério de entropia.

#### 4. Métricas de Avaliação
Para avaliar o desempenho dos modelos, o autor utiliza diversas métricas de avaliação, destacando:

- Acurácia: A porcentagem de previsões corretas sobre o total de previsões feitas.
- Precisão (Precision): A proporção de verdadeiros positivos sobre o total de positivos preditos.
- Recall: A proporção de verdadeiros positivos sobre o total de reais positivos.
- F1-Score: A média harmônica entre precisão e recall, oferecendo um equilíbrio entre ambas.
- Matriz de Confusão: Representação tabular que permite a visualização de verdadeiros e falsos positivos/negativos.

Essas métricas foram escolhidas para oferecer uma visão abrangente do desempenho de cada modelo, especialmente considerando a natureza balanceada ou desbalanceada dos dados.

#### 5. Resultados Obtidos
Os resultados mostraram que o Random Forest apresentou o melhor desempenho geral, com acurácia superior a 85%, seguido pela Support Vector Machine (SVM). A regressão logística e o KNN também tiveram desempenhos aceitáveis, mas com menor acurácia e precisão. O autor conclui que a escolha do algoritmo deve ser baseada tanto na performance quanto na interpretabilidade do modelo, com o Random Forest sendo uma excelente escolha para a tarefa de previsão de doenças cardíacas devido à sua alta precisão e robustez contra overfitting.

### Prevendo Doenças Cardiovasculares com Machine Learning (Medium, 2024)
Um estudo sobre predição de doenças cardíacas.

#### 1. Introdução ao Problema 
As doenças cardíacas, uma das principais preocupações globais de saúde, têm sido alvo de estudos aprofundados, e o campo do aprendizado de máquina tem se destacado como uma ferramenta promissora para lidar com essa questão crucial.

#### 2. Características do Dataset Utilizado 
Nomes | traduções:

- age | idade em anos
- sex | sexo (1 = masc 0 = fem)
- cp (chest pain type) | tipo de dor no peito
  - Valor 0: asymptomatic | assintomática
  - Valor 1: typical angina | típica
  - Valor 2: atypical angina | atípica
  - Valor 3: non-anginal pain | não anginosa
- trestbps: resting blood pressure (in mm Hg on admission to the hospital) | pressão arterial de repouso (em mm Hg na admissão ao hospital)
- chol: serum cholestoral in mg/dl | colesterol sérico em mg/dl
- restecg: resting electrocardiographic results (values 0, 1, 2) | Resultados eletrocardiográficos em repouso: Valores 0, 1, 2.
- fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) | glicemia em jejum
- thalach: maximum heart rate achieved | Frequência cardíaca máxima atingida
- exang: exercise induced angina (1 = yes; 0 = no) | angina induzida por exercício
- oldpeak: ST depression induced by exercise relative to rest | Depressão de ST induzida por exercício em relação ao
- slope | inclinação: a inclinação do pico do segmento ST do exercício
  - Valor 0: inclinação ascendente
  - Valor 1: plano
  - Valor 2: descida<br>
- ca: number of major vessels (0–3) colored by flouroscopy | Número de vasos principais coloridos por fluoroscopia:
  - Valor 0: Normal
  - Valor 1: Com onda ST-T anormal
  - Valor 2: Mostrando provável (ou definida) hipertrofia do ventrículo esquerdo
- thal: 1 = normal; 2 = fixed defect; 3 = reversable defect) | Talassemia: 1 = normal; 2 = problema corrigido; 3 = problema reversível
- target: 0 = não possui doença cardíaca, 1 = possui doença cardíaca

#### 3. Abordagens e Algoritmos Utilizados

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

#### 4. Métricas de Avaliação
As métricas usadas para avaliar o desempenho dos modelos incluem:

- Acurácia - para medir a proporção de previsões corretas.
- Precisão - para avaliar a exatidão das previsões positivas.
- Recall (Sensibilidade) - para medir a taxa de detecção de casos positivos.
- F1-score - combinação de precisão e recall para um resultado balanceado.

#### 5. Resultados Obtidos 
Os resultados mostraram que o algoritmo Random Forest apresentou o melhor desempenho em comparação com os outros modelos testados, como KNN e SVM. O modelo de Random Forest obteve a maior acurácia na previsão de doenças cardíacas, destacando-se por sua capacidade de generalização. Além disso, as métricas de precisão, recall e F1-score também indicaram um bom equilíbrio entre as previsões positivas e negativas, reforçando a eficiência do modelo.

### Detecção de doenças cardíacas usando algoritmos de aprendizado de máquina e um sistema de monitoramento de saúde cardiovascular em tempo real (World Journal of Engineering and Technology, 2018) ###


#### 1. Introdução ao Problema ##

Donças cardíacas são as principais causas de morte em todo o mundo. De acordo com a pesquisa da Organização Mundial da Saúde (OMS), 17,5 milhões de mortes globais ocorrem devido a ataques cardíacos e derrames. Mais de 75% das mortes por doenças cardiovasculares ocorrem principalmente em países de renda média e baixa. Além disso, 80% das mortes que ocorrem por doenças cardiovasculares (DCVs) são devido a derrame e ataque cardíaco. Portanto, a detecção de anormalidades cardíacas no estágio inicial e ferramentas para a previsão de doenças cardíacas podem salvar muitas vidas e ajudar os médicos a elaborar um plano de tratamento eficaz que, em última análise, reduza a taxa de mortalidade devido a doenças cardiovasculares. 

#### 2. Características do Dataset Utilizado
O dataset contém um total de 303 registros de pacientes. Estes dados foram originalmente coletados de quatro diferentes fontes médicas, mas o conjunto de dados completo disponível na UCI é composto principalmente pelos dados do Cleveland Clinic Foundation.

O Dataset obtido para a aplicação de mineração de dados inclui 13 tipos de entrada. Dois bancos de dados de doenças cardíacas disponíveis publicamente com o mesmo tipo e número de atributos são mesclados para formar um conjunto de dados maior e obter maior precisão.
  
- ![image](https://github.com/user-attachments/assets/7a777302-56c1-4d8e-98e3-6c5320e10b20)


Atributos que estão relacionados a diferentes parâmetros do sistema cardiovascular. Para o design do sistema de classificação e predição baseado em nuvem, um atributo-chave foi projetado (ou seja, o número de celular do paciente que atua como o identificador exclusivo do paciente). Atributos previsíveis recebem uma escala numérica para determinar os dois grupos classificados (ou seja, Saudável e Com doença cardíaca). A Tabela 1 mostra os atributos de entrada contendo um total de 13 características fisiológicas retiradas dos dois conjuntos de dados mencionados anteriormente usados ​​para qualquer tipo de predição de anormalidade cardíaca.

 ####  3. Abordagens e Algoritmos Utilizados
As tecnologias utilizadas no estudo foram:

- Sistema de monitoramento contínuo baseado em um microcontrolador Arduino com sensores de temperatura, umidade e batimentos cardíacos para coleta de dados fisiológicos em tempo real.
- Aplicativo inteligente para predição de doenças cardíacas baseado em nuvem, onde os dados são analisados e enviados para médicos através de uma interface móvel.

Já para os algorítmos, temos os seguintes:

- Naive Bayes: Classificador probabilístico que assume independência entre atributos.
- Redes Neurais Artificiais (ANN): Capazes de modelar funções não lineares complexas.
- Support Vector Machine (SVM): Usado para dados lineares e não lineares, com um hiperplano para separar classes.
- Random Forest: Algoritmo de bagging que cria várias árvores de decisão e as combina para previsão.
- Regressão Logística Simples: Utilizada para classificação binária, com uma função logística que retorna valores entre 0 e 1.

 #### 4. Métricas de Avaliação
As métricas de avaliação utilizadas no artigo incluem:

- Taxa de Acerto (Accuracy): Porcentagem de previsões corretas.
- Precisão (Precision): Proporção de verdadeiros positivos em relação ao total de positivos previstos.
- Revocação (Recall): Proporção de verdadeiros positivos em relação ao total de positivos reais.
- F1 Score: Média harmônica entre precisão e revocação, equilibrando ambos.
- AUC-ROC: Área sob a curva da característica de operação do receptor, medindo a capacidade de discriminação do modelo.

Essas métricas foram empregadas para comparar o desempenho dos algoritmos de aprendizado de máquina e selecionar o modelo mais eficiente na predição de doenças cardíacas.

#### 5. Resultados Obtidos

Os resultados obtidos mostraram que cinco algoritmos de aprendizado de máquina apresentaram precisão superior a 80% na previsão de doenças cardíacas. As métricas de desempenho foram as seguintes:

- Naive Bayes: Acurácia de 86,40%
- SVM: Acurácia de 97,53%
- Floresta Aleatória: Acurácia de 95,76%
- Regressão Logística Simples: Acurácia de 95,05%
- Redes Neurais Artificiais: Acurácia de 77,39%

Esses resultados indicam a eficácia dos algoritmos selecionados na detecção de doenças cardíacas.


### Melhorando a previsão de ataques cardíacos com aprendizado de máquina: um estudo no Jordan University Hospital(ALSHRAIDEH, Mohammad, 2024) ###


#### 1. Introdução ao Problema 

Doença cardíaca é uma das principais causas globais de mortalidade, e seu diagnóstico oportuno apresenta desafios significativos devido à sobreposição de sintomas com outras condições de saúde. Essa complexidade pode complicar o tratamento significativamente quando a detecção é tardia.

#### 2. Características do Dataset Utilizado

Esta pesquisa usou um conjunto de dados de doenças cardíacas do JUH em Amã, Jordânia, para testes e treinamento do sistema. O conjunto de dados compreendeu um total de 486 casos. Destes, 324 instâncias estão associadas a pacientes diagnosticados com doenças cardíacas. Os 162 casos restantes pertencem a pacientes sem doenças cardíacas que já visitaram clínicas de cardiologia. Incluiu 58 variáveis ​​essenciais para o diagnóstico de doenças cardíacas, categorizadas da seguinte forma:


* (1) Informações básicas do paciente (por exemplo, idade e sexo);

* (2) Histórico médico do paciente (10 fatores);

* (3) Sintomas relatados (16 fatores);

* (4) Resultados do exame físico (10 fatores);

* (5) Resultados do laboratório de sangue (7 fatores);

* (6) Resultados do ECG (eletrocardiograma) (12 fatores);

O conjunto de dados compreendia predominantemente atributos binários, denotando a presença ou ausência de características. No entanto, características específicas, como tabagismo e ritmo cardíaco, tinham vários valores para indicar a gravidade. Gênero e valores numéricos, como idade, pressão arterial e pulso, também faziam parte do conjunto de dados. A Figura 8 ilustra a planilha de dados usada para reunir variáveis ​​do paciente, destacando a prevalência de valores binários para muitos atributos e, nos casos em que um atributo estava presente, discernindo a gravidade categorizando-a como grave ou leve.


- ![image](/src/images/Patient%20Datasheet.PNG)

#### 3. Abordagens e Algorítmos Utilizados

O artigo discute diversas abordagens de aprendizado supervisionado e não supervisionado, com foco em algoritmos de aprendizado de máquina, como:

- Máquinas de Vetores de Suporte (SVM): Utilizadas para classificação, maximizando a margem entre classes.
- Florestas Aleatórias (RF): Combinam múltiplas árvores de decisão, eficazes em dados de alta dimensionalidade.
- Árvores de Decisão (DTs): Estrutura de árvore para decisões, fácil de interpretar.
- Naive Bayes: Classificação baseada na independência condicional entre características.
- K-Vizinhos Mais Próximos (KNN): Classificação por maioria entre os vizinhos mais próximos.

Esses algoritmos são aplicados na detecção de doenças cardíacas.

#### 4. Métricas de Avaliação

As métricas de avaliação calculadas a partir da matriz de confusão incluem:

- Acurácia: Proporção de previsões corretas (TP + TN) em relação ao total de casos (TP + TN + FP + FN).
- Precisão: Porcentagem de previsões positivas corretas, refletindo a confiabilidade do modelo.
- ROC (Receiver Operating Characteristic): Gráfico que relaciona a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR).
- F1-score: Média harmônica entre precisão e recall, útil para equilibrar ambos os aspectos.

Essas métricas fornecem uma visão abrangente do desempenho do modelo de classificação.

#### 5. Resultados Obtidos

O estudo alcançou uma precisão notável de 94,3% na previsão de doenças cardíacas, utilizando o classificador SVM em conjunto com a otimização por enxame de partículas para a seleção de características. Esses resultados destacam a eficácia dos algoritmos de aprendizado de máquina na categorização de pacientes de acordo com o risco de doenças cardíacas, além de ressaltar a necessidade de mais pesquisas para abordar limitações relacionadas à representatividade dos dados e à interpretabilidade dos modelos.

Fonte:https://onlinelibrary.wiley.com/doi/full/10.1155/2024/5080332

# Descrição do _dataset_ selecionado 

O dataset escolhido para o tema contém atributos pertinentes para criação do modelo listando - dentro de um cenário em que todos os avaliados possuem problemas cardiovasculares - informações gerais sobre cada paciente analisado. Com isso, a seção do projeto foi elaborada com o fundamento de discorrer e explicar cada um dos atributos do conjunto de dados.


* Gênero: variável qualitativa do tipo textual que define se o paciente é um homem ou uma mulher. 1 = masculino, 2 = feminino ;
* Idade: variável quantitativa do tipo numérico que determina a idade de cada um dos pacientes apresentados na base de dados. Range de 30 a 89;
* Pressão Arterial:  variável quantitativa do tipo numérico. De acordo com a UFMG, pressão arterial é a força que o sangue executa contra as paredes da artéria. A partir disso, será possível verificar se os pacientes possuem ou não hipertensão, um fator importante para ataques cardíacos. Range de 90 a 199;
* Colesterol: variável quantitativa do tipo numérico. Segundo o Hospital Israelita Robert Einstein, colesterol é uma gordura que está no organismo do ser humano, o qual produz hormônios masculinos, femininos e também a vitamina D. Este também é um dos principais fatores para ocorrência de um infarto. Range de 150 a 299; 
* Se possui ou não diabete: dado booleano que é considerado qualitativo. Ele indica basicamente se a pessoa possui ou não diabete. Sim = possui diabetes, não = não possui diabetes.
* Situação sobre fumar: dado qualitativo que indica se a pessoa fuma por três categorias. Nunca fumou, fuma atualmente, outros;
* Tipo de dor no peito: dado qualitativo que indica alguns tipos de dor no peito que os pacientes sentiram. Dor anginosa típica, dor anginosa atípica, do não anginosa e assintomático.
* Tratamento utilizado: dado qualitativo o qual mostra os tratamentos recebidos pelos pacientes do dataset. Mudanças no estilo de vida, Angioplastia, Bypass ou uso de Medicação.


Ao analisar cada um dos atributos desta base de dados, foi possível concluir que não será necessário realizar uma limpeza nos dados, pois não foi encontrado valores inesperados dentro do dataset.

Link para o dataset: https://www.kaggle.com/datasets/waqi786/heart-attack-dataset 


# Canvas analítico

![image](/src/images/Canvas%20Revisado.png)

# Referências

**Institute for Health Metrics and Evaluation.** New study reveals latest data on global burden of cardiovascular disease. Institute for Health Metrics and Evaluation. Disponível em: https://www.healthdata.org/news-release/new-study-reveals-latest-data-global-burden-cardiovascular-disease. Acesso em: 01/09/2024.

**Centers for Disease Control and Prevention (CDC).** Heart Disease Facts. Centers for Disease Control and Prevention. Disponível em: https://www.cdc.gov/heartdisease/facts.htm. Acesso em: 01/09/2024.

**Our World in Data.** Death rate from cardiovascular diseases. Our World in Data. Disponível em: https://ourworldindata.org/grapher/cardiovascular-disease-death-rates. Acesso em: 01/09/2024.
UCI Machine Learning Repository. (n.d.). Heart Disease Data Set. Disponível em: https://archive.ics.uci.edu/ml/datasets/heart+disease.

YADAV, L. (2024). Heart Disease Prediction Using Machine Learning. Disponível em: https://www.researchgate.net/profile/Lowlesh-Yadav-2/publication/382530262_Heart_Disease_Prediction_Using_Machine_Learning/links/66a21be4c6e41359a83c0853/Heart-Disease-Prediction-Using-Machine-Learning.pdf. Acesso em 01/09/2024.

LIBERATO, J. (2024). Prevendo doenças cardiovasculares. Medium. Disponível em: https://medium.com/@jonatasliberato/prevendo-doen%C3%A7as-cardiovasculares-2a2c3dd68789. Acesso em: 27/08/20244.

SCIRP. (2018). A Review of Machine Learning Techniques for Heart Disease Prediction. Scientific Research Publishing. Disponível em: https://www.scirp.org/journal/paperinformation?paperid=88650#ref15. Acesso em: 01/09/2024.

FREESZ, Larissa e PINHEIRO, Paula. ENTENDO A PRESSÃO ARTERIAL. **UFMG**, Minas Gerais, 2011. Disponível em <https://www.ufmg.br/cienciaparatodos/wp-content/uploads/2013/12/pag14-Ciencia.pdf>. Acesso em: 28 de ago. de 2024.

OYAMA, Patrícia. Pressão alta: o maior perigo é a falta de informação. **Abbott**, 2018. Disponível em: <https://www.abbottbrasil.com.br/corpnewsroom/nutrition-health-and-wellness/pressao-alta--o-maior-inimigo-e-a-falta-de-informacao.html#:~:text=Quando%20o%20sangue%20circula%20com,angina%20(dores%20no%20peito)>. Acesso em: 28 de ago. de 2024.

ALSHRAIDEH, Mohammad. Enhancing Heart Attack Prediction with Machine Learning: A Study at Jordan University Hospital, **Wiley Online Library**, 2024. Disponível em:<https://onlinelibrary.wiley.com/doi/full/10.1155/2024/5080332>. Acesso em: 1  de set. de 2024.


Colesterol alto: perigo silencioso que ataca o coração. **Blog - Evangelico**, 2023. Disponível em: <https://blog.evangelicohospital.com.br/todos/colesterol-alto-perigo-silencioso-que-ataca-o-coracao/#:~:text=Fernando%20Oliva%2C%20esclarece%20que%20o,e%20s%C3%ADntese%20da%20vitamina%20D.>. Acesso em: 19 de set. de 2024.

Colesterol alto: entenda como tratar. **Dr. Eduardo Enrique**, Ano não informado. Disponível em: <https://www.dreduardoendocrino.com.br/colesterol>. Acesso em: 23 de set. de 2024.


Tratamento para hipertensão: por que consultar um cardiologista é importante? **WeCor**, 2020. Disponível em: <https://www.wecor.com.br/tratamento-para-hipertensao#:~:text=A%20losartana%20%C3%A9%20um%20medicamento,com%20hipertens%C3%A3o%20ou%20insufici%C3%AAncia%20card%C3%ADaca.>. Acesso em: 23 de set. de 2024.


MARIANO, Diego. Métricas de avaliação em machine learning. **Bioinfo**, 2021. Disponível em: <https://bioinfo.com.br/metricas-de-avaliacao-em-machine-learning-acuracia-sensibilidade-precisao-especificidade-e-f-score/>. Acesso em 24 de nov. de 2024.