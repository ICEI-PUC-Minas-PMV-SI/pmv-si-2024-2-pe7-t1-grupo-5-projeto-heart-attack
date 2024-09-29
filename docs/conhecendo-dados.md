# Conhecendo os dados



### Utilizando o Google Colab, conseguimos obter as informações abaixo:


![image](https://github.com/user-attachments/assets/057f62bb-73ee-45ba-a983-3000ab840a01)

O Dataset é composto por 8 colunas: 
- Gender (Gênero Masculino e Feminino);
- Age (Idade);
- Blood Pressure (Pressão Sanguínea);
- Cholesterol (Colesterol);
- Has Diabetes (se paciente tem ou não Diabetes);
- Smoking Status (se paciente é, já foi ou não é fumante);
- Chest Pain Type (tipo de dor no peito que o paciente sente);
- Treatment (tipo de tratamento indicado).
Temos também 1.000 rows (linhas) referente à dados de 1.000 pessoas participantes do estudo/pesquisa.


![image](https://github.com/user-attachments/assets/9e21d732-08c7-4cca-8eb7-d6f990d45edf)

Das 8 colunas presentes no Dataset, 5 colunas apresentam dados do tipo objeto (Gender, Has Diabetes, Smoking Status Chest Pain Type, Treatment) sendo variáveis qualitativas e 3 colunas apresentam dados do tipo inteiro/int64 que são quantitativas (Age, Blood Pressure, Cholesterol). 


![image](https://github.com/user-attachments/assets/21f7d0b3-aed6-4dcb-9d47-2205d19f5156)

Importante destacar que não há dados nulos (linhas e colunas sem conteúdo) nesse Dataset. 


![image](https://github.com/user-attachments/assets/1dbfb213-7420-4129-8a40-f0f6f53371f5)

Realizando o describe do Dataset, temos para as variáveis quantitativas com dados do tipo inteiro/int64 que são numéricas das colunas Age, Blood Pressure e Choleterol, e obtemos os parâmetros abaixo para cada uma das colunas:
- Count (contagem);
- Mean (média);
- Std (desvio padrão);
- Min (mínimo);
- Quartis (25%. 50%, 75%);
- Max (máximo).
  
Para as colunas que possuem variáveis qualitativas faremos uma análise e definiremos a melhor transformação dos dados para se tornarem booleanas ou numéricas.

Importante: aqui começamos a ver algumas características, median 50% e mean média estão próximas, indicativo de formato parecido com distribuição normal, está bem distribuída, com comportamento regular.

![image](https://github.com/user-attachments/assets/fdf340c6-dad0-4156-a379-2d32c551ef40)

Verificamos que o estudo foi realizado com 1.000 pessoas, sendo: 510 Mulheres e 490 Homens, e que a média de idade para ambos os gêneros é de 60 anos.

## Análises Realizadas


O grupo utilizou a ferramenta software Power BI para coletar métricas que auxiliarão nas análises dos pacientes do dataset selecionado. Para elaborar as métricas, foi necessário a utilização da linguagem DAX(Data Analysis Expressions) para criar algumas funções como a média de colesterol dos pacientes envolvidos.

Nesta tabela, os componentes trouxeram um breve detalhamento da média de colesterol dos pacientes. Para realizar o agrupamento dos participantes, foi necessário criar uma coluna condicional através do Power Query, ferramenta ETL para limpeza e preparação de dados do Power BI. Com isso, os pacientes foram separados por sua faixa etária.


![image](/src/images/Colesterol.png)

Para mais detalhamentos, os códigos DAX utilizados se encontram na pasta src.
## Hipótese 1

**Pessoas com colesterol alto possuem mais chance de sofrerem de ataque cardíaco?**

Segundo o Dr. Fernando Oliva afirmou que: “O excesso de colesterol é diretamente responsável por formar depósitos de gordura nas artérias, também conhecidos como placa de ateroma, podendo provocar com o tempo obstruções e até mesmo oclusões destes vasos. Consequentemente, os órgãos deixam de receber aporte sanguíneo, deixando de funcionar. No coração, isso é conhecido como infarto”. 

Isso também pode ser mostrado no dataset escolhido pelos componentes, pois 92,51% dos pacientes possuem colesterol alto. Segue um gráfico que mostra o agrupamento desta informação. Basicamente foi adicionado uma coluna adicional no dataset para agrupar pacientes por colesterol alto e baixo(Acima de 160 mg/dL, de acordo com a Unimed) e a contagem dos pacientes do dataset.

![image](/src/images/Pacientes%20por%20%20Nível%20de%20Colesterol.png)

Com isso, conclui-se que o nível de colesterol é um fator importante para causar ataques cardíacos. 

## Hipótese 2


A tabela de pressão arterial é útil para controlar doenças crônicas, como a hipertensão, entre outras, ou para verificar anormalidades que contribuem para o diagnóstico e sinalizam que é hora de buscar ajuda médica. Segundo o Departamento de Hipertensão Arterial da Sociedade Brasileira de Cardiologia, a pressão arterial normal limítrofe refere-se a valores que estão próximos ao limite superior do que é considerado normal. Isso geralmente indica que a pressão arterial está elevada, mas ainda não é alta o suficiente para ser classificada como hipertensão. No entanto, o risco de avançar para o grau 1 de hipertensão leve é muito grande. 

![image](https://github.com/user-attachments/assets/63d4b971-fb74-495a-837b-92724a40b8a7)

Tabela II. Classificação diagnóstica da hipertensão arterial (adultos com mais de 18 anos de idade).

Quando o sangue circula com a pressão elevada, ele vai machucando as paredes dos vasos sanguíneos, que se tornam endurecidos e mais estreitos. Com o passar do tempo, se o problema não for controlado, os vasos podem entupir e até se romper, o que pode causar infarto, insuficiência cardíaca e angina (dores no peito). 

Teste de Hipóteses

Passo 1 : Formulando a Hipótese. 

Teste de hipótese estatístico: **A média da pressão arterial é superior a 130?**  

Hipótese Nula (H₀): A média da pressão arterial é menor ou igual a 130. 
𝐻0: 𝜇 ≤ 130

Hipótese Alternativa (H₁): A média da pressão arterial é superior a 130. 
𝐻1: 𝜇 > 130

Passo 2 : Estabelecendo as Regiões de Aceitação e Rejeição (RA ; RR);

Para realizar o teste, é necessário calcular a média e o desvio padrão da amostra.

Número de dados selecionados (n): 600 

Média da amostra (xˉ) = 147.5 

Desvio padrão da amostra (s) = 20.7 

Utilizando a fórmula: ![image](https://github.com/user-attachments/assets/9121daea-4abd-4f6b-814d-67c55ad6a60d)

t é aproximadamente = 20.7

Passo 3 : Calculando a estatística do teste;

Para um nível de significância  𝛼 de 0.05 , o valor crítico t para uma distribuição t com 599  para uma cauda (direita) é aproximadamente 1.645.

Passo 4 : Concluindo o teste:

Como o valor calculado de t (20.7) é muito maior do que o valor crítico (1.645), é rejeitado a hipótese nula, pois há evidências suficientes para concluir-se que a média da pressão arterial é significativamente superior a 130. Portanto a média da pressão arterial do grupo analisado, no mínimo, encontra-se em um grau de aproximação ao primeiro estágio de impertensão leve.

Fonte: http://departamentos.cardiol.br/dha/consenso3/capitulo1.asp

## Hipótese 3

**Somente um atributo pode influenciar completamente para definir o tratamento de um paciente?**


Para se explicar o assunto, serão utilizados 2 situações.

* Tratamento Para Hipertensão - WeCor(2020): 

Conforme citado no site, há várias formas de se tratar a pressão arterial, como realizar exercícios físicos, parar de fumar, mudança de vida e uso de medicamentos junto do acompanhamento médico.


* Tratamento Para Colesterol - Dr. Eduardo Enrique(Ano não informado)

Segundo o especialista, para se escolher o melhor tratamento, ão levados em consideração os hábitos de vida, histórico pessoal e familiar de infarto, AVC e morte por doenças cardiovasculares, além do histórico de alterações metabólicas e cardiovasculares, como diabetes e hipertensão.

Desta forma conclui-se que vários atributos estão atrelados para a melhor escolha do tratamento.

Além disso, esses conhecimentos são refletidos na base de dados, é possível constatar essa informação através da matriz de confusão posicionada na seção “Descrição dos achados”.




## Descrição dos achados

A partir da análise descrita e exploratória realizada, descreva todos os achados considerados relevantes para o contexto em que o trabalho se insere. Por exemplo: com relação à centralidade dos dados algo chamou a atenção? Foi possível identificar correlação entre os atributos? Que tipo de correlação (forte, fraca, moderada)? 

![image](https://github.com/user-attachments/assets/96836fd3-a8a2-4e6d-8efe-e7e3ed21e12d)

Elaboramos uma matriz de confusão para compreender melhor a correlação entre as variáveis do Dataset, podemos pegar uma variável x e comparar com outra variável, usando heat map mapa de calor, sendo vermelho para correlações de Pearson fortes e positivas, vermelho menos intenso para moderadas e positivas e vermelho claro para fraca e positiva. Azul para correlações negativa, azul forte forte e negativa, azul menos intenso para moderada e negativa e azul claro para fraca e negativa ou mais próximo do branco sem correlação.

![image](https://github.com/user-attachments/assets/9001447d-acb5-4ba3-ad46-e249157d860d)

Utilizando a IQR (amplitude inter quartil) que é a diferença entre o Q3 (terceiro quartil) e Q1 (primeiro quartil) vemos que que não foram encontrados nenhum outlier no dataset que está sendo trabalhado.

![image](https://github.com/user-attachments/assets/0c2f8274-0aa7-4ba4-9499-d344fea6bdfb)
![image](https://github.com/user-attachments/assets/805afdad-1b89-412e-bde0-30b12443a318)

Usando o pairplot da biblioteca seaborn do python conseguimos identificar algumas características interessantes do dataset:
1) Age (idade): notamos que as idades com mais incidência de problemas caridovasculares possui maior ocorrências para as faixas etárias: 40 anos (possivelmente ocorre por conta de hábitos pouco saudáveis como sedentarismo e má alimentação associados à alta carga de estresse já que se encontram possivelmente profissionamente ativos) e entre 80 e 90 anos (possivelmente em decorrência da idade e consequência de alguns hábitos não saudáveis ao longo da vida);
2) Chest pain type (tipo de dor no peito): notamos que Non-anginal Pain (Dor não anginosa) e Asymptomatic (Assintomático) são as mais frequentes, o que indica que uma dor no peito demanda um exame mais amplo além de causas cardiovasculares (como: pulmões, músculos, esôfago entre outros) e a importância de um acompanhamento mais constante da condição cardíaca, pois há casos que não há sintomas evidentes de possíveis probelmas.
3) Sobre Treatment (tratamento): os tratamentos mais indicados são Lifestyle Changes (Mudanças no estilo de vida) e Coronary Artery Bypass Graft (Enxerto de Revascularização da Artéria Coronária) que consiste em construir um novo caminho para o fluxo de sangue. Vendo isso temos que a adequação da dieta para que seja mais balanceada e adoção dos exercícios físicos na rotina podem ser suficiente para correção ou prevenção de problemas cardiovasculares, e nos caso do Bypass, mostra que muitos casos são extremos e indicam entupimento de veias, tendo como possíveis causas tabagismo, dieta desbalanceada e sedentarismo.
4) Diabetes: notamos que pessoas com essa condição são mais propensas à desenvolverem problemas cardiovasculares;
5) Smoking Status (Status de fumante): olhando o dataset, vimos que fumantes da categoria former (ex-fumante) e current (atualmente fumante) tem mais chances de terem problemas cardíacos, no caso de pessoas que pararam de fumar estão tendo a consequência de ter realizado esse hábito por alguns anos e no caso de pessoas que fumam atualmente podem ser pessoas que fumam há algum tempo ou mesmo pessoas que fumam há pouco tempo e que possuem hábitos pouco saudáveis como sedenterismo e dieta desbalanceada. Importante destacar que nesse estudo não há presença de pessoas não fumantes, mostrando o quanto o tabagismo é altamente relacionado à problemas cardiovasculares.


## Ferramentas utilizadas

A principal ferramenta de software utilizada foi a aplicação da linguagem Python no ambiente Google Colab em conjunto com o Power BI, empregada para coletar métricas que apoiarão nas análises dos pacientes do dataset . As métricas foram realizadas atráves da linguagem DAX(Data Analysis Expressions). Também foi utilizado o Power Query, ferramenta ETL para limpeza e preparação de dados do Power BI.

