# Fundamentos de Machine Learning com Scikit-Learn [26E1_2]

Modelos de machine learning supervisionado constituem a base prática da Inteligência Artificial aplicada a problemas reais de negócio, engenharia e tomada de decisão. Antes de recorrer a modelos genéricos, modelos pré-treinados de propósito geral ou grandes modelos de linguagem, profissionais experientes constroem modelos específicos, ajustados a problemas bem definidos, nos quais desempenho, custo computacional, interpretabilidade e capacidade de generalização são critérios centrais.

## Introdução

Neste projeto, você atuará como um engenheiro de machine learning responsável por estruturar, treinar, avaliar e interpretar modelos supervisionados clássicos, utilizando exclusivamente a biblioteca scikit-learn. O foco não está apenas em alcançar bons resultados numéricos, mas em compreender profundamente o comportamento dos modelos, suas limitações e a relação entre escolhas técnicas e impacto prático.

Ao longo do projeto, você irá contrastar explicitamente modelos supervisionados bem definidos com abordagens genéricas baseadas em modelos pré-treinados, entendendo quando soluções simples, controláveis e interpretáveis são mais adequadas do que modelos complexos ou opacos.
Descrição do Problema e dos Dados

Nesta etapa, você deve apresentar de forma clara e profissional o problema que será resolvido por meio de machine learning supervisionado.

## Seu relatório deve conter:

- O contexto real do problema, descrevendo o domínio de aplicação, o ambiente organizacional ou de mercado e o tipo de decisão que será apoiada pelo modelo.
- A motivação para o uso de machine learning, explicando por que regras fixas, heurísticas simples ou análises estatísticas tradicionais não são suficientes.
- Os principais desafios do domínio, como ruído nos dados, desbalanceamento entre classes, custo assimétrico de erros, necessidade de explicabilidade ou restrições operacionais.
- A descrição técnica do dataset utilizado, incluindo:

    Origem dos dados
    Volume aproximado de observações
    Natureza das variáveis disponíveis
    A definição explícita da variável-alvo, justificando por que o problema é formulado como classificação binária.
    A descrição das features utilizadas, destacando o papel das variáveis contínuas e, se houver variáveis categóricas, justificando seu uso mediante validação do professor.

Não é permitido o uso de datasets artificiais, didáticos ou excessivamente simplificados. O dataset deve representar um problema plausível do mundo real e possuir dimensionalidade suficiente para análise multivariada, contendo no mínimo 10 features utilizáveis pelo classificador. Esse requisito é necessário para viabilizar, em etapas posteriores do bloco, a aplicação de técnicas de redução de dimensionalidade, como PCA, de forma tecnicamente consistente.

## Modelo Baseline: Classificador Linear com Perceptron

Nesta etapa, você irá construir um classificador linear simples, que servirá como modelo baseline para todo o projeto.

Em projetos profissionais de machine learning, é prática comum iniciar a análise com modelos lineares básicos, que permitem compreender explicitamente a geometria da decisão e estabelecer um ponto de comparação claro para modelos mais sofisticados. Exemplos frequentes incluem regressão logística e SVM linear. Neste projeto, entretanto, adotaremos uma abordagem fundamental utilizando o Perceptron, um classificador linear clássico que aprende diretamente um hiperplano separador no espaço das features, com base em erros de classificação.

O modelo aprendido corresponde a uma equação linear da forma:

>ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ

A decisão de classe é obtida a partir do sinal da função linear:

- classe positiva se ŷ ≥ 0
- classe negativa se ŷ < 0

Você deve:

- Construir o modelo utilizando `Perceptron` da biblioteca scikit-learn.
- Justificar conceitualmente o uso do Perceptron como classificador linear baseline.
- Separar os dados em conjuntos de treino e teste, sem uso de validação cruzada nesta etapa.
- Avaliar o classificador utilizando:

    Accuracy
    Precision
    Recall
    F1-score

- Interpretar:

    Os coeficientes aprendidos
    O papel do bias
    A orientação do hiperplano no espaço das features

- Discutir:

    As limitações do Perceptron em problemas não linearmente separáveis
    Sensibilidade a ruído e outliers
    Indícios de underfitting

O objetivo desta etapa é estabelecer um baseline linear geométrico, simples e altamente interpretável, que permita compreender claramente as limitações de classificadores lineares puros e motive a introdução de modelos probabilísticos, baseados em margens e em estruturas mais complexas nas etapas seguintes do projeto.

## Modelo com Árvore de Decisão

Nesta etapa, você deve treinar um modelo de árvore de decisão para o mesmo problema.

Seu trabalho deve incluir:

- Treinamento de uma árvore de decisão utilizando parâmetros padrão ou minimamente ajustados.
- Comparação direta do desempenho com o classificador linear baseline.
- Análise das métricas de classificação utilizadas anteriormente.
- Interpretação das regras aprendidas pelo modelo, incluindo:

    Caminhos da raiz até as folhas.
    Valores de corte das features.
    Profundidade e estrutura da árvore.
    Discussão da coerência das regras aprendidas com o conhecimento do domínio.
    Análise crítica do risco de overfitting associado à flexibilidade do modelo.

O objetivo desta etapa é introduzir modelos não lineares baseados em regras explícitas e avaliar o ganho de capacidade representacional em relação ao baseline linear.

## Validação Cruzada e Busca de Hiperparâmetros

Nesta etapa, você deve estruturar um processo sistemático de validação e otimização do modelo de árvore de decisão.

Requisitos obrigatórios:

- Implementar validação cruzada adequada ao problema.
- Definir um espaço de busca de hiperparâmetros relevante para árvores de decisão.
- Utilizar Grid Search ou Random Search, justificando a escolha.
- Comparar o modelo otimizado com a versão não otimizada da árvore.
- Analisar:

    Robustez do desempenho
    Variação das métricas entre os folds
    Impacto da regularização na estrutura da árvore

- Discutir:

    Generalização do modelo
    Mudanças na interpretabilidade após a otimização
    A noção de mínimos locais no treinamento de árvores

O foco desta etapa é demonstrar controle sobre generalização, estabilidade e complexidade do modelo.

## Modelos Avançados: SVM ou Ensembles

Você deve escolher pelo menos uma das abordagens abaixo:

- Support Vector Machine, linear ou com kernel
- Modelos ensemble, como Random Forest, Gradient Boosting ou Voting Classifier

Para o modelo escolhido, é obrigatório:

- Aplicar validação cruzada.
- Realizar busca de hiperparâmetros.
- Comparar o desempenho com todos os modelos anteriores.
- Avaliar o modelo utilizando as métricas definidas previamente.
- Interpretar o modelo sem o uso de ferramentas externas de explicabilidade.
- Analisar:

    Importância das features, quando aplicável
    Fronteira de decisão ou lógica agregada do ensemble
    Discutir o ganho de desempenho em relação ao aumento de complexidade e custo computacional.

O objetivo desta etapa é avaliar modelos com maior poder preditivo mantendo controle sobre interpretação e generalização.
Entregáveis

Você deve entregar:

1. Código completo, organizado e reprodutível, utilizando exclusivamente scikit-learn.

2. Relatório técnico estruturado, contendo:
- Justificativas das escolhas de modelos e hiperparâmetros
- Interpretação dos parâmetros aprendidos
- Discussão crítica dos resultados

3. Discussão aplicada, abordando:
- Viabilidade do modelo no contexto real
- Limitações técnicas e operacionais
- Possíveis melhorias e extensões futuras

Código funcional, isoladamente, não é suficiente. A avaliação considera fortemente a qualidade da análise, da interpretação e da tomada de decisão.