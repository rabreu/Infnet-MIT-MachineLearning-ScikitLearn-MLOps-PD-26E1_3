# Objetivo Geral do Projeto

Estruturar um projeto completo de machine learning, integrando planejamento, fundação de dados, experimentação, validação, redução de dimensionalidade, rastreamento de experimentos e operacionalização, utilizando scikit-learn e MLflow, com foco em tomada de decisão técnica e impacto de negócio.

---

# Contexto do Projeto

Você deve assumir que:

- O problema de negócio, o dataset e os experimentos iniciais já foram explorados no Projeto de Disciplina anterior.
- Existem múltiplas abordagens de modelagem já testadas.
- O desafio agora é organizar, sistematizar e escalar tecnicamente esse trabalho.

O mesmo dataset deve ser reutilizado ao longo de todo o projeto, permitindo comparações consistentes entre abordagens, aplicação de técnicas de redução de dimensionalidade e análise da evolução dos modelos.

---

# Parte 1 — Estruturação do Projeto de Machine Learning

Nesta etapa, você deve reorganizar o trabalho realizado anteriormente, assumindo explicitamente a perspectiva de um engenheiro de machine learning.

O objetivo não é criar novos modelos, mas dar forma de engenharia ao projeto, reduzindo a dependência de notebooks exploratórios e estabelecendo uma base estruturada para experimentação e operação.

## Você deve:

- Mapear os experimentos já realizados, identificando modelos testados, métricas utilizadas, principais resultados e limitações.
- Definir de forma explícita o objetivo técnico do projeto, critérios de sucesso e métricas de negócio associadas.
- Reestruturar o projeto em código, garantindo que a lógica principal de preparação, treinamento e validação esteja em scripts ou módulos reutilizáveis, e que notebooks sejam usados apenas para exploração ou visualização.
- Analisar os dados sob a ótica de engenharia, apontando riscos iniciais de qualidade, viés e generalização.

**Objetivo da etapa:** demonstrar que você compreende machine learning como um processo de engenharia, e não apenas como execução de algoritmos.

---

# Parte 2 — Fundação de Dados e Diagnóstico Inicial

Nesta etapa, você deve estabelecer a base de dados confiável sobre a qual todos os experimentos serão conduzidos.

## Você deve:

- Estruturar a ingestão de dados, definindo fontes, formatos e estratégias de amostragem.
- Diagnosticar problemas de qualidade de dados, como valores ausentes, ruído, inconsistências e possíveis vieses.
- Analisar o impacto desses problemas na generalização, estabilidade dos resultados e risco de overfitting.
- Documentar limitações estruturais do dataset que não possam ser corrigidas apenas com modelagem.

**Objetivo da etapa:** garantir que a experimentação subsequente seja baseada em dados compreendidos, controlados e tecnicamente defensáveis.

---

# Parte 3 — Experimentação Sistemática de Modelos

Com a base de dados estruturada e compreendida, você deve conduzir experimentos controlados de modelagem.

## Você deve:

- Executar experimentos comparativos entre abordagens candidatas já exploradas no projeto anterior.
- Selecionar modelos considerando desempenho preditivo, custo computacional, complexidade e interpretabilidade.
- Construir pipelines end-to-end de preparação de dados, treinamento e validação utilizando scikit-learn.
- Ajustar modelos com validação cruzada e busca de hiperparâmetros.
- Registrar todos os experimentos no MLflow, incluindo parâmetros, métricas e versões de modelos.

**Objetivo da etapa:** transformar exploração em evidência experimental comparável, capaz de sustentar decisões técnicas.

---

# Parte 4 — Controle de Complexidade e Redução de Dimensionalidade

Nesta etapa, o foco é o controle consciente da complexidade do modelo, do custo computacional e da generalização.

## Você deve:

- Analisar a necessidade de redução de dimensionalidade com base nos resultados experimentais obtidos anteriormente.
- Escolher e aplicar duas técnicas de redução de dimensionalidade, dentre PCA, LDA e t-SNE, justificando explicitamente a escolha de cada uma em função das características dos dados e do objetivo do modelo.

Para cada técnica escolhida:

- Integrar a redução de dimensionalidade ao pipeline de modelagem.
- Treinar novamente os classificadores.

Comparar o desempenho dos modelos com e sem redução de dimensionalidade, analisando:

- Impacto no resultado final da classificação.
- Custo computacional de treinamento e inferência.
- Efeitos sobre a interpretabilidade do modelo.

Discutir os trade-offs observados e justificar se a redução de dimensionalidade é ou não adequada ao contexto do problema.

**Objetivo da etapa:** demonstrar domínio técnico sobre dimensionalidade, overfitting e eficiência do modelo.

---

# Parte 5 — Consolidação Experimental e Seleção Final

Nesta etapa, você deve consolidar os resultados experimentais e justificar tecnicamente a escolha do modelo final.

## Você deve:

- Analisar comparativamente os experimentos registrados no MLflow.
- Justificar a seleção da abordagem final com base em métricas técnicas, custo computacional, complexidade e viabilidade de operação.
- Definir explicitamente o modelo candidato à operação.

**Objetivo da etapa:** fechar o ciclo experimental com uma decisão técnica clara e justificável.

---

# Parte 6 — Operacionalização e Simulação de Produção

Na etapa final, você deve simular ou implementar a operação do modelo selecionado.

## Você deve:

- Persistir modelos treinados em scikit-learn de forma versionada.
- Executar inferência consistente a partir de modelos persistidos.
- Empacotar modelos como artefatos de inferência.
- Expor o modelo por meio de um serviço simples de inferência.
- Integrar o deploy do modelo a um pipeline de CI/CD simulado ou real.
- Definir métricas técnicas do modelo e métricas de impacto de negócio.
- Analisar desempenho pós-deploy.
- Detectar drift de dados e de modelo por meio de comparação estatística.
- Monitorar métricas e versões no MLflow.
- Planejar estratégias de re-treinamento e aprendizado contínuo.

**Objetivo da etapa:** demonstrar que você compreende machine learning como um sistema vivo em produção, sujeito a degradação, mudança de dados e necessidade de monitoramento contínuo.

---

# Entregáveis

Você deve entregar:

- Um repositório organizado contendo pipelines de dados e modelos, código de experimentação e configuração do MLflow.
- Um relatório técnico estruturado, contendo decisões de projeto, análise comparativa de experimentos e justificativa da abordagem final.
- Uma demonstração funcional ou simulação de operação do modelo, incluindo inferência, versionamento e monitoramento (vídeo).

**Observação:** código funcional isoladamente não é suficiente. A avaliação considera fortemente estrutura, rastreabilidade, interpretação e decisões técnicas.

---

# Considerações Finais

Este Projeto de Disciplina consolida a transição do aluno de executor de modelos para engenheiro de machine learning. O foco não está em maximizar métricas isoladas, mas em demonstrar visão sistêmica, maturidade técnica e responsabilidade profissional na construção e operação de sistemas de machine learning.