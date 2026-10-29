# people-analytics-turnover
Análise de turnover em base sintética de RH com Python: limpeza de dados, perguntas de negócio, Pandas, scikit-learn e conclusões.

# Análise de Turnover em RH com Python

Projeto de portfólio de análise de dados: limpeza, tratamento e análise de uma base de RH com foco em **turnover (rotatividade de funcionários)**, respondendo perguntas de negócio como um time de People Analytics faria.

> **Sobre os dados:** a base `base_rh_bruta.csv` é **sintética**, gerada com auxílio do Claude (Anthropic) para simular uma base real de RH, com problemas propositais de qualidade (nulos, grafias diferentes, valores inválidos). Nenhuma informação de pessoas reais é usada. Os resultados servem para praticar a metodologia e **não representam nenhuma empresa real**.

## Objetivos

- Praticar limpeza e tratamento de dados em uma base com formato próximo ao real.
- Responder perguntas de negócio de RH usando análise exploratória e modelos simples.
- Documentar decisões, limitações e cuidados estatísticos de cada análise.

## Tecnologias

- Python 3
- pandas e NumPy
- matplotlib
- scikit-learn (`LinearRegression`, `LogisticRegression`, `StandardScaler`)

## Estrutura do projeto

```
.
├── base_rh_bruta.csv      # base bruta (separador ";")
├── analise_turnover.py    # script de limpeza e análise (células %% [1] a [25])
└── README.md
```

## Como executar

```bash
pip install pandas numpy matplotlib scikit-learn
python analise_turnover.py
```

O script está dividido em células (`# %%`), então também pode ser executado célula a célula no VS Code ou Spyder. **Rode sempre do início ao fim**, a partir de um `read_csv` novo, pois cada coluna é limpa uma única vez.

## Decisões de limpeza

| Tema | Decisão |
|---|---|
| Desligamento (data, tipo, motivo) | Nulo significa **funcionário ativo**. Não é dado faltante e não é preenchido. |
| Datas | Convertidas com `dayfirst=True`; nunca preenchidas com `ffill` ou com a data de hoje. |
| Salário | Limpeza única: remoção de `R$` e espaços, tratamento do formato brasileiro (ponto de milhar e vírgula decimal). |
| Gênero | Grafias padronizadas; nulos viram "Não informado" (sem imputar pela moda). |
| Horas extras, absenteísmo, promoção, clima, salário | **Não imputados**: nulo continua nulo. Cada análise usa `dropna` e informa quantas linhas ficaram de fora. |
| Datas inconsistentes | Desligamento anterior à admissão é identificado e excluído dos cálculos de tempo de casa. |

## Perguntas de negócio respondidas

1. **Turnover por ano e mês:** está aumentando?
2. **Maior turnover** por departamento, cargo e cidade.
3. **Voluntário x involuntário** e principais motivos.
4. **Turnover precoce:** saídas com menos de 1 ano de casa.
6. **Clima x saída:** a satisfação com o clima se relaciona com o desligamento? Quem não respondeu à pesquisa se comporta diferente?
7. **Modelo de trabalho** (presencial, híbrido, remoto) x permanência.
8. **Diferença salarial** entre homens e mulheres (bruta, controlada por grupos e por regressão).
9. **Distância, horas extras e tempo sem promoção** x saída (quartis e regressão logística).
10. **Salário abaixo da média do cargo** aumenta a saída?

## Principais resultados

- **Tendência do turnover:** caiu de 13-14% (2019-2020) para 9,6% em 2021, subiu até 11,2% em 2024 e ficou estável em 11,1% em 2025. 2026 cobre só parte do ano.
- **Tipo de saída:** 69,8% voluntárias e 29,0% involuntárias. Entre as voluntárias, os principais motivos são melhor proposta externa (21,3%), falta de crescimento (19,2%) e insatisfação com o gestor (18,1%).
- **Turnover precoce:** 282 desligados (31,9% dos que têm datas válidas) saíram com menos de 1 ano, concentrados em TI, Comercial, Operações e Financeiro.
- **Clima:** forte associação com a saída. O turnover vai de cerca de 56% nas notas 0 a 3 para 16% nas notas 8 a 10. Quem não respondeu à pesquisa tem turnover de 52,0%, contra 28,3% de quem respondeu.
- **Modelo de trabalho:** presencial 36,3%, híbrido 32,2% e remoto 30,8%, uma diferença modesta.
- **Diferença salarial de gênero:** homens ganham 1,8% a mais na média bruta e 2,0% na regressão com controles. A diferença é pequena, mas não desaparece após os controles.
- **Distância e horas extras:** o turnover sobe com ambas (29,1% para 38,7% entre o 1º e o último quartil de distância; 26,3% para 41,4% em horas extras).
- **Salário relativo ao cargo:** sem associação com a saída (33,2% abaixo da média do cargo contra 33,6% acima).

## Limitações e cuidados

- Os resultados mostram **associação, não causalidade**. Por exemplo, a nota de clima pode ter sido dada perto da saída.
- Não foram calculados p-valores nem testes de significância.
- Categorias não padronizadas (ex.: grafias diferentes de um mesmo departamento) dividem os grupos. O ranking da pergunta 2 deve ser refeito após a padronização.
- Admissões sem data ficam fora do headcount, o que tende a superestimar o turnover.
- Na regressão logística (pergunta 9), só 1.931 das 3.028 linhas foram usadas por causa dos nulos. O sinal de "meses desde a última promoção" ficou contrário ao esperado, o que merece investigação.
- Há outliers de salário que não foram tratados.

## Próximos passos

- Padronizar categorias de departamento, cargo e cidade e refazer a pergunta 2.
- Analisar gestores com turnover acima da média e seus motivos de desligamento (pergunta 5).
- Adicionar testes de significância (qui-quadrado, teste t) e tratamento de outliers.
- Construir um modelo preditivo de desligamento com validação cruzada.
- Criar um dashboard no Power BI com os principais indicadores.

## Transparência sobre o uso de IA

Este projeto foi desenvolvido por mim como exercício de aprendizado. A **base de dados foi gerada com ajuda do Claude**, e recebi **sugestões e ajustes do Claude** em partes do código e das análises. As perguntas de negócio, a execução e a interpretação dos resultados foram trabalhadas por mim. Registro isso de forma aberta por considerar o uso de IA uma ferramenta legítima de estudo, desde que seja declarado.

## Autor

**Alex Cruz**
Estudante de Ciência de Dados (Univesp)
São Paulo, SP

- LinkedIn: _adicione o link_
- GitHub: _adicione o link_
