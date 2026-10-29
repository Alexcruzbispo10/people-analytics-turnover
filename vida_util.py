# ==============================================================================
# LIMPEZA E ANÁLISE DE TURNOVER - BASE RH (VERSÃO CORRIGIDA)
# ==============================================================================
# Regras principais deste script:
#   - Nulo em data_desligamento / tipo_desligamento / motivo_desligamento
#     significa "funcionário ainda ativo". NÃO é dado faltante e NÃO se preenche.
#   - Datas nunca são preenchidas com ffill ou com a data de hoje.
#   - Salário, gênero, clima, horas extras, absenteísmo e promoção NÃO são
#     imputados: nulo continua nulo. Cada análise decide como tratar (dropna)
#     e informa quantas linhas ficaram de fora.
#   - A limpeza de cada coluna acontece UMA vez, na etapa de limpeza.
#   - Rode sempre do início ao fim, a partir de um read_csv novo.
# ==============================================================================

# %% [1] Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

# %% [2] Carregamento da base bruta (separador ; comum em CSV brasileiro)
df = pd.read_csv('base_rh_bruta.csv', sep=';')

# %% [3] Exploração inicial
print(df.head())
print('Linhas:', len(df))
print('Colunas:', df.columns.tolist())
df.info()
print(df.isnull().sum())
print('Linhas duplicadas:', df.duplicated().sum())

# %% [4] Colunas de desligamento: NÃO remover (são a base do turnover)

# %% [5] Tratamento de colunas categóricas e numéricas simples
# Gênero: padroniza as grafias; nulo ou valor desconhecido vira 'Não informado'
# (não se preenche com a moda: isso inventaria o gênero de pessoas reais)
df['genero'] = (
    df['genero'].astype('string').str.strip().str.lower()
    .map({'m': 'Masculino', 'masculino': 'Masculino',
          'f': 'Feminino', 'feminino': 'Feminino'})
    .fillna('Não informado')
)
print(df['genero'].value_counts())

# gestor_id: 0 = sem gestor identificado
df['gestor_id'] = df['gestor_id'].fillna(0)

# Categóricas: 'não informado' mantém a categoria visível nas análises
for col in ['escolaridade', 'estado_civil', 'modelo_trabalho']:
    df[col] = df[col].fillna('não informado')

# Dependentes: nulo tratado como 0 e convertido para inteiro
df['num_dependentes'] = df['num_dependentes'].fillna(0).astype(int)

# Horas extras, absenteísmo e meses desde a última promoção:
# NÃO imputados. Zero significaria "não fez hora extra / acabou de ser promovido",
# o que pode não ser verdade e criaria relações falsas. Ficam como NaN.
for col in ['horas_extras_mes', 'absenteismo_dias_12m',
            'meses_desde_ultima_promocao', 'distancia_casa_trabalho_km']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# %% [6] Nota de avaliação de desempenho
# Textos inválidos (ex.: '-') viram NaN; não se imputa
df['nota_avaliacao_desempenho'] = pd.to_numeric(df['nota_avaliacao_desempenho'], errors='coerce')

# %% [7] Satisfação com o clima: NÃO imputar
# A flag marca quem respondeu a pesquisa. Nulo = não respondeu.
df['satisfacao_clima_0a10'] = pd.to_numeric(df['satisfacao_clima_0a10'], errors='coerce')
df['respondeu_clima'] = df['satisfacao_clima_0a10'].notna()
print('Responderam:', df['respondeu_clima'].sum(),
      '| Não responderam:', (~df['respondeu_clima']).sum())

# %% [8] Salário base: limpeza feita UMA vez
# Remove R$ e espaços. Quando há vírgula (formato brasileiro), remove o ponto de
# milhar e troca a vírgula por ponto. Valores sem vírgula (ex.: '5190.37') ficam como estão,
# assim o ponto decimal não é apagado por engano.
s = df['salario_base'].astype(str).str.replace(r'[R$\s]', '', regex=True)
tem_virgula = s.str.contains(',')
s = s.where(~tem_virgula,
            s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False))
df['salario_base'] = pd.to_numeric(s, errors='coerce')   # NaN fica NaN

print('Salários inválidos/nulos:', df['salario_base'].isna().sum())
print(df['salario_base'].describe())   # confira se min/max são plausíveis

# %% [9] Datas: converter SEM preencher
df['data_admissao'] = pd.to_datetime(df['data_admissao'], dayfirst=True, errors='coerce')
df['data_desligamento'] = pd.to_datetime(df['data_desligamento'], dayfirst=True, errors='coerce')

# Nulo em data_desligamento = ainda ativo
df['desligado'] = df['data_desligamento'].notna()

# Tempo de casa (anos): até a saída para desligados; até a data de referência para ativos
data_ref = max(df['data_admissao'].max(), df['data_desligamento'].max())
fim_vinculo = df['data_desligamento'].fillna(data_ref)
df['tempo_casa_anos'] = (fim_vinculo - df['data_admissao']).dt.days / 365.25

print('Admissões sem data (ficam fora dos cálculos de tempo de casa):',
      df['data_admissao'].isna().sum())
print('Desligados:', df['desligado'].sum(), '| Ativos:', (~df['desligado']).sum())

# Datas inconsistentes: desligamento antes da admissão
inconsistentes = df[df['data_desligamento'] < df['data_admissao']]
print('Datas inconsistentes (desligamento < admissão):', len(inconsistentes))

# %% [10] Exploração das admissões
print('Data de admissão mais frequente:', df['data_admissao'].mode()[0])
print('Admissões por ano:')
print(df['data_admissao'].dt.year.value_counts().sort_index())

# Quantidade de cargos diferentes por cidade
print(df.groupby('cidade')['cargo'].nunique().sort_values(ascending=False))

# %% [11] Validação final
# Nulos esperados: colunas de desligamento (ativos) e as colunas NÃO imputadas
print(df.isnull().sum())

# ==============================================================================
# ANÁLISE
# ==============================================================================

# %% [12] Pergunta 1: turnover por ano e mês. Está aumentando?
# Turnover (%) = desligamentos no período / headcount médio do período * 100
inicio = df['data_admissao'].min()
fim = max(df['data_admissao'].max(), df['data_desligamento'].max())

linhas = []
for m in pd.period_range(inicio, fim, freq='M'):
    ini, final = m.start_time, m.end_time

    hc_inicio = ((df['data_admissao'] < ini) &
                 (df['data_desligamento'].isna() | (df['data_desligamento'] >= ini))).sum()
    hc_fim = ((df['data_admissao'] <= final) &
              (df['data_desligamento'].isna() | (df['data_desligamento'] > final))).sum()
    saidas = df['data_desligamento'].between(ini, final).sum()

    linhas.append({
        'periodo': m,
        'ano': m.year,
        'mes': m.month,
        'desligamentos': saidas,
        'headcount_medio': (hc_inicio + hc_fim) / 2,
    })

turnover_mensal = pd.DataFrame(linhas)
turnover_mensal['turnover_%'] = (
    turnover_mensal['desligamentos'] / turnover_mensal['headcount_medio'] * 100
)
# Média móvel de 12 meses: suaviza oscilações e mostra a tendência
turnover_mensal['media_movel_12m'] = turnover_mensal['turnover_%'].rolling(12).mean()

turnover_anual = turnover_mensal.groupby('ano').agg(
    desligamentos=('desligamentos', 'sum'),
    headcount_medio=('headcount_medio', 'mean'),
)
turnover_anual['turnover_%'] = (
    turnover_anual['desligamentos'] / turnover_anual['headcount_medio'] * 100
)
print(turnover_anual)
# Atenção: o último ano pode estar incompleto e aparecer artificialmente menor.

x = turnover_mensal['periodo'].dt.to_timestamp()
plt.figure(figsize=(10, 4))
plt.plot(x, turnover_mensal['turnover_%'], alpha=0.4, label='Mensal')
plt.plot(x, turnover_mensal['media_movel_12m'], linewidth=2, label='Média móvel 12 meses')
plt.title('Turnover mensal (%)')
plt.legend()
plt.tight_layout()
plt.grid()
plt.xticks(rotation=45)
plt.show()

# %% [13] Pergunta 2: departamentos, cargos e cidades com maior turnover
# Turnover do grupo = desligados / total de pessoas que já passaram pelo grupo
for col in ['departamento', 'cargo', 'cidade']:
    r = df.groupby(col)['desligado'].agg(desligados='sum', total='count')
    r['turnover_%'] = r['desligados'] / r['total'] * 100
    # Evita distorção de grupos muito pequenos (1 de 2 = 50%)
    r = r[r['total'] >= 10]
    print(f'\nMaior turnover por {col}:')
    print(r.sort_values('turnover_%', ascending=False).head(10))

# %% [14] Pergunta 3: voluntário x involuntário e principais motivos
# Aqui só entram os desligados. Nulo nessas colunas, para um desligado,
# é de fato dado faltante e pode virar 'não informado'.
desligados = df[df['desligado']].copy()
desligados['tipo_desligamento'] = desligados['tipo_desligamento'].fillna('não informado')
desligados['motivo_desligamento'] = desligados['motivo_desligamento'].fillna('não informado')

print('\nProporção por tipo de desligamento (%):')
print(desligados['tipo_desligamento'].value_counts(normalize=True) * 100)

print('\nMotivos dentro de cada tipo (%):')
print(desligados.groupby('tipo_desligamento')['motivo_desligamento']
      .value_counts(normalize=True) * 100)

print('\nPrincipais motivos no geral (%):')
print(desligados['motivo_desligamento'].value_counts(normalize=True).head(10) * 100)

# %% [15] Pergunta 4: turnover precoce (menos de 1 ano de casa)
desligados['tempo_casa_dias'] = (
    desligados['data_desligamento'] - desligados['data_admissao']
).dt.days

# Registros sem data de admissão (ou datas inconsistentes) ficam fora da conta
validos = desligados[desligados['tempo_casa_dias'].notna() & (desligados['tempo_casa_dias'] >= 0)].copy()
validos['tipo_turnover'] = np.where(validos['tempo_casa_dias'] < 365, 'Precoce', 'Normal')

precoce = validos[validos['tipo_turnover'] == 'Precoce']
print(f'\nSaíram com menos de 1 ano de casa: {len(precoce)} '
      f'({len(precoce) / len(validos) * 100:.1f}% dos desligados com datas válidas)')

# Onde o turnover precoce se concentra
print(precoce['departamento'].value_counts().head(5))

# %% [16] Pergunta 6: clima x saída. Quem não respondeu se comporta diferente?
# 6.1 Taxa de desligamento por nota (só quem respondeu)
clima = (
    df[df['respondeu_clima']]
    .groupby('satisfacao_clima_0a10')['desligado']
    .agg(total='count', desligados='sum')
)
clima['turnover_%'] = (clima['desligados'] / clima['total'] * 100).round(1)
print(clima)

# 6.2 Respondeu x não respondeu
resp = df.groupby('respondeu_clima')['desligado'].agg(total='count', desligados='sum')
resp['turnover_%'] = (resp['desligados'] / resp['total'] * 100).round(1)
print('\nRespondeu (True) x Não respondeu (False):')
print(resp)

clima['turnover_%'].plot(kind='bar', figsize=(8, 4))
plt.title('Turnover (%) por nota de clima (apenas quem respondeu)')
plt.xlabel('Nota de clima (0 a 10)')
plt.ylabel('Turnover (%)')
plt.tight_layout()
plt.show()

# %% [17] Pergunta 7: modelo de trabalho x permanência
# Proporção dentro de cada modelo (comparável entre grupos de tamanhos diferentes)
modelo = pd.crosstab(df['modelo_trabalho'], df['desligado'], normalize='index') * 100
print(modelo.round(1))
print('\nTamanho de cada grupo:')
print(df['modelo_trabalho'].value_counts())

if True in modelo.columns:
    modelo[True].sort_values().plot(kind='barh', figsize=(7, 3))
    plt.title('Turnover (%) por modelo de trabalho')
    plt.xlabel('Turnover (%)')
    plt.tight_layout()
    plt.show()

# %% [18] Pergunta 8: diferença salarial entre homens e mulheres
# Só Masculino/Feminino com salário válido (sem imputação)
sub = df[df['genero'].isin(['Masculino', 'Feminino'])].dropna(subset=['salario_base']).copy()
print('Linhas usadas:', len(sub), 'de', len(df))

# 8.1 Diferença bruta
bruta = sub.groupby('genero')['salario_base'].agg(['count', 'mean', 'median'])
print('=== DIFERENÇA BRUTA ===')
print(bruta.round(2))
dif_bruta = (bruta.loc['Masculino', 'mean'] / bruta.loc['Feminino', 'mean'] - 1) * 100
print(f'Diferença bruta (média): homens ganham {dif_bruta:.2f}% em relação às mulheres')

# %% [19] 8.2 Diferença controlada por cargo, departamento e escolaridade
fatores_controle = ['cargo', 'departamento', 'escolaridade']
g = (
    sub.groupby(fatores_controle + ['genero'])['salario_base']
    .agg(['count', 'mean'])
    .unstack('genero')
    .dropna()
)
# Exige pelo menos 5 pessoas de cada gênero em cada combinação
g = g[(g[('count', 'Masculino')] >= 5) & (g[('count', 'Feminino')] >= 5)]

if len(g) == 0:
    print('Nenhuma combinação com pelo menos 5 pessoas de cada gênero.')
else:
    m = g[('mean', 'Masculino')]
    f = g[('mean', 'Feminino')]
    dif = (m / f - 1) * 100
    peso = g[('count', 'Masculino')] + g[('count', 'Feminino')]
    dif_pond = np.average(dif, weights=peso)
    print(f'Combinações comparáveis: {len(g)}')
    print(f'Diferença média ponderada pelo tamanho do grupo: {dif_pond:.2f}%')
    print(dif.describe().round(2))

# %% [20] 8.3 Regressão: controla cargo, departamento, escolaridade, idade e tempo de casa
# (experiência aproximada por idade e tempo de casa; são proxies)
cat = ['genero', 'cargo', 'departamento', 'escolaridade']
num = ['idade', 'tempo_casa_anos']
dados = sub[cat + num + ['salario_base']].dropna()
dados = dados[dados['salario_base'] > 0]
print('Linhas na regressão:', len(dados), 'de', len(sub))

X = pd.get_dummies(dados[cat + num], columns=cat, drop_first=True, dtype=float)
y = np.log(dados['salario_base'])

reg = LinearRegression().fit(X, y)
coef_genero = pd.Series(reg.coef_, index=X.columns)['genero_Masculino']
dif_controlada = (np.exp(coef_genero) - 1) * 100

print(f'\nDiferença controlada (regressão): homens ganham {dif_controlada:.2f}% '
      f'em relação às mulheres, com cargo, departamento, escolaridade, idade e tempo de casa iguais.')

# %% [21] Veredito da pergunta 8 (texto depende do resultado)
def descreve(dif):
    if abs(dif) < 1:
        return 'praticamente não há diferença (menos de 1%)'
    quem = 'homens ganham mais' if dif > 0 else 'mulheres ganham mais'
    return f'{quem} ({abs(dif):.1f}%)'

print('Bruta:      ', descreve(dif_bruta))
print('Controlada: ', descreve(dif_controlada))
# Se a bruta for grande e a controlada pequena, a diferença se explica pela composição
# (cargos, escolaridade etc.). Se a controlada continua relevante, ela persiste.
# Obs.: sem p-valor, não dá para afirmar significância estatística.

# %% [22] Pergunta 9: distância, horas extras e tempo sem promoção x saída
cols_9 = ['distancia_casa_trabalho_km', 'horas_extras_mes', 'meses_desde_ultima_promocao']

# 9.1 Desligados x ativos
print(df.groupby('desligado')[cols_9].agg(['count', 'mean', 'median']).round(2))

# 9.2 Taxa de saída por quartil
for c in cols_9:
    faixas = pd.qcut(df[c], q=4, duplicates='drop')
    taxa = df.groupby(faixas, observed=True)['desligado'].agg(total='count', turnover='mean')
    taxa['turnover'] = (taxa['turnover'] * 100).round(1)
    print(f'\n{c}')
    print(taxa)

# %% [23] 9.3 Regressão logística
dados9 = df[cols_9 + ['desligado']].dropna()
print('Linhas usadas:', len(dados9), 'de', len(df))

X9 = StandardScaler().fit_transform(dados9[cols_9])
y9 = dados9['desligado'].astype(int)
logit = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X9, y9)

res = pd.DataFrame({'variavel': cols_9, 'odds_ratio': np.exp(logit.coef_[0]).round(3)})
print(res)   # odds ratio por 1 desvio-padrão; acima de 1 = aumenta a chance de sair

plt.figure(figsize=(7, 4))
plt.bar(res['variavel'], res['odds_ratio'])
plt.axhline(1, color='red', linestyle='--')
plt.ylabel('Odds ratio')
plt.title('Efeito de cada variável sobre a chance de saída')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()

# %% [24] Pergunta 10: quem ganha abaixo da média do próprio cargo sai mais?
base10 = df.dropna(subset=['salario_base']).copy()

# Só cargos com pelo menos 10 pessoas (média de cargo pequeno é pouco confiável)
tamanho_cargo = base10.groupby('cargo')['salario_base'].transform('count')
base10 = base10[tamanho_cargo >= 10].copy()
print('Linhas usadas:', len(base10), 'de', len(df))

base10['media_cargo'] = base10.groupby('cargo')['salario_base'].transform('mean')
base10['abaixo_media'] = base10['salario_base'] < base10['media_cargo']

abaixo = base10.groupby('abaixo_media')['desligado'].agg(total='count', desligados='sum')
abaixo['turnover_%'] = (abaixo['desligados'] / abaixo['total'] * 100).round(1)
print(abaixo)

# Versão gradual: posição salarial relativa dentro do cargo
base10['razao_salarial'] = base10['salario_base'] / base10['media_cargo']
base10['faixa_salarial'] = pd.qcut(
    base10['razao_salarial'], q=4,
    labels=['Q1 (mais abaixo)', 'Q2', 'Q3', 'Q4 (mais acima)']
)
faixas10 = base10.groupby('faixa_salarial', observed=True)['desligado'].mean() * 100
print('\nTurnover (%) por faixa de salário relativo ao cargo:')
print(faixas10.round(1))

faixas10.plot(kind='bar', figsize=(7, 4))
plt.title('Turnover (%) por posição salarial dentro do cargo')
plt.ylabel('Turnover (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
# %% [25] Conclusões
# P1: O turnover anual caiu de 13-14% (2019-2020) para 9,6% em 2021 e subiu aos poucos
#     até 11,2% em 2024, ficando estável em 11,1% em 2025. Ou seja, subiu entre 2021 e 2024,
#     mas não continua subindo. 2026 (4,95%) cobre só metade do ano, cerca de 10% anualizado.
#     Limite: as admissões sem data ficam fora do headcount, o que tende a superestimar o turnover.
# P2: Os departamentos, cargos e cidades do topo ficam numa faixa estreita (cerca de 32% a 40%),
#     sem um grupo que se destaque. As grafias não padronizadas (ti, T.I., Tecnologia da Informação)
#     dividem os grupos, então esse ranking precisa ser refeito depois da padronização.
# P3: 69,8% dos desligamentos são voluntários e 29,0% involuntários. Entre os voluntários, os
#     principais motivos são melhor proposta externa (21,3%), falta de crescimento (19,2%) e
#     insatisfação com o gestor (18,1%). Entre os involuntários, predominam baixo desempenho
#     (35,3%) e reestruturação (29,8%). Limite: 21,4% dos motivos não foram informados.
# P4: 282 desligados (31,9% dos que têm datas válidas) saíram com menos de 1 ano de casa,
#     concentrados em TI (59), Comercial (52), Operações (35) e Financeiro (35).
# P6: Sim, o clima está fortemente associado à saída. O turnover vai de cerca de 56% nas notas
#     0 a 3 para 40% nas notas 4 e 5, 25% nas notas 6 e 7 e 16% nas notas 8 a 10 (8,5% na nota 10).
#     Quem não respondeu à pesquisa tem turnover de 52,0%, contra 28,3% de quem respondeu.
#     Limite: a nota pode ter sido dada perto da saída, e parte de quem não respondeu pode ter
#     saído antes da pesquisa. É associação, não causa.
# P7: O turnover é de 36,3% no presencial, 32,2% no híbrido e 30,8% no remoto. A diferença é
#     modesta (5,5 pontos entre presencial e remoto), com grupos de 1.127, 1.311 e 500 pessoas.
#     Limite: não foi feito teste de significância.
# P8: Os homens ganham 1,8% a mais na média bruta, 1,4% na média ponderada por cargo,
#     departamento e escolaridade, e 2,0% na regressão que controla também idade e tempo de casa.
#     A diferença é pequena, mas não desaparece depois dos controles. Há muita variação entre
#     combinações (de -15% a +14%). Limites: sem p-valor, não dá para afirmar significância, e
#     há outliers de salário e categorias não padronizadas.
# P9: Distância, horas extras e tempo sem promoção estão associados à saída, mas o sinal da
#     promoção é contrário ao esperado. O turnover sobe com a distância (29,1% no 1º quartil
#     contra 38,7% no último) e com as horas extras (26,3% contra 41,4%). Quem saiu tem menos
#     meses desde a última promoção (média de 18,6 contra 24,4 dos ativos). Odds ratios:
#     1,151 (distância), 1,043 (horas extras) e 0,722 (promoção). Limite: só 1.931 das 3.028
#     linhas foram usadas, por causa dos nulos.
# P10: Não há associação. O turnover é de 33,2% entre quem ganha abaixo da média do cargo e
#      33,6% entre quem ganha acima. Por quartil, a variação é mínima (32,7% a 34,1%), sem padrão.
#      Nesta base, ganhar abaixo da média do cargo não aumenta a saída.