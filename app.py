import numpy as np
import pandas as pd
import streamlit as st
from new_recsys import *

pd.set_option('display.max_rows', 100)

# Carrega os recursos e a matriz X (pré-treinada offline)
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()
Ynorm, Ymean = normalizeRatings(Y, R)

movieList, movieList_df = load_Movie_List_pd()

df_filmes = pd.DataFrame({'title_ptbr': movieList, 'nota': 0})

# Interface Streamlit
st.title(
    'Projeto de Extensão Universitária: Sistema de Recomendação baseado em'
    ' Aprendizado de Máquina'
)
st.write(
    'Navegue ou pesquise na tabela e atribua notas de 1 a 5 apenas nos filmes'
    ' que assistiu:'
)

df_editado = st.data_editor(
    df_filmes,
    column_config={
        'title_ptbr': st.column_config.TextColumn('Título do Filme', disabled=True),
        'nota': st.column_config.SelectboxColumn(
            'Sua Nota (1-5)',
            options=[0, 1, 2, 3, 4, 5],
            default=0,
            help='Altere para a nota desejada. Mantenha 0 se não assistiu.',
        ),
    },
    hide_index=True,
    use_container_width=True,
    height=500,
)

vetor_notas = np.array(df_editado['nota'].tolist())

st.divider()

if st.button('Salvar Avaliações', type='primary'):
  total_avaliados = np.sum(vetor_notas > 0)
  st.success(f'Avaliações salvas! Você avaliou {total_avaliados} filme(s).')

  st.write('### Detalhes de suas avaliações:')
  st.write(f'Tamanho total da lista: `{len(vetor_notas)}`')

  for idx in range(len(df_editado)):
    if vetor_notas[idx] > 0:
      st.write(
          'ID do filme: ',
          idx,
          'Título: ',
          df_editado.loc[idx, 'title_ptbr'],
          'Nota dada: ',
          df_editado.loc[idx, 'nota'],
      )

  # ==============================================================================
  # INFERÊNCIA EM TEMPO REAL COM NUMPY (Substituição do TensorFlow)
  # ==============================================================================
  rated_indices = np.where(vetor_notas > 0)[0]

  if len(rated_indices) > 0:
    # 1. Normalização das notas fornecidas pelo usuário
    user_ratings = vetor_notas[rated_indices]
    Y_user_norm = user_ratings - Ymean[rated_indices].flatten()

    # 2. Solução analítica do vetor W_novo (Mínimos Quadrados / Ridge Regression)
    X_rated = X[rated_indices]
    n_feat = X.shape[1]
    lambda_reg = 1.0

    A = np.dot(X_rated.T, X_rated) + lambda_reg * np.eye(
        n_feat, dtype=np.float64
    )
    V = np.dot(X_rated.T, Y_user_norm)

    # Resolve o sistema linear A * W_novo = V em < 0.001s
    W_novo = np.linalg.solve(A, V)

    # 3. Predição de notas para todos os filmes do catálogo
    my_predictions = np.dot(X, W_novo) + Ymean.flatten()
  else:
    # Se nenhuma nota for fornecida, utiliza as médias gerais
    my_predictions = Ymean.flatten()
  # ==============================================================================

  st.header('Filmes recomendados e previsão de avaliação do usuário:')

  # Ordenação decrescente com NumPy
  ix = np.argsort(my_predictions)[::-1]
  i = 0
  l = 0
  while i < 20 and l < len(ix):
    j = ix[l]
    if (
        vetor_notas[j] == 0
        and movieList_df['number of ratings'].iloc[int(j)] > 30
    ):
      st.write(
          f'Previsão de avaliação {my_predictions[j]:0.2f} para o filme'
          f' {movieList[j]}'
      )
      i += 1
    l += 1

  st.header('Filmes não recomendados e previsão de avaliação do usuário:')

  # Ordenação crescente com NumPy
  ix = np.argsort(my_predictions)
  i = 0
  l = 0
  while i < 20 and l < len(ix):
    j = ix[l]
    if (
        vetor_notas[j] == 0
        and movieList_df['number of ratings'].iloc[int(j)] > 20
    ):
      st.write(
          f'Previsão de avaliação {my_predictions[j]:0.2f} para o filme'
          f' {movieList[j]}'
      )
      i += 1
    l += 1

  st.header('Avaliação dada comparada ao do modelo treinado:')
  for i in range(len(vetor_notas)):
    if vetor_notas[i] > 0:
      st.subheader(f'Filme: {movieList[i]}')
      st.write('Nota dada pelo usuário: ', vetor_notas[i])
      st.write('Nota prevista: ', my_predictions[i])