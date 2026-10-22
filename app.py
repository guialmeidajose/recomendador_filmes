import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow import keras
from recsys_utils import *

pd.set_option('display.max_rows', 100)

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

#Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

tsmean =  np.mean(Y[0, R[0, :].astype(bool)])

# Reduce the data set size so that this runs faster
num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]


# Evaluate cost function with regularization = 1.5
J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5)

movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

df_filmes = pd.DataFrame({
    "title": movieList,
    "nota": 0
})
#streamlit
st.title("Projeto de Extensão Universitária: Sistema de Recomendação baseado em Aprendizado de Máquina")

st.write("Navegue ou pesquise na tabela e atribua notas de 1 a 5 apenas nos filmes que assistiu:")

# 2. Exibição da tabela rolável compacta
df_editado = st.data_editor(
    df_filmes,
    column_config={
        "title": st.column_config.TextColumn("Título do Filme", disabled=True),
        "nota": st.column_config.SelectboxColumn(
            "Sua Nota (1-5)",
            options=[0, 1, 2, 3, 4, 5],
            default=0,
            help="Altere para a nota desejada. Mantenha 0 se não assistiu."
        )
    },
    hide_index=True,
    use_container_width=True,
    height=500  # Define a altura da caixa com barra de rolagem
)

# 3. Extração da lista final de 4000 elementos
vetor_notas = df_editado["nota"].tolist()
vetor_notas = np.array(vetor_notas)
avaliacoes = [i for i in range(len(vetor_notas)) if vetor_notas[i] > 0]


st.divider()

# 4. Botão de confirmação/processamento
if st.button("Salvar Avaliações", type="primary"):
    # Conta quantos filmes receberam nota maior que 0
    total_avaliados = sum(1 for nota in vetor_notas if nota > 0)
    
    st.success(f"Avaliações salvas! Você avaliou {total_avaliados} filme(s).")
    
    # Exemplo do vetor final obtido
    st.write("### Detalhes de suas avaliações:")
    st.write(f"Tamanho total da lista: `{len(vetor_notas)}`")
    
    # Exibe quais posições do vetor foram alteradas
    for idx in range(len(df_editado)):
        if vetor_notas[idx] > 0:
            st.write("ID do filme: ", idx, "Título: ", df_editado.loc[idx, "title"], "Nota dada: ", df_editado.loc[idx, "nota"])
    

    # Reload ratings
    Y, R = load_ratings_small()

    # Add new user ratings to Y 
    Y = np.c_[vetor_notas, Y]

    # Add new user indicator matrix to R
    R = np.c_[(vetor_notas != 0).astype(int), R]

    # Normalize the Dataset
    Ynorm, Ymean = normalizeRatings(Y, R)


    #  Useful Values
    num_movies, num_users = Y.shape
    num_features = 100

    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(1234) # for consistent results
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')


    st.header("Treinamento")
    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    iterations = 200
    lambda_ = 1
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost 
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient( cost_value, [X,W,b] )

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip(grads, [X,W,b]) )

        # Log periodically.
        if iter % 20 == 0:
            st.write(f"Função de Custo na iteração: {iter}: {cost_value:0.1f}")



    # Make a prediction using trained weights and biases
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

    #restore the mean
    pm = p + Ymean

    my_predictions = pm[:,0]

    st.header("Filmes recomandados e previsão de avaliação do usuário:")

    # sort predictions
    ix = tf.argsort(my_predictions, direction='DESCENDING')
    k = 0
    i = 0
    l = 0
    while i < 20:
        j = ix[l]
        if j not in vetor_notas and movieList_df["number of ratings"].iloc[int(j)] > 30:
            st.write(f'Previsão de avaliação {my_predictions[j]:0.2f} para o filme {movieList[j]}')
            i += 1
        l += 1



    st.header("Filmes não recomandados e previsão de avaliação do usuário:")

    # sort predictions
    ix = tf.argsort(my_predictions, direction='ASCENDING')
    k = 0
    i = 0
    l = 0
    while i < 20:
        j = ix[l]
        if j not in vetor_notas and movieList_df["number of ratings"].iloc[int(j)] > 20:
            st.write(f'Previsão de avaliação {my_predictions[j]:0.2f} para o filme {movieList[j]}')
            i += 1
        l += 1


    st.header('Avaliação dada comparada ao do modelo treinado:')
    for i in range(len(vetor_notas)):
        if vetor_notas[i] > 0:
            st.write("Filme: ", movieList[i], " . Nota dada pelo usuário:", vetor_notas[i]," Nota prevista: ", my_predictions[i])


