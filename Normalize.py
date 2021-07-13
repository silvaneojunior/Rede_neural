import config
import tensorflow as tf

def normalize(dados,mean_flag=True,std_flag=True,eigen_flag=False):
    '''
    Faz a decomposição espectral do matrix de covariância do conjunto de dados que foi inserido.
    
    inputs:
    dataset: 2-D numpy array (ou objeto análogo com compatibilidade com o numpy)
    tol=float <= 0
    
    outputs:
    lista: contém uma lista com as os auto-vetores e auto-valores da matriz de covariância dos dados na primeira entrada, uma função que faz a mudança de base para a base dos auto-vetores da matriz de covariância na segunda e a função inversa da mudança de base na terceira.
    
    Comentários:
    tol é a tolerância para auto-valores pequenos, como a matriz de covariância é sempre simetrica definida positiva, então, por definição, todos os seus auto-valores são positivos, assim, para eliminar os auto-valores pequenos basta eliminar os auto-valores que são menores que uma constante C, definimos esta constante como 10**tol, caso tol seja diferente de 0, do contrário todos os auto-valores são aceitos.
    '''
    
    data=tf.convert_to_tensor(dados)
    if mean_flag:
        mean=tf.math.reduce_mean(data,axis=0,keepdims=True)
    else:
        mean=tf.math.reduce_mean(data,axis=0,keepdims=True)*0
    if std_flag:    
        std=tf.math.reduce_std(data,axis=0,keepdims=True)+10**-10
    else:
        std=(tf.math.reduce_std(data,axis=0,keepdims=True)+10**-10)**0
    
    normaliza=lambda x: (x-mean)/std
    desnormaliza=lambda x: x*std+mean
    
    norm_data=normaliza(data)
    
    cov_matrix=tf.matmul(tf.transpose(norm_data),norm_data)/data.shape[0]

    if eigen_flag:
        eigen_values,eigen_vector=tf.linalg.eigh(cov_matrix)

        ortogonal_data=tf.matmul(norm_data,eigen_vector)
        std_ortogonal=tf.math.reduce_std(ortogonal_data,axis=0,keepdims=True)+10**-10
    else:
        eigen_values=tf.ones([data.shape[1]],dtype=data.dtype)
        eigen_vector=tf.eye(data.shape[1],dtype=data.dtype)

        std_ortogonal=tf.ones([1,data.shape[1]],dtype=data.dtype)

    transforma_pca=lambda x: tf.matmul(normaliza(x),eigen_vector)/std_ortogonal
    destransforma_pca=lambda x: desnormaliza(tf.matmul((x*std_ortogonal),tf.transpose(eigen_vector)))
        
    return({'Média':mean,
            'Desvio padrão': std,
            'Auto valores':eigen_values,
            'Auto vetores':eigen_vector,
            'Desvio padrão após mudança de base':std_ortogonal,
            'Normaliza':normaliza,
            'Desnormaliza':desnormaliza,
            'PCA':transforma_pca,
            'DesPCA':destransforma_pca})