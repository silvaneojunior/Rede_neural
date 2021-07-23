import tensorflow as tf

def normalize(dados,mean_flag=True,std_flag=True,eigen_flag=False):
    data=tf.convert_to_tensor(dados)
    if mean_flag:
        mean=tf.math.reduce_mean(data,axis=0,keepdims=True)
    else:
        mean=tf.math.reduce_mean(data,axis=0,keepdims=True)*0
    if std_flag:    
        std=tf.math.reduce_std(data,axis=0,keepdims=True)+10**-10
    else:
        std=(tf.math.reduce_std(data,axis=0,keepdims=True)+10**-10)**0
    
    normalize=lambda x: (x-mean)/std
    unnormalize=lambda x: x*std+mean
    
    norm_data=normalize(data)
    
    cov_matrix=tf.matmul(tf.transpose(norm_data),norm_data)/data.shape[0]

    if eigen_flag:
        eigen_values,eigen_vector=tf.linalg.eigh(cov_matrix)

        orthogonal_data=tf.matmul(norm_data,eigen_vector)
        std_orthogonal=tf.math.reduce_std(orthogonal_data,axis=0,keepdims=True)+10**-10
    else:
        eigen_values=tf.ones([data.shape[1]],dtype=data.dtype)
        eigen_vector=tf.eye(data.shape[1],dtype=data.dtype)

        std_orthogonal=tf.ones([1,data.shape[1]],dtype=data.dtype)

    transform_pca=lambda x: tf.matmul(normalize(x),eigen_vector)/std_orthogonal
    untransform_pca=lambda x: unnormalize(tf.matmul((x*std_orthogonal),tf.transpose(eigen_vector)))
        
    return({'Mean':mean,
            'Standard Deviation': std,
            'Eigen-values':eigen_values,
            'Eigen-vectors':eigen_vector,
            'Eigen-vectors Standard Deviation':std_orthogonal,
            'Normlaize':normalize,
            'Unnormlaize':unnormalize,
            'PCA':transform_pca,
            'UnPCA':untransform_pca})