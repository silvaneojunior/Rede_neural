import config
import numpy as np

def normalize(dados,mean_flag=False,std_flag=False):
    '''
    Esta função calcula a média e o desvio padrão de um conjunto de dados.
    
    inputs:
    dataset: 2-D numpy array (ou objeto análogo com compatibilidade com o numpy)
    mean_flag: bool
    std_flag: bool
    
    outputs:
    lista: contém uma lista com as normalizações na primeira entrada, uma função que normaliza dados na segunda e a função inversa da normalização na terceira.
    
    Comentários:
    mean_flag ativa a normalização da média: calcula-se a média das coordenadas do dataset, depois subtrai-se a média dos dados.Caso a flag estaja desativada, o valor retornado é zero.
    std_flag ativa a normalização da variância: calcula-se o desvio padrão das coordenadas do dataset, depois dividi-se os dados pelo desvio padrão.Caso a flag estaja desativada, é retornado um vetor de 1's.
    '''
    normalizacoes=[]
    dataset=dados[:]
    
    if mean_flag:
        mean=np.mean(dataset,axis=1).reshape([len(dataset),1])
        dataset=(dataset-mean)
        normalizacoes.append(mean)
    else:
        normalizacoes.append(0)

    if std_flag:
        std=np.std(dataset,axis=1).reshape([len(dataset),1])
        np.place(std,std==0,1)
        dataset=(dataset/std)
        normalizacoes.append(std)
    else:
        std=np.asarray([[1]]*dataset.shape[0],dtype=config.float_type)
        normalizacoes.append(std)
        
    return(normalizacoes,lambda x: ((x-normalizacoes[0])/normalizacoes[1]),lambda x: ((normalizacoes[1]*x)+normalizacoes[0]))

def eigen_decomp(dados,tol=False):
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
    dataset=dados[:]
    covar=np.cov(dataset)
    eigen=np.linalg.eig(covar)
    eigen_values=eigen[0]
    eigen_vectors=eigen[1].T

    if tol:
        eigen_matrix=eigen_vectors[eigen_values>10**(tolerancia_eigen),:]
    else:
        eigen_matrix=eigen_vectors


    eigen_matrix=eigen_matrix.astype(config.float_type)

    dataset=np.dot(eigen_matrix,dataset)
    return([eigen_matrix,eigen_values],lambda x: np.dot(eigen_matrix,x),lambda x: np.dot(eigen_matrix.T,x))