import numpy as np

def normalize(dados,eigen_flag=False,mean_flag=False,std_flag=False,tolerancia_eigen=False):
    '''
    Esta função normaliza um banco de dados segundo as flag's selecionadas.
    inputs:
    dataset: 2-D numpy array (ou objeto análogo com compatibilidade com o numpy)
    eigen_flag: bool
    mean_flag: bool
    std_flag: bool
    tolerancia_eigen: float
    
    outputs:
    list: contém uma lista com as normalizações na primeira coordenada e uma função que normaliza dados na segunda.
    
    Comentários:
    eigen_flag ativa a normalização por auto_valores: primeiro calcula-se a matriz de covariância das coordenas do dataset, depois calcula-se a decomposição espectral da matriz e faz-se a mudança de coordenada para a base dos auto-vetores da matriz. Caso a flag estaja desativada, a matriz retornada é a identidade. Opcional: Remove os auto-vetores com auto-valores abaixo da tolerancia.
    mean_flag ativa a normalização da média: calcula-se a média das coordenadas do dataset, depois subtrai-se a média dos dados.Caso a flag estaja desativada, o valor retornado é zero.
    std_flag ativa a normalização da variância: calcula-se o desvio padrão das coordenadas do dataset, depois dividi-se os dados pelo desvio padrão.Caso a flag estaja desativada, é retornado um vetor de 1's.
    tolerancia_eigen define o valor apartir do qual os auto-vetores são excluidos: qualquer auto-vetor cujo o auto-valor é menor do que 10**tolerancia_eigen é excluido, se tolerancia_eigen>=0, nenhum valor é excluido.
    '''
    normalizacoes=[]
    dataset=dados[:]
    if eigen_flag:
        covar=np.cov(dataset)
        eigen=np.linalg.eig(covar)
        eigen_values=eigen[0]
        eigen_vectors=eigen[1].T
        
        if tolerancia_eigen is not None:
            eigen_matrix=eigen_vectors[eigen_values>10**(tolerancia_eigen),:]
        else:
            eigen_matrix=eigen_vectors

            
        eigen_matrix=eigen_matrix.astype('float32')

        dataset=np.dot(eigen_matrix,dataset)
        normalizacoes.append(eigen_matrix)
    else:
        normalizacoes.append(np.identity(dataset.shape[0]))

    if mean_flag:
        mean=np.mean(dataset,axis=1).reshape([len(dataset),1])
        dataset=(dataset-mean).astype('float32')
        normalizacoes.append(mean)
    else:
        normalizacoes.append(0)

    if std_flag:
        std=np.std(dataset,axis=1).reshape([len(dataset),1])
        np.place(std,std==0,1)
        dataset=(dataset/std).astype('float32')
        normalizacoes.append(std)
    else:
        std=np.asarray([[1]]*dataset.shape[0],dtype='float32')
        normalizacoes.append(std)
        
    return(normalizacoes,lambda x: ((np.dot(normalizacoes[0],x)-normalizacoes[1])/normalizacoes[2]).astype('float32'))