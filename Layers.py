import numpy as np
import tensorflow as tf
import config

__version__='2.0.1'

'''
Este arquivo armazena o código para a criação das camadas da rede neural, os tipos de camada disponíveis são:

Fully Connected Layer (FC) - Camada básica da rede neural, recebe os inputs, multiplica por uma matriz de pesos, soma um vetor de viés em cada coluna da matriz resultante e aplica a função de ativação em cada coordenada. A otimização desta camada é feita na matriz de pesos e no vetor de viés.

Convolution Layer (Conv) - Camada que usa a operação de convolução, recomendada para dados nos quais a ordem das coordenadas do vetor tem muita importância, por exemplo imagens. Esta camada recebe inputs, aplica a operação de convolução usando uma matriz (filtro), soma-se um vetor de viés e aplica-se a função de ativação em cada coordenada. A otimização desta camada é feita no filtro e no vetor de viés.

Transposed Convolution Layer ou Deconvolution Layer (DeConv) - Camada que usa a operação de convolução transposta, esta camada pode ser entendida como a inversa da camada de Convolução e é recomendada para redes onde tenta-se reconstruir (ou gerar) uma imagem. Esta camada recebe inputs, aplica a operação de convolução transposta usando uma matriz (filtro), soma-se um vetor de viés e aplica-se a função de ativação em cada coordenada. A otimização desta camada é feita no filtro e no vetor de viés.

Batch Normalization Layer (BN) - Camada que faz a normalização dos inputs, é recomendada em Deep Neural Network após uma sequência de camada que representem uma etapa do processo, não há contraindicações para uso em outros contextos, porém, deve-se atentar ao fato de que para a normalização é necessário que os pacotes de treino tenham tamanho estritamente maior que 1, além disso, quanto menor o tamanho dos pacotes, maior a instabilidade desta camada, na minha experiência, é recomendável que o batch_size seja ao menos 100. Esta camada recebe os inputs, calcula a média e o desvio padrão amostral dos inputs, depois subtrai a média dos inputs e divide pelo desvio padrão, após isto, multiplica o valor resultante por um fator de correção e soma ao resultado um outro fator de correção. A otimização desta camada é feita nos fatores de correção.

Intergration Layer - Esta camada integra duas redes neurais, seu uso é restrito a GAN, porém, caso o usuário ache útil, nada impede que seja utilizada em outro contexto. Esta camada recebe inputs, usa estes valores como entrada em uma rede neural e retorna o valor obtido. Este rede neural não passa por otimização.

Intergration Layer - Esta camada une outras camada, tendo como output a junção dos outputs das camadas contidas nela. Este rede neural não passa por otimização, mas carrega os parematros das camadas contidas nela.

Além disso, vale esclarecer o que são alguns termos que serão vistos em todas as camadas:
Dropout - É uma otimização feita a partir da eliminação temporária de algumas coordenadas do valor de ativação da rede. O dropout_rate é um número entre 0 e 1 que representa a probabilidade de uma coordenada da ativação da camada não ser utilizada naquela execução. Ao se usar o dropout evita-se overfitting, pois inibimos a rede a se apoiar excessivamente em algumas poucas coordenadas, ou seja, fica mais difícil decorar os dados visto que eles mudam levemente a cada execução.
'''

class Layer:
    '''
    Esta é a classe base de todas as camadas.
    '''
    def __init__(self,funcao_erro=None,funcao_acuracia=None):
        '''
        Inicializando a camada.
        
        funcao_erro = função ou None
        funcao_acuracia = funçao ou None
        
        As funções de erro e acurácia devem receber dois tensorflow.Tensor (ou objeto compatível), sendo o primeiro o valor computado pela rede, ou seja, o valor esperado da classificação de um dado observado e o segundo valor deve ser a real classificação do dado, ou seja, o valor observado da classificação. Estas funções devem ser todas feita usando funções do tensorflow, recomenda-se que o usuário use as funções já feitas no arquivo 'Functions.py'. Caso esta camada não contribua para a função de custo da rede neural em que será utilizada, o usuário deve informar None em ambas as funções.   
        '''
        if funcao_erro is None:
            #Caso não seja informada a função de erro, esta camada não será usada no cálculo do custo da rede neural.
            self.cost_flag=False
        else:
            #Caso seja informada a função de erro, esta camada será usada no cálculo do custo da rede neural com peso 1, sendo que o peso é custumizável.
            self.cost_flag=True
            self.cost_weight=1
        self.funcao_erro=funcao_erro
        self.funcao_acuracia=funcao_acuracia
        self.params=[]
        
    def cost(self,inputs,outputs):
        return self.funcao_erro(inputs,outputs)
    def accuracy_check(self,inputs,outputs):
        return [self.funcao_erro(inputs,outputs),self.funcao_acuracia(inputs,outputs)]
    
class FC(Layer):
    '''
    Esta é a classe para a criação de camadas do tipo FC.
    FC é a camada básica da rede neural, sejam X o input da rede, W a matriz de pesos, B o vetor de viés e F a função de ativação, definimos:
    
    Y=F(W*X+B)
    
    Sendo que Y é o retorno desta rede.
    Obs.: W*X representa aplicar a transformação linear W no vetor X.
    '''
    def __init__(self,n_inputs,n_neurons,ativ_func,dropout_rate=0.0,funcao_erro=None,funcao_acuracia=None):
        '''
        Inicializando a camada.
        
        n_inputs = int > 0
        n_neurons = int > 0
        ativ_func = função
        dropout_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        n_inputs é a dimensão da entrada desta camada, por exemplo, se esta camada recebe vetores com 10 coordenadas e retorna vetores com 20 coordenadas, então n_inputs=10.
        
        n_outputs é a quantidade de neurônios desta camada, por exemplo, se esta camada recebe vetores com 10 coordenadas e retorna vetores com 20 coordenadas, então n_outputs=20.
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        dropout_rate é a probabilidade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Fully connected layer'
        self.n_inputs,self.n_neurons,self.ativ_func,self.dropout_rate=n_inputs,n_neurons,ativ_func,float(dropout_rate)
        
        self.n_outputs=n_neurons
        
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a n_inputs**-1
        self.w = tf.Variable(
                initial_value=tf.random.normal([n_neurons,self.n_inputs],0, np.sqrt(1.0/(n_inputs)),dtype=config.float_type),
            trainable=True)
        #Inicializando o vetor de viés da camada, o valor inicial do viés é sempre 0.
        self.b = tf.Variable(
                initial_value=tf.zeros((n_neurons,1),dtype=config.float_type),
            trainable=True)
        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.
        
        self.params=[self.w,self.b]

    def execute(self,inpt,train_flag):
        '''
        Este método computa a ativação desta camada.
        
        inputs:
        inpt=tensorflow.Tensor ou objeto compatível
        train_flag = bool
        
        outputs:
        Objeto da mesma classe do inputs.
        
        Comentários:
        O inpt é a entrada da camada e deve vier em um formato compatível com as operações do tensorflow.
        train_flag indica se a execução da camada será usada para treino, se train_flag=True, então usa-se o dropout, de outro modo não se usa dropout.
        '''
        inputs=inpt
        #calculando ativação.
        ativation=tf.matmul(self.w,inputs)+self.b
        if train_flag:
            #criando máscara de dropout.
            mask=tf.reshape(tf.random.categorical(tf.math.log([[self.dropout_rate,1-self.dropout_rate]]),self.n_outputs*inpt.shape[1]),ativation.shape)
            #aplicando dropout.
            ativation=ativation*tf.cast(mask,dtype=config.float_type)
        else:
            #Fazendo a correção dos valores de ativação.
            ativation=ativation*(1-self.dropout_rate)
        
        outputs=self.ativ_func(ativation)
        return outputs
    
class Conv(Layer):
    '''
    Esta é a classe para a criação de camadas do tipo Conv.
    Sejam X o input da rede, W a matriz de pesos (ou filtro), B o vetor de viés e F a função de ativação, definimos:
    
    Y=F(W(*)X+B) , onde (*) é a operação de convolução.
    
    Sendo que Y é o retorno desta rede.
    Não explicarei o que é a operação de convolução, pois seria necessário usar imagens, porém descreverei os aspectos principais da operação. Suponhamos que X seja uma matriz NxM e W uma matriz LxK, então W(*)X é uma matriz com N-L+1 linhas e M-K+1 colunas, desta forma, a operação de convolução diminui a dimensão de X. Alguns parametros adicionais podem ser adicionados a convolução para torna-la mais eficiente (ou ineficiente, caso os parametros sejam escolhidos erroneamente), neste código há implementado apenas um deste parametros chamado de stride, o stride é uma lista com dois valores inteiros que irão dividir a dimensão da matriz Y, para ficar mais claro, voltemos ao exemplo anterior com X uma matriz NxM e W uma matriz LxX, ao aplicar a operação de convolução com stride = (A,B) temos que W(*)X tem (N-L+1)/A linhas e (M-K+1)/B colunas. Obviamente o leitor deve estar se perguntando "Mas e se N-L+1 não for múltiplo de A?", pois bem, neste caso ocorre um processo chamado de padding, que consiste em adicionar colunas ou linhas de zero na matriz de modo que a divisão passe a dar um número inteiro. Vale observar que o número de linhas da matriz final depende apenas do número de linhas do filtro, do número de linhas da do input e da primeira coordenada do stride, e o mesmo vale para o número de colunas. Além disso, lembre-se que X deve ser inserido na camada como um vetor e ele será redimensionado internamente segundo o formato indicado pelo usuário, portanto é bem útil saber calcular qual a dimensão da saída deste tipo de camada. Por último, mas não menos importante, a camada de convolução é especialmente útil para lidar com imagens, porém imagens podem ser compostas de mais de uma matriz (geralmente tem-se uma matriz para cada cor), por isto, esse código foi feito para lidar com "matrizes com 3 dimensões", assim, devemos pensar em X não como uma matriz NxM, mas como um tensor NxMxO, onde O é o número de mapas de X, da mesma forma o filtro costuma ter vários mapas, logo devemos pensar em W como sendo LxKxI, então, tomando uma convolução com stride = (A,B), temos que Y terá dimensão:
    
    ceiling((N-L+1)/A) linhas
    ceiling((M-K+1)/B) colunas
    I mapas
    
    Onde ceiling é a função que recebe um número real retorna o menor inteiro maior do que este número.
    Obs.: Recomendo muito fortemente que o usuário pesquise sobre a operação de convolução para que fique claro o que esta camada faz.
    '''
    def __init__(self,input_shape,filter_shape,stride,ativ_func,dropout_rate=0.0,funcao_erro=None,funcao_acuracia=None):
        '''
        input_shape = lista ou tupla com 3 inteiros
        filter_shape = lista ou tupla com 3 inteiros
        stride = tupla com 2 inteiros
        ativ_func = função
        dropout_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        input_shape são as dimensões da entrada da camada e deve seguir a seguinte ordem (Quantidade de mapas, Quantidade de linhas, Quantidade de colunas).
        
        filter_shape são as dimensões do filtro da camada e deve seguir a seguinte ordem (Quantidade de mapas, Quantidade de linhas, Quantidade de colunas).
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        dropout_rate é a probabilidade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Convolution layer'
        self.input_shape=input_shape
        self.n_inputs=input_shape[0]*input_shape[1]*input_shape[2]
        
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding='VALID'
        
        self.ativ_func=ativ_func
        self.dropout_rate=float(dropout_rate)
        
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a 1 dividido pelo número de elementos em cada mapa do filtro
        self.w = tf.Variable(
                tf.random.normal((self.filter_shape[1:]+self.input_shape[0:1]+self.filter_shape[0:1]),
                                 0,
                                 np.sqrt(1.0/(self.filter_shape[1]*self.filter_shape[2])),
                                 dtype=config.float_type),
                trainable=True)
        #Inicializando o vetor de viés da camada, o valor inicial do viés é sempre 0.
        self.b = tf.Variable(
                tf.zeros([1,1,1,self.filter_shape[0]],dtype=config.float_type),
            trainable=True)
        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.
        self.params=[self.w,self.b]
        
        #calculando o formato da saída.
        self.output_shape=[self.filter_shape[0],
                         int(np.ceil(((self.input_shape[1])-(self.filter_shape[1]-1))/self.stride[0])),
                         int(np.ceil(((self.input_shape[2])-(self.filter_shape[2]-1))/self.stride[1]))]
        self.n_outputs=self.output_shape[0]*self.output_shape[1]*self.output_shape[2]
        
    def execute(self,inpt,train_flag):
        '''
        Este método computa a ativação desta camada.
        
        inputs:
        inpt=tensorflow.Tensor ou objeto compatível
        train_flag = bool
        
        outputs:
        Objeto da mesma classe do inputs.
        
        Comentários:
        O inpt é a entrada da camada e deve vier em um formato compatível com as operações do tensorflow.
        train_flag indica se a execução da camada será usada para treino, se train_flag=True, então usa-se o dropout, de outro modo não se usa dropout.
        '''
        inputs=tf.transpose(tf.reshape(inpt,self.input_shape+[inpt.shape[1]]),[3,1,2,0])
        #calculando a operação de convolução.
        input_filtered=tf.nn.conv2d(input=inputs,
                              filters=self.w,
                              padding=self.padding,
                              strides=self.stride
                              )
        #calculando ativação.
        ativation=tf.transpose(input_filtered+self.b,[3,1,2,0])
        if train_flag:
            #criando máscara de dropout.
            mask=tf.reshape(tf.random.categorical(tf.math.log([[self.dropout_rate,1-self.dropout_rate]]),self.n_outputs*inpt.shape[1]),ativation.shape)
            #aplicando dropout.
            ativation=ativation*tf.cast(mask,dtype=config.float_type)
        else:
            #Fazendo a correção dos valores de ativação.
            ativation=ativation*(1-self.dropout_rate)
        
        outputs=tf.reshape(self.ativ_func(ativation),[self.n_outputs,inpt.shape[1]])
        return outputs
    
    
class DeConv(Layer):
    '''
    Esta é a classe para a criação de camadas do tipo DeConv.
    Sejam X o input da rede, W a matriz de pesos (ou filtro), B o vetor de viés e F a função de ativação, definimos:
    
    Y=F(W(*)X+B) , onde (*) é a operação de convolução transposta.
    
    Sendo que Y é o retorno desta rede.
    Não explicarei o que é a operação de convolução transposta, pois seria necessário usar imagens, porém descreverei os aspectos principais da operação. Suponhamos que X seja uma matriz NxM e W uma matriz LxK, então W(*)X é uma matriz com N+L-1 linhas e M+K-1 colunas, desta forma, a operação de convolução diminui a dimensão de X. Alguns parametros adicionais podem ser adicionados a convolução transposta para torna-la mais eficiente (ou ineficiente, caso os parametros sejam escolhidos erroneamente), neste código há implementado apenas um deste parametros chamado de stride, o stride é uma lista com dois valores inteiros que irão multiplicar a dimensão da matriz Y, para ficar mais claro, voltemos ao exemplo anterior com X uma matriz NxM e W uma matriz LxX, ao aplicar a operação de convolução com stride = (A,B) temos que W(*)X tem A*(N+L-1) linhas e B*(M+K-1) colunas. Vale observar que o número de linhas da matriz final depende apenas do número de linhas do filtro, do número de linhas da do input e da primeira coordenada do stride, e o mesmo vale para o número de colunas. Além disso, lembre-se que X deve ser inserido na camada como um vetor e ele será redimensionado internamente segundo o formato indicado pelo usuário, portanto é bem útil saber calcular qual a dimensão da saída deste tipo de camada. Por último, mas não menos importante, a camada de convolução transposta é especialmente útil para gerar imagens, porém imagens podem ser compostas de mais de uma matriz (geralmente tem-se uma matriz para cada cor), por isto, esse código foi feito para lidar com "matrizes com 3 dimensões", assim, devemos pensar em X não como uma matriz NxM, mas como um tensor NxMxO, onde O é o número de mapas de X, da mesma forma o filtro costuma ter vários mapas, logo devemos pensar em W como sendo LxKxI, então, tomando uma convolução transposta com stride = (A,B), temos que Y terá dimensão:
    
    A*(N+L-1) linhas
    B*(M+K-1) colunas
    I mapas
    
    Obs.: Recomendo muito fortemente que o usuário pesquise sobre a operação de convolução transposta para que fique claro o que esta camada faz.
    '''
    def __init__(self,input_shape,filter_shape,stride,ativ_func,dropout_rate=0.0,funcao_erro=None,funcao_acuracia=None):
        '''
        input_shape = lista ou tupla com 3 inteiros
        filter_shape = lista ou tupla com 3 inteiros
        stride = tupla com 2 inteiros
        ativ_func = função
        dropout_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        input_shape são as dimensões da entrada da camada e deve seguir a seguinte ordem (Quantidade de mapas, Quantidade de linhas, Quantidade de colunas).
        
        filter_shape são as dimensões do filtro da camada e deve seguir a seguinte ordem (Quantidade de mapas, Quantidade de linhas, Quantidade de colunas).
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        dropout_rate é a probabilidade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Deconvolution layer'
        self.input_shape=input_shape
        self.n_inputs=input_shape[0]*input_shape[1]*input_shape[2]
        
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding='VALID'
        
        self.ativ_func=ativ_func
        self.dropout_rate=float(dropout_rate)
        
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a 1 dividido pelo número de elementos em cada mapa do filtro
        self.w = tf.Variable(
                tf.random.normal(self.filter_shape[1:]+self.filter_shape[:1]+input_shape[:1],
                                 0,
                                 np.sqrt(1.0/(self.filter_shape[1]*self.filter_shape[2])),
                                 dtype=config.float_type),
                trainable=True)
        #Inicializando o vetor de viés da camada, o valor inicial do viés é sempre 0.
        self.b = tf.Variable(
                tf.zeros([1,1,1,self.filter_shape[0]],dtype=config.float_type),
            trainable=True)
        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.
        self.params=[self.w,self.b]
        #calculando o formato da saída.
        self.output_shape=[self.filter_shape[0],
                           int(np.ceil(self.stride[0]*self.input_shape[1]+self.filter_shape[1]-1)),
                           int(np.ceil(self.stride[1]*self.input_shape[2]+self.filter_shape[2]-1))]
        self.n_outputs=self.output_shape[0]*self.output_shape[1]*self.output_shape[2]
        
    def execute(self,inpt,train_flag):
        '''
        Este método computa a ativação desta camada.
        
        inputs:
        inpt=tensorflow.Tensor ou objeto compatível
        train_flag = bool
        
        outputs:
        Objeto da mesma classe do inputs.
        
        Comentários:
        O inpt é a entrada da camada e deve vier em um formato compatível com as operações do tensorflow.
        train_flag indica se a execução da camada será usada para treino, se train_flag=True, então usa-se o dropout, de outro modo não se usa dropout.
        '''
        #calculando a operação de convolução.
        inputs=tf.transpose(tf.reshape(inpt,self.input_shape+[inpt.shape[1]]),[3,1,2,0])

        input_filtered=tf.nn.conv2d_transpose(input=inputs,
                              filters=self.w,
                              output_shape=[inpt.shape[1]]+self.output_shape[1:]+self.output_shape[:1],
                              padding=self.padding,
                              strides=self.stride
                              )
        #calculando ativação.
        ativation=tf.transpose(input_filtered+self.b,[3,1,2,0])
        if train_flag:
            #criando máscara de dropout.
            mask=tf.reshape(tf.random.categorical(tf.math.log([[self.dropout_rate,1-self.dropout_rate]]),self.n_outputs*inpt.shape[1]),ativation.shape)
            #aplicando dropout.
            ativation=ativation*tf.cast(mask,dtype=config.float_type)
        else:
            #Fazendo a correção dos valores de ativação.
            ativation=ativation*(1-self.dropout_rate)
        
        outputs=tf.reshape(self.ativ_func(ativation),[self.n_outputs,inpt.shape[1]])
        return outputs
        
class BN(Layer):
    '''
    Esta é a classe para a criação de camadas do tipo BN.
    Seja X a matriz de inputs, onde cada coluna é um dado distinto, e seja M o vetor de médias de X e S o vetor dos desvios padrões amostrais (ambos calculados pelas linhas), então:
    
    Y=(X-M)/S
    
    Sendo C e D o fator de correção da média e do desvio padrão respectivamente, então:
    
    Y*=D*Y+C
    
    Onde Y* é o output da camada.
    Durante o treino, M e S são calculados da forma usual, porém, fora do treino, usa-se M* e S* como estimadores de M e S. Seja A o fator de atualização, então tomamos:
    M_0=0
    S_0=0
    
    E então, durante a i-ésima etapa de treino, definimos:
    
    M_i = A*M + (1-A)*M_(i-1)
    S_i = A*S + (1-A)*S_(i-1)
    
    Assim, após a N etapas de treino, usamos:
    
    M* = M_N
    S* = S_N
    '''
    def __init__(self,n_inputs,update_rate=0.1,funcao_erro=None,funcao_acuracia=None):
        '''
        n_inputs = int > 0
        update_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        n_inputs é a dimensão da entrada desta camada, por exeplo, se esta camada recebe vetores com 10 coordenadas e retorna vetores com 20 coordenadas, então n_inputs=10.
        
        update_rate é a taxa de atualização da média e do desvio padrão a ser usado fora do treino, equivale ao A da descrição da classe.
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        dropout_rate é a probabildade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Batch normalization layer'
        self.n_inputs,self.n_outputs=n_inputs,n_inputs
        self.update_rate=update_rate
        
        #Criando as variáveis treinaveis. self.mean_scale é o fator de correção da média e self.std_scale é o fator de correção do desvio padrão.
        self.mean_scale=tf.Variable(0,dtype=config.float_type,trainable=True)
        self.std_scale=tf.Variable(1,dtype=config.float_type,trainable=True)
        #Criando as variáveis não treinaaveis, elas armazenam média e o desvio padrão que serão usados fora no treino, essa média e este desvio padrão são chamados aqui de globais, pois estimam a média e o desvio padrão da população de onde vem os inputs.
        self.global_mean=tf.Variable([[0]]*n_inputs,dtype=config.float_type,trainable=False)
        self.global_std=tf.Variable([[1]]*n_inputs,dtype=config.float_type,trainable=False)
        
        self.params=[self.mean_scale,self.std_scale]

    def execute(self,inpt,train_flag):
        '''
        Este método computa a ativação desta camada.
        
        inputs:
        inpt=tensorflow.Tensor ou objeto compatível
        train_flag = bool
        
        outputs:
        Objeto da mesma classe do inputs.
        
        Comentários:
        O inpt é a entrada da camada e deve vier em um formato compatível com as operações do tensorflow.
        train_flag indica se a execução da camada será usada para treino, se train_flag=True, então usa-se o dropout, de outro modo não se usa dropout.
        '''
        inputs=inpt
        
        if train_flag:
            current_mean=tf.math.reduce_mean(inputs,axis=1,keepdims=True)
            current_std=tf.math.reduce_std(inputs,axis=1,keepdims=True)*(inpt.shape[1]/(inpt.shape[1]-1))
            mask=tf.cast(current_std!=0,dtype=config.float_type)
            ativation=self.std_scale*((inputs-current_mean)/(current_std+10**(-10)))+self.mean_scale
            self.global_mean.assign((1-self.update_rate)*self.global_mean+self.update_rate*current_mean)
            self.global_std.assign((1-self.update_rate)*self.global_std+self.update_rate*current_std)
        else:
            mask=tf.cast(self.global_std!=0,dtype=config.float_type)
            ativation=self.std_scale*((inputs-self.global_mean)/(self.global_std+10**(-10)))+self.mean_scale
        
        outputs=ativation*mask
        return outputs
    
class integration(Layer):
    '''
    Esta é a classe usada para criar a camada de integração entre de duas redes neurais.
    Seja X o input desta rede e F uma função que representa a execução de uma rede neural, então:
    
    Y=F(X)
    
    Onde Y é o output desta camada.
    '''
    def __init__(self,layers,funcao_erro=None,funcao_acuracia=None):
        '''
        layers = lista de objetos da classe Layer
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        layers é uma lista com as camadas da rede neural que será integrada a rede neural que contém esta camada de integração.  

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Integration layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.layers=layers
    def execute(self,inpt,train_flag):
        '''
        Este método computa a ativação desta camada.
        
        inputs:
        inpt=tensorflow.Tensor ou objeto compatível
        train_flag = bool
        
        outputs:
        Objeto da mesma classe do inputs.
        
        Comentários:
        O inpt é a entrada da camada e deve vier em um formato compatível com as operações do tensorflow.
        train_flag indica se a execução da camada será usada para treino, se train_flag=True, então usa-se o dropout, de outro modo não se usa dropout.
        '''
        inputs=inpt
        for i in self.layers:
            inputs=i.execute(inputs,train_flag)
        return inputs
    
class Mixed(Layer):
    '''
    Esta é a classe usada para criar a camada de mistura entre outras camadas.
    '''
    def __init__(self,layers,funcao_erro=None,funcao_acuracia=None):
        '''
        layers = lista de objetos da classe Layer
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        layers é uma lista com as camadas a serem misturadas, estas camadas devem receber o mesmo número de argumentos.

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Integration layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.n_inputs=layers[0].n_inputs
        self.n_outputs=sum([layer.n_outputs for layer in layers])
        self.layers=layers
        self.params=[param for layer in layers for param in self.params]
    def execute(self,inpt,train_flag):
        '''
        Este método computa a ativação desta camada.
        
        inputs:
        inpt=tensorflow.Tensor ou objeto compatível
        train_flag = bool
        
        outputs:
        Objeto da mesma classe do inputs.
        
        Comentários:
        O inpt é a entrada da camada e deve vier em um formato compatível com as operações do tensorflow.
        train_flag indica se a execução da camada será usada para treino, se train_flag=True, então usa-se o dropout, de outro modo não se usa dropout.
        '''
        inputs=self.layers[0].execute(inpt,train_flag)
        for i in self.layers[1:]:
            inputs=tf.concat([inputs,i.execute(inpt,train_flag)],axis=0)
        return inputs