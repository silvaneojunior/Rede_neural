import tensorflow as tf
import config
from Functions import L2_regularization
import Functions as default

__version__='2.3.0'

'''
Este arquivo armazena o código para a criação das camadas da rede neural, os tipos de camada disponíveis são:

Fully Connected Layer (FC) - Camada básica da rede neural, recebe os inputs, multiplica por uma matriz de pesos, soma um vetor de viés em cada coluna da matriz resultante e aplica a função de ativação em cada coordenada. A otimização desta camada é feita na matriz de pesos e no vetor de viés.

Convolution Layer (Conv) - Camada que usa a operação de convolução, recomendada para dados nos quais a ordem das coordenadas do vetor tem muita importância, por exemplo imagens. Esta camada recebe inputs, aplica a operação de convolução usando uma matriz (filtro), soma-se um vetor de viés e aplica-se a função de ativação em cada coordenada. A otimização desta camada é feita no filtro e no vetor de viés.

Transposed Convolution Layer ou Deconvolution Layer (DeConv) - Camada que usa a operação de convolução transposta, esta camada pode ser entendida como a inversa da camada de Convolução e é recomendada para redes onde tenta-se reconstruir (ou gerar) uma imagem. Esta camada recebe inputs, aplica a operação de convolução transposta usando uma matriz (filtro), soma-se um vetor de viés e aplica-se a função de ativação em cada coordenada. A otimização desta camada é feita no filtro e no vetor de viés.

Batch Normalization Layer (BN) - Camada que faz a normalização dos inputs, é recomendada em Deep Neural Network após uma sequência de camada que representem uma etapa do processo, não há contraindicações para uso em outros contextos, porém, deve-se atentar ao fato de que para a normalização é necessário que os pacotes de treino tenham tamanho estritamente maior que 1, além disso, quanto menor o tamanho dos pacotes, maior a instabilidade desta camada, na minha experiência, é recomendável que o batch_size seja ao menos 100. Esta camada recebe os inputs, calcula a média e o desvio padrão amostral dos inputs, depois subtrai a média dos inputs e divide pelo desvio padrão, após isto, multiplica o valor resultante por um fator de correção e soma ao resultado um outro fator de correção. A otimização desta camada é feita nos fatores de correção.

Intergration Layer - Esta camada integra duas redes neurais, seu uso é restrito a GAN, porém, caso o usuário ache útil, nada impede que seja utilizada em outro contexto. Esta camada recebe inputs, usa estes valores como entrada em uma rede neural e retorna o valor obtido. Esta camada não passa por otimização.

Mixed Layer - Esta camada une outras camada, tendo como output a junção dos outputs das camadas contidas nela. Esta camada não passa por otimização, mas carrega os parematros das camadas contidas nela.

Batch Layer - Esta camada agrupa outras camada, criando assim um bloco de camadas que são executadas como uma única camada. Esta camada não passa por otimização, mas carrega os parematros das camadas contidas nela.

Bool Layer - Esta camada divide os dados segundo um critério pre-inserido (a divisão é feita nas colunas da matriz, ou seja, separa-se as amostras em grupos) e aplica uma camada em cada um destes grupo, após isto, ela agrupa os resultados das camadas e retorna-os. Esta camada não passa por otimização, mas carrega os parematros das camadas contidas nela.

Além disso, vale esclarecer o que são alguns termos que serão vistos em todas as camadas:
Dropout - É uma otimização feita a partir da eliminação temporária de algumas coordenadas do valor de ativação da rede. O dropout_rate é um número entre 0 e 1 que representa a probabilidade de uma coordenada da ativação da camada não ser utilizada naquela execução. Ao se usar o dropout evita-se overfitting, pois inibimos a rede a se apoiar excessivamente em algumas poucas coordenadas, ou seja, fica mais difícil decorar os dados visto que eles mudam levemente a cada execução.
'''
tf.config.optimizer.set_jit(config.XLA_opt)
zero=config.underflow_constant_value

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)

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
        self.recurrent=False
        self.funcao_erro=funcao_erro
        self.funcao_acuracia=funcao_acuracia
        self.params=[]
        self.fixed_cost=0
        
    def cost(self,inputs,outputs):
        return self.funcao_erro(inputs,outputs)
    #@tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    def accuracy_check(self,inputs,outputs):
        return self.funcao_erro(inputs,outputs),self.funcao_acuracia(inputs,outputs)
    
class FC(Layer):
    '''
    Esta é a classe para a criação de camadas do tipo FC.
    FC é a camada básica da rede neural, sejam X o input da rede, W a matriz de pesos, B o vetor de viés e F a função de ativação, definimos:
    
    Y=F(W*X+B)
    
    Sendo que Y é o retorno desta rede.
    Obs.: W*X representa aplicar a transformação linear W no vetor X.
    '''
    def __init__(self,n_inputs,n_neurons,ativ_func,bias_offset=0,normalization=-1,dropout_rate=0.0,weight_var=None,funcao_erro=None,funcao_acuracia=None,funcao_regularizacao=L2_regularization):
        '''
        Inicializando a camada.
        
        n_inputs = int > 0
        n_neurons = int > 0
        ativ_func = função
        bias_offset = float
        normalization = float<=1
        dropout_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        n_inputs é a dimensão da entrada desta camada, por exemplo, se esta camada recebe vetores com 10 coordenadas e retorna vetores com 20 coordenadas, então n_inputs=10.
        
        n_outputs é a quantidade de neurônios desta camada, por exemplo, se esta camada recebe vetores com 10 coordenadas e retorna vetores com 20 coordenadas, então n_outputs=20.
        
        bias_offset é o valor inicial do bias.
        
        normalization é a velocidade de atualização dos parametros de normalização, se menor que zero, então não haverá normalização.
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        dropout_rate é a probabilidade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Fully connected layer'
        self.n_inputs,self.n_neurons,self.ativ_func,self.dropout_rate=n_inputs,n_neurons,ativ_func,float(dropout_rate)
        
        self.n_outputs=n_neurons
        
        if weight_var is None:
            weight_var=1.0/(n_inputs)
        
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a n_inputs**-1
        self.w = tf.Variable(
                initial_value=tf.random.normal([self.n_inputs,n_neurons],0, weight_var**0.5,dtype=config.var_float_type),
            trainable=True)
        self.fixed_cost=funcao_regularizacao(self.w)
        self.params.append(self.w)
        if normalization<0:
            self.normalization=False
            #Inicializando o vetor de viés da camada.
            self.b = tf.Variable(
                    initial_value=tf.zeros((1,n_neurons),dtype=config.var_float_type)+bias_offset,
                trainable=True)
            self.params.append(self.b)
        else:
            #Criando normalização
            self.normalization=True
            self.norm_layer=BN(n_inputs=n_neurons,update_rate=normalization)
            self.params.append(self.norm_layer.params[0])
            self.params.append(self.norm_layer.params[1])
        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.

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
        if train_flag:
            #aplicando dropout.
            inputs=tf.nn.dropout(inpt,self.dropout_rate)
        else:
            #Fazendo a correção dos valores do input.
            inputs=inpt/(1-self.dropout_rate)
        #calculando ativação.
        ativation=tf.matmul(inputs,tf.cast(self.w,inputs.dtype))
        if self.normalization:
            ativation=self.norm_layer.execute(ativation,train_flag)
        else:
            ativation+=tf.cast(self.b,inputs.dtype)
        
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
    def __init__(self,filter_shape,stride,ativ_func,padding='VALID',bias_offset=0,weight_var=None,dropout_rate=0.0,funcao_erro=None,funcao_acuracia=None,funcao_regularizacao=L2_regularization):
        '''
        filter_shape = lista ou tupla com 3 inteiros
        stride = tupla com 2 inteiros
        ativ_func = função
        bias_offset = float
        normalization = float<=1
        dropout_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        filter_shape são as dimensões do filtro da camada e deve seguir a seguinte ordem (Quantidade de mapas, Quantidade de linhas, Quantidade de colunas).
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        bias_offset é o valor inicial do bias.
        
        normalization é a velocidade de atualização dos parametros de normalização, se menor que zero, então não haverá normalização.
        
        dropout_rate é a probabilidade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Convolution layer'
        
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding=padding
        
        self.ativ_func=ativ_func
        self.dropout_rate=float(dropout_rate)
        
        if weight_var is None:
            weight_var=1.0/(self.filter_shape[0]*self.filter_shape[1])
        
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a 1 dividido pelo número de elementos em cada mapa do filtro
        self.w = tf.Variable(
                tf.random.normal(self.filter_shape,
                                 0,
                                 (weight_var)**0.5,
                                 dtype=config.var_float_type),
                trainable=True)
        self.fixed_cost=funcao_regularizacao(self.w)
        self.params.append(self.w)
        #Inicializando o vetor de viés da camada.
        self.b = tf.Variable(
                tf.zeros([1,1,1,self.filter_shape[3]],dtype=config.var_float_type)+bias_offset,
            trainable=True)
        self.params.append(self.b)
        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.
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
        inputs=inpt#tf.transpose(inpt,[1,2,3,0])
        if train_flag:
            #aplicando dropout.
            inputs=tf.nn.dropout(inputs,self.dropout_rate)
        else:
            #Fazendo a correção dos valores do input.
            inputs=inputs/(1-self.dropout_rate)
        #calculando a operação de convolução.
        input_filtered=tf.nn.conv2d(input=inputs,
                              filters=tf.cast(self.w,inputs.dtype),
                              padding=self.padding,
                              strides=self.stride
                              )
        #calculando ativação.
        ativation=input_filtered+tf.cast(self.b,inputs.dtype)
        #ativation=tf.transpose(ativation,[3,0,1,2])
        
        outputs=self.ativ_func(ativation)
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
    def __init__(self,filter_shape,stride,ativ_func,padding='VALID',bias_offset=0,dropout_rate=0.0,weight_var=None,funcao_erro=None,funcao_acuracia=None,funcao_regularizacao=L2_regularization):
        '''
        filter_shape = lista ou tupla com 3 inteiros
        stride = tupla com 2 inteiros
        ativ_func = função
        dropout_rate = float, entre 0 e 1
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        filter_shape são as dimensões do filtro da camada e deve seguir a seguinte ordem (Quantidade de mapas, Quantidade de linhas, Quantidade de colunas).
        
        ativ_func é a função de ativação da camada, deve receber apenas um tensorflow.Tensor e retornar um tensroflow.Tensor, além disso, deve ser feita apenas com objetos e funções do tensorflow. Recomenda-se fortemente que o usuário use funções do arquivo 'Functions.py'.
        
        dropout_rate é a probabilidade de uma coordenada da ativação da camada ser multiplicada por zero durante a execução
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Deconvolution layer'
        
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding=padding
        
        self.ativ_func=ativ_func
        self.dropout_rate=float(dropout_rate)
        if weight_var is None:
            weight_var=1.0/(self.filter_shape[0]*self.filter_shape[1])
            
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a 1 dividido pelo número de elementos em cada mapa do filtro
        self.w = tf.Variable(
                tf.random.normal(self.filter_shape,
                                 0,
                                 (weight_var)**0.5,
                                 dtype=config.var_float_type),
                trainable=True)
        self.params.append(self.w)
        self.fixed_cost=funcao_regularizacao(self.w)
        #Inicializando o vetor de viés da camada.
        self.b = tf.Variable(
                tf.zeros([1,1,1,self.filter_shape[2]],dtype=config.var_float_type)+bias_offset,
            trainable=True)
        self.params.append(self.b)

        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.
        
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
        inputs=inpt#tf.transpose(inpt,[1,2,3,0])
        
        if train_flag:
            #aplicando dropout.
            inputs=tf.nn.dropout(inputs,self.dropout_rate)
        else:
            #Fazendo a correção dos valores do input.
            inputs=inputs/(1-self.dropout_rate)
        
        if self.padding=='VALID':
            output_shape=[self.stride[0]*inputs.shape[1]+self.filter_shape[0]-1,
                          self.stride[1]*inputs.shape[2]+self.filter_shape[1]-1,
                          self.filter_shape[2]]
        elif self.padding=='SAME':
            output_shape=[inputs.shape[1],
                          inputs.shape[2],
                          self.filter_shape[2]]
        else:
            output_shape=[self.stride[0]*inputs.shape[1]+self.filter_shape[0]-1-sum(self.padding[1]),
                          self.stride[1]*inputs.shape[2]+self.filter_shape[1]-1-sum(self.padding[2]),
                          self.filter_shape[2]]
        
        input_filtered=tf.nn.conv2d_transpose(input=inputs,
                                              filters=tf.cast(self.w,inputs.dtype),
                                              output_shape=[inputs.shape[0]]+output_shape,
                                              padding=self.padding,
                                              strides=self.stride
                                             )
        #calculando ativação.
        ativation=input_filtered+tf.cast(self.b,inputs.dtype)
        #ativation=tf.transpose(ativation,[3,0,1,2])
        
        outputs=self.ativ_func(ativation)
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
        
        #Criando as variáveis treinaveis. self.mean_scale é o fator de correção da média e self.var_scale é o fator de correção da variância.
        self.mean_scale=tf.Variable(0,dtype=config.var_float_type,trainable=True)
        self.var_scale=tf.Variable(1,dtype=config.var_float_type,trainable=True)
        #Criando as variáveis não treinaaveis, elas armazenam média e o desvio padrão que serão usados fora no treino, essa média e esta variância são chamados aqui de globais, pois estimam a média e o desvio padrão da população de onde vem os inputs.
        self.global_mean=tf.Variable([[0]*n_inputs],dtype=config.var_float_type,trainable=False)
        self.global_var=tf.Variable([[1]*n_inputs],dtype=config.var_float_type,trainable=False)
        
        self.params=[self.mean_scale,self.var_scale]

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
            current_mean,current_var=tf.nn.moments(inputs,axes=0,keepdims=True)
            current_var=current_var
            mask=tf.cast(current_var!=0,dtype=current_mean.dtype)
            ativation=tf.nn.batch_normalization(inputs, current_mean, current_var, tf.cast(self.mean_scale,inputs.dtype), tf.cast(self.var_scale,inputs.dtype), zero)
            self.global_mean.assign((1-self.update_rate)*self.global_mean+self.update_rate*tf.cast(current_mean,config.var_float_type))
            self.global_var.assign((1-self.update_rate)*self.global_var+self.update_rate*tf.cast(current_var,config.var_float_type))
        else:
            mask=tf.cast(self.global_var!=0,dtype=inpt.dtype)
            ativation=tf.nn.batch_normalization(inputs, tf.cast(self.global_mean,inputs.dtype), tf.cast(self.global_var,inputs.dtype), tf.cast(self.mean_scale,inputs.dtype), tf.cast(self.var_scale,inputs.dtype), zero)
        
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
    def __init__(self,layers,index=None,funcao_erro=None,funcao_acuracia=None):
        '''
        layers = lista de objetos da classe Layer
        index = lista de lista de inteiros com os indices da separação ou None.
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        layers é uma lista com as camadas a serem misturadas, estas camadas devem receber o mesmo número de argumentos.
        
        index é um parametro opcional que informa quais os indices que devem ser usados em cada camada, cada elemento deve ser uma lista com o indice do primeiro valor e o indice do ultimo valor de cada grupo, sendo que cada grupo será usado em uma camada, segundo a ordem dos grupos e camadas. Caso index seja None, então cada camada receberá o input completo
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Mixed layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.n_inputs=layers[0].n_inputs
        self.n_outputs=sum([layer.n_outputs for layer in layers])
        self.index=index
        self.layers=layers
        self.params=[param for layer in layers for param in layer.params]
        self.fixed_cost=sum([layer.fixed_cost for layer in layers])
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
        if self.index is None:
            outputs=[layer.execute(inpt,train_flag) for layer in self.layers]
        else:
            outputs=[layer.execute(inpt[:,a:b],train_flag) for layer,[a,b] in zip(self.layers,self.index)]
        return tf.concat(outputs,axis=1)
    
class Batch(Layer):
    '''
    Esta é a classe usada para criar um bloco de camadas.
    '''
    def __init__(self,layers,funcao_erro=None,funcao_acuracia=None):
        '''
        layers = lista de objetos da classe Layer
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        layers é a lista de camadas a serem agrupados no bloco.

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        
        self.type='Batch layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.layers=layers
        self.n_inputs=layers[0].n_inputs
        self.n_outputs=layers[-1].n_outputs
        self.params=[param for layer in layers for param in layer.params]
        self.fixed_cost=sum([layer.fixed_cost for layer in layers])
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
    
class Bool(Layer):
    '''
    Esta é a classe usada para criar uma camada que opera como um "if".
    '''
    def __init__(self,test,layers,funcao_erro=None,funcao_acuracia=None):
        '''
        test = função
        layers = dict
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        test é um função que retorna um valor que esteja em layers.keys().
        
        layers é um dicionário com objetos da chaves e objetos da classe layers associados a ela.

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Bool layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.layers=layers
        self.test=test
        self.params=[param for layer in layers.values() for param in layer.params]
        self.fixed_cost=sum([layer.fixed_cost for layer in layers.values()])
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
        values=self.test(inpt)
        outputs=[]
        indexes=[]
        for i in self.layers.keys():
            indexes.append(tf.squeeze(tf.where(values==i)))
            outputs.append(self.layers[i].execute(tf.gather(inpt,indexes[-1],axis=0),train_flag))
        index=tf.concat(indexes,axis=1)
        output=tf.concat(outputs,axis=0)
        return tf.gather(output,tf.argsort(index),axis=0)
    
class Noise(Layer):
    '''
    Esta é a classe usada para criar uma camada que adiciona ruído a rede.
    '''
    def __init__(self,std,funcao_erro=None,funcao_acuracia=None):
        '''
        n_inputs = int
        std = float > 0
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        std é o desvio padrão do ruído adicionado.

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Bool layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.std=std
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
        return inpt+tf.random.normal(inpt.shape,0,self.std,dtype=inpt.dtype)
    
class Cast(Layer):
    '''
    Esta é a classe usada para converter o tipo da variável na rede.
    '''
    def __init__(self,dtype,funcao_erro=None,funcao_acuracia=None):
        '''
        dtype = string
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        dtype deve ser uma string informando em qual tipo o input deve ser transformado.

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Cast layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.dtype=dtype
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
        return tf.cast(inpt,self.dtype)
    
class Reshape(Layer):
    '''
    Esta é a classe usada para mudar o formato dos dados.
    '''
    def __init__(self,shape,funcao_erro=None,funcao_acuracia=None):
        '''
        shape = tupla de inteiros
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        shape é o formato para o qual os dados devem ser transformados.

        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        self.type='Cast layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.shape=shape
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
        return tf.reshape(inpt,[tf.shape(inpt)[0]]+self.shape)
    
class Pooling(Layer):
    '''
    Esta é a classe para a criação de camadas do tipo Pooling.
    Essencialmente, o que esta camada faz o mesmo da camada de convolução, porém, ao invés de se somar os valores entregues ao filtro com os pesos do filtro, tira-se o máximo dos valores iniciais.
    '''
    def __init__(self,pooling_size,stride=None,funcao_erro=None,funcao_acuracia=None):
        '''
        pooling_size = tupla com 2 inteiros
        stride = tupla com 2 inteiros
        funcao_erro = função ou None
        funcao_acuracia = função ou None
        
        pooling_size é uma lista/tupla com a dimensão do agrupamento.
        
        normalization é a velocidade de atualização dos parametros de normalização, se menor que zero, então não haverá normalização.
        
        As funções de erro e acurácia seguem a mesma descrição que está na classe Layer.
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Convolution layer'
        self.pooling_size=pooling_size
        
        if stride is None:
            self.stride=self.pooling_size
        else:
            self.stride=stride
            
        self.padding='VALID'
        
        #Inicializando os pesos da camada aleatoriamente, seguindo uma distribuição normal com média zero e desvio padrão igual a 1 dividido pelo número de elementos em cada mapa do filtro
        
        self.fixed_cost=0
        #Observe que as variáveis receberam o argumento trainable=True, isto sinaliza para o tensorflow que estas variáveis devem ser "vigiadas" para o cálculo de suas derivadas.
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
        inputs=inpt#tf.transpose(inpt,[3,1,2,0])
        #calculando a operação de convolução.
        input_filtered=tf.nn.max_pool2d(input=inputs,
                                        ksize=self.pooling_size,
                                        padding=self.padding,
                                        strides=self.stride
                                        )
        outputs=input_filtered#tf.transpose(input_filtered,[3,1,2,0])
        return outputs
    
class LSTM(Layer):
    '''
    A escrever
    '''
    def __init__(self,n_inputs,n_outputs,ativ_func=default.TanH,recur_func=default.Sigmoid,return_sequence=False,bias_offset=0,dropout_rate=0.0,funcao_erro=None,funcao_acuracia=None,funcao_regularizacao=L2_regularization):
        '''
        A preencher
        
        '''
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Long short-therm memory layer'
        self.dropout_rate=float(dropout_rate)
        self.return_sequence=return_sequence
        
        self.n_inputs=n_inputs
        self.n_outputs=n_outputs
        self.ativ_func=ativ_func
        self.recur_func=recur_func
        
        self.params=[]
        self.fixed_cost=0
        
        self.w = tf.Variable(
                initial_value=tf.random.normal([self.n_inputs,n_outputs*4],0, (n_outputs*4)**-0.5),
            trainable=True)
        self.fixed_cost+=funcao_regularizacao(self.w)
        self.params.append(self.w)
        
        self.recur_w = tf.Variable(
                initial_value=tf.random.normal([n_outputs,n_outputs*4],0, (n_outputs*4)**-0.5),
            trainable=True)
        self.fixed_cost+=funcao_regularizacao(self.recur_w)
        self.params.append(self.recur_w)
        
        self.b = tf.Variable(
                initial_value=tf.zeros((1,n_outputs*4))+bias_offset,
            trainable=True)        
        self.params.append(self.b)
        self.recurrent=True

    def cell(self,h_before,inpts):
        h_t,s_t=tf.unstack(h_before,axis=0)
        ativacao=tf.matmul(inpts,self.w)+tf.matmul(h_t,self.recur_w)+self.b
        at_1,at_2,at_3,at_4=tf.split(ativacao,4,axis=1)

        s_t=self.recur_func(at_1)*s_t+self.recur_func(at_2)*self.ativ_func(at_3)
        h_t=self.ativ_func(s_t)*self.recur_func(at_4)
        return tf.stack([h_t,s_t],axis=0)

    def execute(self,inpt,train_flag=False,h_before=None,s_before=None):
        if h_before is None or s_before is None:
            h_t=tf.pad(inpt[:,0,:]*0,[[0,0],[0,self.n_outputs]])[:,-self.n_outputs:]
            h_t=tf.stack([h_t,h_t],axis=0)
        else:
            h_t=h_before
        if self.return_sequence:
            output_tensor=tf.scan(self.cell,tf.transpose(inpt,[1,0,2]),initializer=h_t,swap_memory=config.swap_memory_flag)[:,1]
            output_tensor=tf.transpose(output_tensor,[1,0,2])
        else:
            for i in range(tf.shape(inpt)[1]):
                tf.autograph.experimental.set_loop_options(swap_memory=config.swap_memory_flag)
                h_t=self.cell(h_t,inpt[:,i,:])
            output_tensor=h_t[0,:,:self.n_outputs]
        return output
    
class Gather(FC):
    def __init__(self,n_inputs,keep_index,funcao_erro=None,funcao_acuracia=None,funcao_regularizacao=L2_regularization):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        
        self.type='Gather layer'
        self.n_inputs,self.n_outputs,self.keep_index=n_inputs,len(keep_index),keep_index
        self.params=[]
    def execute(self,inpt,train_flag):
        '''
        A escrever
        '''
        return tf.gather(inpt,self.keep_index,axis=1)
    
class TimeDistributed(FC):
    def __init__(self,layer,funcao_erro=None,funcao_acuracia=None,funcao_regularizacao=L2_regularization):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        
        self.type='Time Distributed layer'
        self.layer=layer
        self.params=layer.params
        
        self.train_loop=lambda x: self.layer.execute(x,True)
        self.exec_loop=lambda x: self.layer.execute(x,False)
    def execute(self,inpt,train_flag):
        '''
        A escrever
        '''
        if train_flag:
            activation=tf.map_fn(self.exec_loop,tf.transpose(inpt,[1,0,2]))
        else:
            activation=tf.map_fn(self.train_loop,tf.transpose(inpt,[1,0,2]))
        activation=tf.transpose(activation,[1,0,2])
        return activation

class One_hot(Layer):
    def __init__(self,vec_len,axis,funcao_erro=None,funcao_acuracia=None):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.vec_len=vec_len
        self.axis=axis
    def execute(self,inpt,train_flag):
        activ=tf.concat(tf.unstack(tf.one_hot(inpt,self.vec_len,dtype=config.float_type),axis=self.axis+1),axis=self.axis)
        return activ