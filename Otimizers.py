import tensorflow as tf
import numpy as np
import config

'''
Este módulo armazena as funções que geram os otimizadores para a rede neural, os otimizadores disponíveis são o Stochastic Gradient Decendent (SGD), Adaptive Moment Estimation (Adam) e Root Mean Squared Propagation (RMSProp). Para mais informações sobre este otimizadores, consulte o Google.
'''

def create_otimizer_SGD(network):
    '''
    Cria a função de otimização usando o SGD para a rede neural inserida.
    
    inputs:
    network = Um objeto da classe NeuralNetwork
    
    outputs:
    Uma lista com a função de otimização na primeira entrada, a função de atualização dos parâmetros de treino na segunda entrada e uma função que reseta o gradiente antigo.    
    '''
    #Inicializando variáveis usadas no treino
    weight_penalty=tf.Variable(0.0,trainable=False)
    learnrate=tf.Variable(1.0,trainable=False)
    #Grads_old armazena uma espécie de média dos gradientes antigos a serem incorporados no gradiente atual.
    grads_old = [tf.Variable(tf.zeros(param.shape,config.float_type)) for param in network.params]
    #A variável fric representa a velocidade de atualização do grads_old, o valor recomendade é 0.1.
    fric=tf.Variable(np.cast[config.float_type](0),trainable=False)
    
    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.
    @tf.function
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        with tf.GradientTape(persistent=True) as watch_grad:
            #calculando regularização.
            l2_norm_squared = tf.math.reduce_sum([tf.math.reduce_sum(layer.params[0]**2) for layer in network.layers if len(layer.params)>1])
            regularization=0.5*weight_penalty*l2_norm_squared/network.n_subsets
            cost = regularization
            #Adicionando o custo de cada camada ao custo total. Cada camada tem uma flag que informa se o custo da camada deve ser somado ao custo total.
            current_value=network.layers[0].execute(x,True)
            if network.layers[0].cost_flag:
                cost=cost+network.layers[0].cost(current_value,y)
            for i in range(1,len(network.layers)):
                current_value=network.layers[i].execute(current_value,True)
                if network.layers[i].cost_flag:
                    cost=cost+network.layers[i].cost(current_value,y)
        
        #Calculando o gradiente do custo em relação aos parâmetros da rede.
        grads = watch_grad.gradient(cost, network.params)
        updates=[]
        #Atualizando os pesos da rede.
        for param, grad,grad_old in zip(network.params, grads,grads_old):
            updates.append(param.assign(param-learnrate*(grad_old*fric+(1-fric)*grad)))
            updates.append(grad_old.assign(grad_old*fric+(1-fric)*grad))
        return updates
    
    #Criando função que atualiza os parâmetros de treino.
    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.assign(x_1),learnrate.assign(x_2),fric.assign(x_3))
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [i.assign(i*0) for i in grads_old]

    return (update_weigths,update_parameter,reset_hist)

def create_otimizer_ADAM(network):
    '''
    Cria a função de otimização usando o Adam para a rede neural inserida.
    
    inputs:
    network = Um objeto da classe NeuralNetwork
    
    outputs:
    Uma lista com a função de otimização na primeira entrada, a função de atualização dos parâmetros de treino na segunda entrada e uma função que reseta as estimativas do primeiro e segundo momento do gradiente.    
    '''
    #Inicializando variáveis usadas no treino
    weight_penalty=tf.Variable(0.0,trainable=False)
    learnrate=tf.Variable(1.0,trainable=False)
    #Taxa de atualização da estimativa do primeiro e segundo momentos do gradiente. O valor inicializado é o recomendado.
    update_rate_1=tf.Variable(0.9,trainable=False)
    update_rate_2=tf.Variable(0.999,trainable=False)
    #adam_counter conta quantas rodas de treino já ocorreram
    adam_counter = tf.Variable(1.0,trainable=False)
    #adam_moment_1 e adam_moment_2 armazenam as estimativas do primeiro e segundo momento do gradiente a serem usadas durante o treino.
    adam_moment_1 = [tf.Variable(np.zeros(param.shape),dtype=config.float_type,trainable=False) for param in network.params]
    adam_moment_2 = [tf.Variable(np.zeros(param.shape),dtype=config.float_type,trainable=False) for param in network.params]
    
    #Criando função que atualiza os parâmetros de treino.
    update_parameter=lambda x_1,x_2,x_3,x_4: (weight_penalty.assign(x_1),learnrate.assign(x_2),update_rate_1.assign(x_3),update_rate_2.assign(x_4))
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [i.assign(i*0) for i in adam_moment_1]+[i.assign(i*0) for i in adam_moment_2]+[adam_counter.assign(np.int16(1))]
    
    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.
    @tf.function
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        with tf.GradientTape(persistent=True) as watch_grad:
            #calculando a regularização
            l2_norm_squared = tf.math.reduce_sum([tf.math.reduce_sum(layer.params[0]**2) for layer in network.layers if len(layer.params)>1])

            regularization=0.5*weight_penalty*l2_norm_squared/network.n_subsets
            cost = regularization
            #Adicionando o custo de cada camada ao custo total. Cada camada tem uma flag que informa se o custo da camada deve ser somado ao custo total.
            current_value=network.layers[0].execute(x,True)
            if network.layers[0].cost_flag:
                cost=cost+network.layers[0].cost(current_value,y)
            for i in range(1,len(network.layers)):
                current_value=network.layers[i].execute(current_value,True)
                if network.layers[i].cost_flag:
                    cost=cost+network.layers[i].cost(current_value,y)
        
        #Calculando gradiente.
        grads = watch_grad.gradient(cost, network.params)
        
        #Atualizando os pesos da rede.
        updates=[]
        for param, grad,moment_1,moment_2 in zip(network.params, grads,adam_moment_1,adam_moment_2):
            updates.append(
                (param.assign(param-learnrate*(((moment_1*update_rate_1+(1-update_rate_1)*grad)/(1-update_rate_1**adam_counter))/((((moment_2*update_rate_2+(1-update_rate_2)*(grad**2))**0.5)/(1-update_rate_2**adam_counter))+10**(-40)))))
            )
            updates.append(
                moment_1.assign(moment_1*update_rate_1+(1-update_rate_1)*grad)
            )
            updates.append(
                moment_2.assign(moment_2*update_rate_2+(1-update_rate_2)*(grad**2))
            )
            updates.append(
                adam_counter.assign(adam_counter+1)
            )
        return updates
    
    return (update_weigths,update_parameter,reset_hist)


def create_otimizer_RMSProp(network):
    '''
    Cria a função de otimização usando o RMSProp para a rede neural inserida.
    
    inputs:
    network = Um objeto da classe NeuralNetwork
    
    outputs:
    Uma lista com a função de otimização na primeira entrada, a função de atualização dos parâmetros de treino na segunda entrada e uma função que reseta a estimativa do segundo momento do gradiente.    
    '''
    #Inicializando variáveis usadas no treino
    weight_penalty=tf.Variable(0.0,trainable=False)
    learnrate=tf.Variable(1.0,trainable=False)
    #Taxa de atualização da estimativa segundo momentos do gradiente. O valor inicializado é o recomendado.
    update_rate=tf.Variable(0.9,trainable=False)
    #adam_moment_2 armazena a estimativa do segundo momento do gradiente a ser usada durante o treino.
    grad_moment_2 = [tf.Variable(np.zeros(param.shape),dtype=config.float_type,trainable=False) for param in network.params]
    
    #Criando função que atualiza os parâmetros de treino.
    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.assign(x_1),learnrate.assign(x_2),update_rate.assign(x_3))
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [i.assign(i*0) for i in grad_moment_2]
    
    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.
    @tf.function
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        with tf.GradientTape(persistent=True) as watch_grad:
            #calculando a regularização
            l2_norm_squared = tf.math.reduce_sum([tf.math.reduce_sum(layer.params[0]**2) for layer in network.layers if len(layer.params)>1])

            regularization=0.5*weight_penalty*l2_norm_squared/network.n_subsets
            cost = regularization
            #Adicionando o custo de cada camada ao custo total. Cada camada tem uma flag que informa se o custo da camada deve ser somado ao custo total.
            current_value=network.layers[0].execute(x,True)
            if network.layers[0].cost_flag:
                cost=cost+network.layers[0].cost(current_value,y)
            for i in range(1,len(network.layers)):
                current_value=network.layers[i].execute(current_value,True)
                if network.layers[i].cost_flag:
                    cost=cost+network.layers[i].cost(current_value,y)
                    
        #Calculando gradiente.
        grads = watch_grad.gradient(cost, network.params)
        #Atualizando os pesos da rede.
        updates=[]
        for param, grad,moment_2 in zip(network.params, grads,grad_moment_2):
            updates.append(
                (param.assign(param-learnrate*(
                    (grad/(((moment_2*update_rate+(1-update_rate)*(grad**2)+10**(-5))**0.5))
                )
                             )
                )
            ))
            updates.append(
                moment_2.assign(moment_2*update_rate+(1-update_rate)*(grad**2))
            )
        return updates
    
    return (update_weigths,update_parameter,reset_hist)