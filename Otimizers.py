import tensorflow as tf
import config

'''
Este módulo armazena as funções que geram os otimizadores para a rede neural, os otimizadores disponíveis são o Stochastic Gradient Decendent (SGD), Adaptive Moment Estimation (Adam) e Root Mean Squared Propagation (RMSProp). Para mais informações sobre este otimizadores, consulte o Google.
'''
tf.config.optimizer.set_jit(config.XLA_opt)

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)

zero=config.underflow_constant_value

def create_otimizer_SGD(network):
    '''
    Cria a função de otimização usando o SGD para a rede neural inserida.
    
    inputs:
    network = Um objeto da classe NeuralNetwork
    
    outputs:
    Uma lista com a função de otimização na primeira entrada, a função de atualização dos parâmetros de treino na segunda entrada e uma função que reseta o gradiente antigo.    
    '''
    #Inicializando variáveis usadas no treino
    weight_penalty=tf.Variable(0.0,dtype=config.float_type,trainable=False)
    learnrate=tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    #Grads_old armazena uma espécie de média dos gradientes antigos a serem incorporados no gradiente atual.
    grads_old = [tf.Variable(tf.zeros(param.shape,config.var_float_type)) for param in network.params]
    #A variável fric representa a velocidade de atualização do grads_old, o valor recomendade é 0.1.
    fric=tf.Variable(0,dtype=config.var_float_type,trainable=False)
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    flag_nan=tf.Variable(False,trainable=False)
    count=tf.Variable(0,dtype='int32',trainable=False)
    
    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        with tf.GradientTape(persistent=False) as watch_grad:
            regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
            prop_outputs=network.foward_prop(x,y)
            cost=tf.math.reduce_mean(prop_outputs[0])
            cost=(regularization+cost)*(10**(-network.grad_scale))
        #Calculando o gradiente do custo em relação aos parâmetros da rede.
        grads = watch_grad.gradient(cost, network.params)
        updates=[flag_nan.assign(False)]
        #Atualizando os pesos da rede.
        for grad in grads:
            updates.append(flag_nan.assign(
                    tf.math.reduce_any([not(tf.math.reduce_all(tf.math.is_finite(grad))),flag_nan])
            ))
        for param, pre_grad,grad_old in zip(network.params, grads,grads_old):
            grad=pre_grad*tf.cast(10**(network.grad_scale),dtype=config.var_float_type)
            updates.append(grad_old.assign(tf.cond(flag_nan,
                                                  lambda: grad_old,
                                                  lambda: grad_old*fric+(1-fric)*grad)))
            
            updates.append(param.assign(
                tf.cond(flag_nan,
                    lambda: param,
                    lambda: param-learnrate*grad_old
                )))
            
        updates.append(
            count.assign(tf.cond(flag_nan,lambda: 0,lambda: count+1)))
        updates.append(network.grad_scale.assign(
            tf.cond(flag_nan,
                    lambda: network.grad_scale+1,
                    lambda: tf.cond(tf.math.logical_and(tf.equal(count%network.change_freq,0),tf.math.greater(network.grad_scale,0)),lambda: network.grad_scale-1,lambda: network.grad_scale))))
        
        return [updates,
               tf.cond(flag_nan,
                    lambda: 'Flag NaN',
                    lambda: tf.strings.as_string(tf.cast(cost*(10**(network.grad_scale)),'float32'))),
               (tf.math.reduce_max([tf.math.reduce_max((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
               (tf.math.reduce_min([tf.math.reduce_min((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
               cost*10**(network.grad_scale)]+prop_outputs[1:]
    
    #Criando função que atualiza os parâmetros de treino.
    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.assign(x_1),learnrate.assign(x_2),fric.assign(x_3))
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [i.assign(i*0) for i in grads_old]

    return (update_weigths,update_parameter,reset_hist)

def create_otimizer_Adam(network):
    '''
    Cria a função de otimização usando o Adam para a rede neural inserida.
    
    inputs:
    network = Um objeto da classe NeuralNetwork
    
    outputs:
    Uma lista com a função de otimização na primeira entrada, a função de atualização dos parâmetros de treino na segunda entrada e uma função que reseta as estimativas do primeiro e segundo momento do gradiente.    
    '''
    #Inicializando variáveis usadas no treino
    weight_penalty=tf.Variable(0.0,dtype=config.var_float_type,trainable=False)
    learnrate=tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    #Taxa de atualização da estimativa do primeiro e segundo momentos do gradiente. O valor inicializado é o recomendado.
    update_rate_1=tf.Variable(0.9,dtype=config.var_float_type,trainable=False)
    update_rate_2=tf.Variable(0.999,dtype=config.var_float_type,trainable=False)
    #adam_counter conta quantas rodas de treino já ocorreram
    adam_counter = tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    #adam_moment_1 e adam_moment_2 armazenam as estimativas do primeiro e segundo momento do gradiente a serem usadas durante o treino.
    adam_moment_1 = [tf.Variable(tf.zeros(param.shape,dtype=config.var_float_type),dtype=config.var_float_type,trainable=False) for param in network.params]
    adam_moment_2 = [tf.Variable(tf.zeros(param.shape,dtype=config.var_float_type),dtype=config.var_float_type,trainable=False) for param in network.params]
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    flag_nan=tf.Variable(False,trainable=False)
    count=tf.Variable(0,dtype='int32',trainable=False)
    
    #Criando função que atualiza os parâmetros de treino.
    update_parameter=lambda x_1,x_2,x_3,x_4: (weight_penalty.assign(x_1),learnrate.assign(x_2),update_rate_1.assign(x_3),update_rate_2.assign(x_4))
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [i.assign(i*0) for i in adam_moment_1]+[i.assign(i*0) for i in adam_moment_2]+[adam_counter.assign(1)]

    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        with tf.GradientTape(persistent=False,watch_accessed_variables=False) as watch_grad:
            watch_grad.watch(network.params)
            regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
            prop_outputs=network.foward_prop(x,y)
            cost=tf.math.reduce_mean(prop_outputs[0])
            cost=(regularization+cost)*(10**(-network.grad_scale))
        
        #Calculando gradiente.
        grads = watch_grad.gradient(cost, network.params)
        #Atualizando os pesos da rede.
        updates=[flag_nan.assign(False)]
        for grad in grads:
            updates.append(flag_nan.assign(
                    tf.math.reduce_any([not(tf.math.reduce_all(tf.math.is_finite(grad))),flag_nan])
            ))
        for param, pre_grad,moment_1,moment_2 in zip(network.params, grads,adam_moment_1,adam_moment_2):
            grad=pre_grad*tf.cast(10**(network.grad_scale),dtype=config.var_float_type)
            updates.append(
                moment_1.assign(tf.cond(flag_nan,
                    lambda:moment_1,
                    lambda:moment_1*update_rate_1+(1-update_rate_1)*grad))
            )
            updates.append(
                moment_2.assign(tf.cond(flag_nan,
                    lambda:moment_2,
                    lambda:moment_2*update_rate_2+(1-update_rate_2)*grad**2))
            )
            updates.append(
                param.assign(tf.cond(flag_nan,
                    lambda:param,
                    lambda:param-learnrate*(moment_1/(1-update_rate_1**adam_counter))/(((moment_2/(1-update_rate_2**adam_counter))+zero)**0.5)))
            )
            
            updates.append(
                adam_counter.assign(tf.cond(flag_nan,
                    lambda:adam_counter,
                    lambda:adam_counter+1))
            )
        updates.append(
            count.assign(tf.cond(flag_nan,lambda: 0,lambda: count+1)))
        updates.append(network.grad_scale.assign(
            tf.cond(flag_nan,
                    lambda: network.grad_scale+1,
                    lambda: tf.cond(tf.equal(count%network.change_freq,0),lambda: network.grad_scale-1,lambda: network.grad_scale))))
        
        return [updates,
               tf.cond(flag_nan,
                    lambda: 'Flag NaN',
                    lambda: tf.strings.as_string(tf.cast(cost*(10**(network.grad_scale)),'float32'))),
               (tf.math.reduce_max([tf.math.reduce_max((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
               (tf.math.reduce_min([tf.math.reduce_min((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
               cost*10**(network.grad_scale)]+prop_outputs[1:]
    
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
    weight_penalty=tf.Variable(0.0,dtype=config.float_type,trainable=False)
    learnrate=tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    #Taxa de atualização da estimativa segundo momentos do gradiente. O valor inicializado é o recomendado.
    update_rate=tf.Variable(0.9,dtype=config.var_float_type,trainable=False)
    #adam_moment_2 armazena a estimativa do segundo momento do gradiente a ser usada durante o treino.
    grad_moment_2 = [tf.Variable(tf.zeros(param.shape,dtype=config.var_float_type),dtype=config.var_float_type,trainable=False) for param in network.params]
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    flag_nan=tf.Variable(False,trainable=False)
    count=tf.Variable(0,dtype='int32',trainable=False)
    
    #Criando função que atualiza os parâmetros de treino.
    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.assign(x_1),learnrate.assign(x_2),update_rate.assign(x_3))
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [i.assign(i*0) for i in grad_moment_2]
    
    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.    
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        with tf.GradientTape(persistent=False) as watch_grad:
            regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
            prop_outputs=network.foward_prop(x,y)
            cost=tf.math.reduce_mean(prop_outputs[0])
            cost=(regularization+cost)*(10**(-network.grad_scale))
                    
        #Calculando gradiente.
        grads = watch_grad.gradient(cost, network.params)
        #Atualizando os pesos da rede.
        updates=[flag_nan.assign(False)]
        for grad in grads:
            updates.append(flag_nan.assign(
                    tf.math.reduce_any([not(tf.math.reduce_all(tf.math.is_finite(grad))),flag_nan])
            ))            
        for param, pre_grad,moment_2 in zip(network.params, grads,grad_moment_2):
            grad=pre_grad*tf.cast(10**(network.grad_scale),dtype=config.var_float_type)
            updates.append(
                moment_2.assign(tf.cond(flag_nan,
                    lambda:moment_2,
                    lambda:moment_2*update_rate+(1-update_rate)*grad**2))
            )
            
            updates.append(
                param.assign(tf.cond(flag_nan,
                    lambda:param,
                    lambda:param-learnrate*
                    grad/((moment_2+zero)**0.5)
            )))
        updates.append(
            count.assign(tf.cond(flag_nan,lambda: 0,lambda: count+1)))
        updates.append(network.grad_scale.assign(
            tf.cond(flag_nan,
                     lambda: network.grad_scale+1,
                     lambda: tf.cond(tf.math.logical_and(tf.equal(count%network.change_freq,0),tf.math.greater(network.grad_scale,0)),lambda: network.grad_scale-1,lambda: network.grad_scale))))
        return [updates,
               tf.cond(flag_nan,
                    lambda: 'Flag NaN',
                    lambda: tf.strings.as_string(tf.cast(cost*(10**(network.grad_scale)),'float32'))),
               (tf.math.reduce_max([tf.math.reduce_max((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
               (tf.math.reduce_min([tf.math.reduce_min((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
               cost*10**(network.grad_scale)]+prop_outputs[1:]
    
    return (update_weigths,update_parameter,reset_hist)


def create_otimizer_GA(network):
    '''
    Cria a função de otimização usando o RMSProp para a rede neural inserida.
    
    inputs:
    network = Um objeto da classe NeuralNetwork
    
    outputs:
    Uma lista com a função de otimização na primeira entrada, a função de atualização dos parâmetros de treino na segunda entrada e uma função que reseta a estimativa do segundo momento do gradiente.    
    '''
    #Inicializando variáveis usadas no treino
    weight_penalty=tf.Variable(0.0,dtype=config.float_type,trainable=False)
    pop_size=tf.Variable(1000,dtype='int32',trainable=False)
    survival_size=tf.Variable(10,dtype='int32',trainable=False)
    mutation_var=tf.Variable(1,dtype=config.float_type,trainable=False)
    
    params_len=len(network.params)
    
    flag_nan=tf.Variable(False,trainable=False)
    network.last_best_breed=[tf.concat([tf.expand_dims(network.params[indice],axis=-1) for i in tf.range(survival_size)],axis=-1) for indice in range(params_len)]
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    
    params_shape=[tf.TensorSpec(param.shape,dtype=param.dtype) for param in network.params]
    
    #Criando função que atualiza os parâmetros de treino.
    def update_parameter(x_1,x_2,x_3,x_4):
        weight_penalty.assign(x_1)
        pop_size.assign(x_2)
        survival_size.assign(x_3)
        mutation_var.assign(x_4)
        network.last_best_breed=[tf.concat([tf.expand_dims(network.params[indice],axis=-1) for i in tf.range(x_3)],axis=-1) for indice in range(params_len)]
        pass
    #Criando a função que reseta o histórico do gradiente.
    reset_hist=lambda: [[i.assign(i*0) for i in network.params] for i in tf.range(survival_size)]
    
    def cost_calc(x,y,params):
        for param, new_param in zip(network.params, params):
            param.assign(new_param)
        regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
        prop_outputs=network.foward_prop(x,y)
        cost=tf.math.reduce_mean(prop_outputs[0])
        cost=(regularization+cost)
        return cost
    
    #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.    
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    #definindo a função que atualiza os pesos da rede neural.
    def update_weigths(x,y):
        #Esta função atualiza os pesos usando o gradiente.
        #Ao usar a função GradientTape, o tensorflow irá registrar a propagação das variáveis a serem otimizadas (veja o arquivo 'Layers.py' para saber quais são a variáveis a serem otimizadas) no escopo do with, assim o tensorflow poderá calcular o gradiente do custo em relação as variáveis a serem otimizadas de maneira eficiente.
        misture_values=tf.random.uniform([survival_size,pop_size])
        misture_values=misture_values/tf.reduce_sum(misture_values,axis=0,keepdims=True)
        pop=[tf.random.normal(shape=[1],mean=tf.matmul(param,misture_values), stddev=mutation_var) for param in network.last_best_breed]

        costs=tf.zeros([0],dtype=config.float_type)
        #tf.map_fn(lambda param:cost_calc(x,y,param),pop)
        for indiv in tf.range(pop_size):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(costs,tf.TensorShape([None]))],
                                                       parallel_iterations=10000)
            params=[tf.gather(param,indiv,axis=-1) for param in pop]
            costs=tf.concat([costs,[cost_calc(x,y,params)]],axis=0)
        
        survival_cost=tf.math.reduce_min(costs)
        indices=tf.argsort(costs)[:survival_size]
        network.last_best_breed=[tf.gather(param,indices,axis=-1) for param in pop]
                    
        updates=[]
        indice=tf.random.uniform(shape=[],maxval=survival_size,dtype='int32')
        for param, new_param in zip(network.params, network.last_best_breed):
            updates.append(
                param.assign(tf.gather(new_param,indice,axis=-1))
            )
        
        return updates,[tf.strings.as_string(tf.cast(tf.math.reduce_min(survival_cost),'float32'))]+[0]*5
    
    return (update_weigths,update_parameter,reset_hist)