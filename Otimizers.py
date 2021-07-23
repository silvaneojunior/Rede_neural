import tensorflow as tf
import config

tf.config.optimizer.set_jit(config.XLA_opt)

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)

zero=config.underflow_constant_value

def create_otimizer_SGD(network):
    weight_penalty=tf.Variable(0.0,dtype=config.float_type,trainable=False)
    learnrate=tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    grads_old = [tf.Variable(tf.zeros(param.shape,config.var_float_type)) for param in network.params]
    fric=tf.Variable(0,dtype=config.var_float_type,trainable=False)
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    flag_nan=tf.Variable(False,trainable=False)
    count=tf.Variable(0,dtype='int32',trainable=False)
    
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    def update_weigths(x,y):
        with tf.GradientTape(persistent=False) as watch_grad:
            regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
            cost=(regularization+network.foward_prop(x,y))*(10**(-network.grad_scale))
        grads = watch_grad.gradient(cost, network.params)
        updates=[]
        updates.append(flag_nan.assign(
                    tf.math.reduce_any([not(tf.math.reduce_all(tf.math.is_finite(grad))) for grad in grads])
            ))
        for param,pre_grad,grad_old in zip(network.params, grads,grads_old):
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
               cost*(10**(network.grad_scale)),
               (
                   (tf.math.reduce_min([tf.math.reduce_min((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
                   (tf.math.reduce_max([tf.math.reduce_max((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale)
               )]

    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.assign(x_1),learnrate.assign(x_2),fric.assign(x_3))
    reset_hist=lambda: [i.assign(i*0) for i in grads_old]
    return (update_weigths,update_parameter,reset_hist)

def create_otimizer_RMSProp(network):
    weight_penalty=tf.Variable(0.0,dtype=config.float_type,trainable=False)
    learnrate=tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    
    update_rate=tf.Variable(0.9,dtype=config.var_float_type,trainable=False)
    grad_moment_2 = [tf.Variable(tf.zeros(param.shape,dtype=config.var_float_type),dtype=config.var_float_type,trainable=False) for param in network.params]
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    flag_nan=tf.Variable(False,trainable=False)
    count=tf.Variable(0,dtype='int32',trainable=False)
    
    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.assign(x_1),learnrate.assign(x_2),update_rate.assign(x_3))
    reset_hist=lambda: [i.assign(i*0) for i in grad_moment_2]
    
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    def update_weigths(x,y):
        with tf.GradientTape(persistent=False) as watch_grad:
            regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
            cost=(regularization+network.foward_prop(x,y))*(10**(-network.grad_scale))

        grads = watch_grad.gradient(cost, network.params)
        updates=[]
        updates.append(flag_nan.assign(
                    tf.math.reduce_any([not(tf.math.reduce_all(tf.math.is_finite(grad))) for grad in grads])
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
               cost*(10**(network.grad_scale)),
               (
                   (tf.math.reduce_min([tf.math.reduce_min((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
                   (tf.math.reduce_max([tf.math.reduce_max((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale)
               )]
    return (update_weigths,update_parameter,reset_hist)

def create_otimizer_Adam(network):
    weight_penalty=tf.Variable(0.0,dtype=config.var_float_type,trainable=False)
    learnrate=tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    
    update_rate_1=tf.Variable(0.9,dtype=config.var_float_type,trainable=False)
    update_rate_2=tf.Variable(0.999,dtype=config.var_float_type,trainable=False)
    adam_counter = tf.Variable(1.0,dtype=config.var_float_type,trainable=False)
    adam_moment_1 = [tf.Variable(tf.zeros(param.shape,dtype=config.var_float_type),dtype=config.var_float_type,trainable=False) for param in network.params]
    adam_moment_2 = [tf.Variable(tf.zeros(param.shape,dtype=config.var_float_type),dtype=config.var_float_type,trainable=False) for param in network.params]
    
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)
    flag_nan=tf.Variable(False,trainable=False)
    count=tf.Variable(0,dtype='int32',trainable=False)
    
    update_parameter=lambda x_1,x_2,x_3,x_4: (weight_penalty.assign(x_1),learnrate.assign(x_2),update_rate_1.assign(x_3),update_rate_2.assign(x_4))
    reset_hist=lambda: [i.assign(i*0) for i in adam_moment_1]+[i.assign(i*0) for i in adam_moment_2]+[adam_counter.assign(1)]

    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    def update_weigths(x,y):
        with tf.GradientTape(persistent=False,watch_accessed_variables=False) as watch_grad:
            watch_grad.watch(network.params)
            regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
            cost=(regularization+network.foward_prop(x,y))*(10**(-network.grad_scale))

        grads = watch_grad.gradient(cost, network.params)
        updates=[]
        updates.append(flag_nan.assign(
                    tf.math.reduce_any([not(tf.math.reduce_all(tf.math.is_finite(grad))) for grad in grads])
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
                    lambda: tf.cond(tf.math.logical_and(tf.equal(count%network.change_freq,0),tf.math.greater(network.grad_scale,0)),lambda: network.grad_scale-1,lambda: network.grad_scale))))
        
        return [updates,
               cost*(10**(network.grad_scale)),
               (
                   (tf.math.reduce_min([tf.math.reduce_min((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale),
                   (tf.math.reduce_max([tf.math.reduce_max((grad*(10**(network.grad_scale)))**2) for grad in grads])**0.5)*10**(network.grad_scale)
               )]
    
    return (update_weigths,update_parameter,reset_hist)

def create_otimizer_GA(network):
    weight_penalty=tf.Variable(0.0,dtype=config.float_type,trainable=False)
    pop_size=tf.Variable(1000,dtype='int32',trainable=False)
    survival_size=tf.Variable(10,dtype='int32',trainable=False)
    mutation_var=tf.Variable(1,dtype=config.float_type,trainable=False)
    params_len=len(network.params)
    network.last_best_breed=[tf.concat([tf.expand_dims(network.params[indice],axis=-1) for i in tf.range(survival_size)],axis=-1) for indice in range(params_len)]
    network.grad_scale=tf.Variable(0,dtype=config.float_type,trainable=False)

    def update_parameter(x_1,x_2,x_3,x_4):
        weight_penalty.assign(x_1)
        pop_size.assign(x_2)
        survival_size.assign(x_3)
        mutation_var.assign(x_4)
        network.last_best_breed=[tf.concat([tf.expand_dims(network.params[indice],axis=-1) for i in tf.range(x_3)],axis=-1) for indice in range(params_len)]
        pass
    reset_hist=lambda: [[i.assign(i*0) for i in network.params] for i in tf.range(survival_size)]
    
    def cost_calc(x,y,params):
        for param, new_param in zip(network.params, params):
            param.assign(new_param)
        regularization=tf.cast(weight_penalty*sum([layer.fixed_cost for layer in network.layers])/network.n_subsets,dtype=config.float_type)
        cost=(regularization+network.foward_prop(x,y))*(10**(-network.grad_scale))
        return cost
    
    @tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    def update_weigths(x,y):
        mixture_values=tf.random.uniform([survival_size,pop_size])
        mixture_values=mixture_values/tf.reduce_sum(mixture_values,axis=0,keepdims=True)
        pop=[tf.random.normal(shape=[1],mean=tf.matmul(param,mixture_values), stddev=mutation_var) for param in network.last_best_breed]

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