import tensorflow as tf
import config
from Functions import L2_regularization
import Functions as default

__version__='2.4.1'

tf.config.optimizer.set_jit(config.XLA_opt)
zero=config.underflow_constant_value

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)

class Layer:
    def __init__(self,error_function=None,accur_function=None):
        if error_function is None:
            #Caso não seja informada a função de erro, esta camada não será usada no cálculo do custo da rede neural.
            self.cost_flag=False
        else:
            #Caso seja informada a função de erro, esta camada será usada no cálculo do custo da rede neural com peso 1, sendo que o peso é custumizável.
            self.cost_flag=True
            self.cost_weight=1
        self.recurrent=False
        self.error_function=error_function
        self.accur_function=accur_function
        self.params=[]
        self.fixed_cost=0
        
    def cost(self,inputs,outputs,mask=None):
        return self.error_function(inputs,outputs,mask=None)
    #@tf.function(experimental_compile=config.XLA_func,experimental_relax_shapes=True)
    def accuracy_check(self,inputs,outputs,mask=None):
        return self.error_function(inputs,outputs,mask=None),self.accur_function(inputs,outputs,mask=None)
    def compute_mask(self, inputs, mask=None):
        return mask
    def predict(self,inputs,mask=None):
        return self.execute(inputs,mask)
    
class FC(Layer):
    def __init__(self,n_inputs,n_neurons,activ_function,use_bias=True,bias_offset=0,weight_c=6,error_function=None,accur_function=None,regul_function=L2_regularization):
        Layer.__init__(self,error_function,accur_function)
        self.type='Fully connected layer'
        self.n_inputs,self.n_neurons,self.activ_function=n_inputs,n_neurons,activ_function
        
        self.n_outputs=n_neurons
        
        glorot_limit=(weight_c/(n_inputs+n_neurons))**0.5
        self.w = tf.Variable(
                initial_value=tf.random.uniform([n_inputs,n_neurons],-glorot_limit,glorot_limit,dtype=config.var_float_type),
            trainable=True)
        self.fixed_cost=regul_function(self.w)
        self.params.append(self.w)
        self.b = tf.Variable(
                initial_value=tf.zeros((1,n_neurons),dtype=config.var_float_type)+bias_offset,
            trainable=use_bias)
        if use_bias:
            self.params.append(self.b)

    def execute(self,inpt,mask=None):
        activation=tf.matmul(inpt,tf.cast(self.w,inpt.dtype))+tf.cast(self.b,inpt.dtype)
        outputs=self.activ_function(activation)
        return outputs
    
class Conv(Layer):
    def __init__(self,filter_shape,activ_function,stride=[1,1],padding='VALID',use_bias=True,bias_offset=0,weight_c=6,error_function=None,accur_function=None,regul_function=L2_regularization):
        Layer.__init__(self,error_function,accur_function)
        self.type='Convolution layer'
        
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding=padding
        
        self.activ_function=activ_function
        
        glorot_limit=(weight_c/(self.filter_shape[0]*self.filter_shape[1]*self.filter_shape[2]+self.filter_shape[3]))**0.5
        self.w = tf.Variable(
                tf.random.uniform(self.filter_shape,
                                 -glorot_limit,
                                 glorot_limit,
                                 dtype=config.var_float_type),
                trainable=True)
        self.fixed_cost=regul_function(self.w)
        self.params.append(self.w)
        self.b = tf.Variable(
                tf.zeros([1,1,1,self.filter_shape[3]],dtype=config.var_float_type)+bias_offset,
            trainable=use_bias)
        if use_bias:
            self.params.append(self.b)
    def execute(self,inpt,mask=None):
        inputs=inpt
        input_filtered=tf.nn.conv2d(input=inputs,
                              filters=tf.cast(self.w,inputs.dtype),
                              padding=self.padding,
                              strides=self.stride
                              )
        activation=input_filtered+tf.cast(self.b,inputs.dtype)
        
        outputs=self.activ_function(activation)
        return outputs
    
    
class DeConv(Layer):
    def __init__(self,filter_shape,activ_function,stride=[1,1],padding='VALID',use_bias=True,bias_offset=0,weight_c=6,error_function=None,accur_function=None,regul_function=L2_regularization):
        Layer.__init__(self,error_function,accur_function)
        self.type='Deconvolution layer'
        
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding=padding
        
        self.activ_function=activ_function
        
        glorot_limit=(weight_c/(self.filter_shape[0]*self.filter_shape[1]*self.filter_shape[2]+self.filter_shape[3]))**0.5
        self.w = tf.Variable(
                tf.random.uniform(self.filter_shape,
                                 -glorot_limit,
                                 glorot_limit,
                                 dtype=config.var_float_type),
                trainable=True)
        self.params.append(self.w)
        self.fixed_cost=regul_function(self.w)
        self.b = tf.Variable(
                tf.zeros([1,1,1,self.filter_shape[2]],dtype=config.var_float_type)+bias_offset,
            trainable=use_bias)
        if use_bias:
            self.params.append(self.b)
    def execute(self,inpt,mask=None):
        inputs=inpt
        
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
        activation=input_filtered+tf.cast(self.b,inputs.dtype)
        
        outputs=self.activ_function(activation)
        return outputs
        
class BN(Layer):
    def __init__(self,n_inputs,update_rate=0.1,error_function=None,accur_function=None):
        Layer.__init__(self,error_function,accur_function)
        self.type='Batch normalization layer'
        self.n_inputs,self.n_outputs=n_inputs,n_inputs
        self.update_rate=update_rate
        
        self.mean_scale=tf.Variable(1,dtype=config.var_float_type,trainable=True)
        self.std_scale=tf.Variable(1,dtype=config.var_float_type,trainable=True)
        self.global_mean=tf.Variable(tf.zeros(n_inputs),dtype=config.var_float_type,trainable=False)
        self.global_std=tf.Variable(tf.ones(n_inputs),dtype=config.var_float_type,trainable=False)
        
        self.params=[self.mean_scale,self.std_scale]
    def execute(self,inpt,mask=None):
        inputs=inpt
        
        current_mean,current_std=tf.math.reduce_mean(inputs,axis=0,keepdims=False),tf.math.reduce_std(inputs,axis=0,keepdims=False)
        current_std=current_std
        mask=tf.cast(current_std!=0,dtype=current_mean.dtype)
        activation=(inputs-tf.cast(self.mean_scale,inputs.dtype)*current_mean)/(tf.cast(self.std_scale,inputs.dtype)*current_std+zero)
        self.global_mean.assign((1-self.update_rate)*self.global_mean+self.update_rate*tf.cast(current_mean,config.var_float_type))
        self.global_std.assign((1-self.update_rate)*self.global_std+self.update_rate*tf.cast(current_std,config.var_float_type))
        
        outputs=activation*mask
        return outputs
    def predict(self,inpt,mask=None):
        inputs=inpt
        
        mask=tf.cast(self.global_std!=0,dtype=inpt.dtype)
        activation=(inputs-tf.cast(self.mean_scale,inputs.dtype)*self.global_mean)/(tf.cast(self.std_scale,inputs.dtype)*self.global_std+zero)
        
        outputs=activation*mask
        return outputs
    
class integration(Layer):
    def __init__(self,layers,error_function=None,accur_function=None):
        self.type='Integration layer'
        Layer.__init__(self,error_function,accur_function)
        self.layers=layers
    def execute(self,inpt,mask=None):
        inputs=inpt
        mask=None
        for i in self.layers:
            inputs,mask=i.execute(inputs,mask),i.compute_mask(inputs,mask)
        return inputs
    
    def predict(self,inpt,mask=None):
        inputs=inpt
        mask=None
        for i in self.layers:
            inputs,mask=i.predict(inputs,mask),i.compute_mask(inputs,mask)
        return inputs
    
class Mixed(Layer):
    def __init__(self,layers,index=None,error_function=None,accur_function=None):
        self.type='Mixed layer'
        Layer.__init__(self,error_function,accur_function)
        self.n_inputs=layers[0].n_inputs
        self.n_outputs=sum([layer.n_outputs for layer in layers])
        self.index=index
        self.layers=layers
        self.params=[param for layer in layers for param in layer.params]
        self.fixed_cost=sum([layer.fixed_cost for layer in layers])
    def execute(self,inpt,mask=None):
        if self.index is None:
            outputs=[layer.execute(inpt,mask) for layer in self.layers]
        else:
            outputs=[layer.execute(inpt[:,a:b],mask) for layer,[a,b] in zip(self.layers,self.index)]
        return tf.concat(outputs,axis=1)
    
    def predict(self,inpt,mask=None):
        if self.index is None:
            outputs=[layer.predict(inpt,mask) for layer in self.layers]
        else:
            outputs=[layer.predict(inpt[:,a:b],mask) for layer,[a,b] in zip(self.layers,self.index)]
        return tf.concat(outputs,axis=1)
    
class Batch(Layer):
    def __init__(self,layers,error_function=None,accur_function=None):
        self.type='Batch layer'
        Layer.__init__(self,error_function,accur_function)
        self.layers=layers
        self.n_inputs=layers[0].n_inputs
        self.n_outputs=layers[-1].n_outputs
        self.params=[param for layer in layers for param in layer.params]
        self.fixed_cost=sum([layer.fixed_cost for layer in layers])
    def execute(self,inpt,mask=None):
        inputs=inpt
        mask=None
        for i in self.layers:
            inputs,mask=i.execute(inputs,mask),i.compute_mask(inputs,mask)
            
        return inputs
    
class Bool(Layer):
    def __init__(self,test,layers,error_function=None,accur_function=None):
        self.type='Bool layer'
        Layer.__init__(self,error_function,accur_function)
        self.layers=layers
        self.test=test
        self.params=[param for layer in layers.values() for param in layer.params]
        self.fixed_cost=sum([layer.fixed_cost for layer in layers.values()])
    def execute(self,inpt,mask=None):
        values=self.test(inpt)
        outputs=[]
        indexes=[]
        for i in self.layers.keys():
            indexes.append(tf.squeeze(tf.where(values==i)))
            outputs.append(self.layers[i].execute(tf.gather(inpt,indexes[-1],axis=0),mask))
        index=tf.concat(indexes,axis=1)
        output=tf.concat(outputs,axis=0)
        return tf.gather(output,tf.argsort(index),axis=0)
    def predict(self,inpt,mask=None):
        values=self.test(inpt)
        outputs=[]
        indexes=[]
        for i in self.layers.keys():
            indexes.append(tf.squeeze(tf.where(values==i)))
            outputs.append(self.layers[i].predict(tf.gather(inpt,indexes[-1],axis=0),mask))
        index=tf.concat(indexes,axis=1)
        output=tf.concat(outputs,axis=0)
        return tf.gather(output,tf.argsort(index),axis=0)
    
class Noise(Layer):
    def __init__(self,std,error_function=None,accur_function=None):
        self.type='Bool layer'
        Layer.__init__(self,error_function,accur_function)
        self.std=std
    def execute(self,inpt,mask=None):
        return inpt+tf.random.normal(inpt.shape,0,self.std,dtype=inpt.dtype)
    
class Cast(Layer):
    def __init__(self,dtype,error_function=None,accur_function=None):
        self.type='Cast layer'
        Layer.__init__(self,error_function,accur_function)
        self.dtype=dtype
    def execute(self,inpt,mask=None):
        return tf.cast(inpt,self.dtype)
    
class Reshape(Layer):
    def __init__(self,shape,error_function=None,accur_function=None):
        self.type='Cast layer'
        Layer.__init__(self,error_function,accur_function)
        self.shape=shape
    def execute(self,inpt,mask=None):
        return tf.reshape(inpt,[tf.shape(inpt)[0]]+self.shape)
    
class MaxPooling(Layer):
    def __init__(self,pooling_size,stride=None,error_function=None,accur_function=None):
        Layer.__init__(self,error_function,accur_function)
        self.type='Convolution layer'
        self.pooling_size=pooling_size
        
        if stride is None:
            self.stride=self.pooling_size
        else:
            self.stride=stride
            
        self.padding='VALID'

        self.fixed_cost=0
    def execute(self,inpt,mask=None):
        inputs=inpt
        input_filtered=tf.nn.max_pool2d(input=inputs,
                                        ksize=self.pooling_size,
                                        padding=self.padding,
                                        strides=self.stride
                                        )
        outputs=input_filtered
        return outputs
    
class MeanPooling(Layer):
    def __init__(self,pooling_size,stride=None,error_function=None,accur_function=None):
        Layer.__init__(self,error_function,accur_function)
        self.type='Convolution layer'
        self.pooling_size=pooling_size
        
        if stride is None:
            self.stride=self.pooling_size
        else:
            self.stride=stride
            
        self.padding='VALID'

        self.fixed_cost=0
    def execute(self,inpt,mask=None):
        inputs=inpt
        input_filtered=tf.nn.avg_pool2d(input=inputs,
                                        ksize=self.pooling_size,
                                        padding=self.padding,
                                        strides=self.stride
                                        )
        outputs=input_filtered
        return outputs
    
class LSTM(Layer):
    def __init__(self,n_inputs,n_outputs,activ_function=default.TanH,recur_func=default.Sigmoid,return_sequence=False,use_bias=True,bias_offset=0,weight_c=6,error_function=None,accur_function=None,regul_function=L2_regularization):
        Layer.__init__(self,error_function,accur_function)
        self.return_sequence=return_sequence
        
        self.n_inputs=n_inputs
        self.n_outputs=n_outputs
        self.activ_function=activ_function
        self.recur_func=recur_func
        
        self.params=[]
        self.fixed_cost=0
        
        glorot_limit=(weight_c/(n_inputs+n_outputs))**0.5
        self.w = tf.Variable(
                initial_value=tf.random.uniform([n_inputs,n_outputs*4],-glorot_limit,glorot_limit),
            trainable=True)
        self.fixed_cost+=regul_function(self.w)
        self.params.append(self.w)
        
        self.recur_w = tf.Variable(
                initial_value=tf.random.uniform([n_inputs,n_outputs*4],-glorot_limit,glorot_limit),
            trainable=True)
        self.fixed_cost+=regul_function(self.recur_w)
        self.params.append(self.recur_w)
        
        self.b = tf.Variable(
                initial_value=tf.zeros((1,n_outputs*4))+bias_offset,
            trainable=False) 
        if use_bias:
            self.params.append(self.b)

    def cell(self,h_before,inpts_list):
        inpts,mask=inpts_list
        h_t,s_t=tf.unstack(h_before,axis=0)
        activation=self.b+tf.matmul(inpts,self.w)+tf.matmul(h_t,self.recur_w)
        at_1,at_2,at_3,at_4=tf.split(activation,4,axis=1)

        s_t=tf.nn.sigmoid(at_2)*s_t+tf.nn.sigmoid(at_1)*tf.nn.tanh(at_3)
        h_t=tf.nn.tanh(s_t)*tf.nn.sigmoid(at_4)
    
        return tf.where(mask,tf.stack([h_t,s_t],axis=0),h_before)
    def compute_mask(self, inputs, mask=None):
        return mask if self.return_sequence else None
    def execute(self,inpt,mask=None,h_before=None):
        if h_before is None:
            h_t=tf.pad(inpt[:,0,:]*0,[[0,0],[0,self.n_outputs]])[:,-self.n_outputs:]
            h_t=tf.stack([h_t,h_t],axis=0)
        else:
            h_t=h_before

        output_tensor=tf.scan(self.cell,
                              [tf.transpose(inpt,[1,0,2]),
                               tf.transpose(mask,[1,0,2])],
                              initializer=h_t,
                              swap_memory=config.swap_memory_flag)[:,0]
        output_tensor=tf.transpose(output_tensor,[1,0,2])
        if self.return_sequence:
            output_tensor[:,-1,:]
            
        return output_tensor
    
class Gather(FC):
    def __init__(self,n_inputs,keep_index,error_function=None,accur_function=None,regul_function=L2_regularization):
        Layer.__init__(self,error_function,accur_function)
        
        self.type='Gather layer'
        self.n_inputs,self.n_outputs,self.keep_index=n_inputs,len(keep_index),keep_index
        self.params=[]
    def execute(self,inpt,mask=None):
        return tf.gather(inpt,self.keep_index,axis=1)
    
class TimeDistributed(FC):
    def __init__(self,layer,error_function=None,accur_function=None,regul_function=L2_regularization):
        Layer.__init__(self,error_function,accur_function)
        
        self.type='Time Distributed layer'
        self.layer=layer
        self.n_outputs=layer.n_outputs
        self.params=layer.params
        
        self.exec_step=lambda x: self.layer.execute(x[0],x[1])
        self.pred_step=lambda x: self.layer.predict(x[0],x[1])
    def execute(self,inpt,mask=None):
        activation=tf.map_fn(self.exec_step,(tf.transpose(inpt,[1,0,2]),tf.transpose(mask,[1,0,2])),fn_output_signature=tf.TensorSpec([inpt.shape[0],self.n_outputs],inpt.dtype))
        activation=tf.transpose(activation,[1,0,2])
        return activation
    
    def predict(self,inpt,mask=None):
        activation=tf.map_fn(self.pred_step,(tf.transpose(inpt,[1,0,2]),tf.transpose(mask,[1,0,2])),fn_output_signature=tf.TensorSpec([inpt.shape[0],self.n_outputs],inpt.dtype))
        activation=tf.transpose(activation,[1,0,2])
        return activation

class One_hot(Layer):
    def __init__(self,vec_len,axis,error_function=None,accur_function=None):
        Layer.__init__(self,error_function,accur_function)
        self.vec_len=vec_len
        self.axis=axis
    def execute(self,inpt,mask=None):
        activ=tf.concat(tf.unstack(tf.one_hot(inpt,self.vec_len,dtype=config.float_type),axis=self.axis+1),axis=self.axis)
        return activ
    
class Dropout(Layer):
    def __init__(self,dropout_rate,error_function=None,accur_function=None):
        Layer.__init__(self,error_function,accur_function)
        self.dropout_rate=dropout_rate
    def execute(self,inpt,mask=None):
        return tf.nn.dropout(inpt,self.dropout_rate)
    def predict(self,inpt,mask=None):
        return inpt/(1-self.dropout_rate)
    
class Mask(Layer):
    def __init__(self,mask_value=0,error_function=None,accur_function=None):
        Layer.__init__(self,error_function,accur_function)
        self.mask_value=0
    def compute_mask(self, inputs, mask=None):
        return tf.math.logical_not(tf.math.reduce_all(tf.math.equal(inputs,self.mask_value),axis=-1,keepdims=True))
    def execute(self,inpt,mask=None):
        return inpt