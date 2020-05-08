import numpy as np
import theano
from theano import sparse
import theano.tensor as tensor
from theano.tensor.nnet import conv2d,conv2d_transpose
from theano.tensor.signal import pool
from theano.tensor import shared_randomstreams

class Layer:
    def __init__(self,funcao_erro=None,funcao_acuracia=None):
        if funcao_erro is None:
            self.cost_flag=False
        else:
            self.cost_flag=True
            self.cost_weight=theano.shared(np.float32(1))
        self.funcao_erro=funcao_erro
        self.funcao_acuracia=funcao_acuracia
        self.set_input_flag=True
        self.params=[]
        self.updates=[]
        self.givens={}
        
    def cost(self,network):
        return self.funcao_erro(self.outputs_dropout,network.y,network)
    def accuracy_check(self,network):
        return [self.funcao_erro(self.outputs_dropout,network.y,network),self.funcao_acuracia(self.outputs,network.y,network)]
    def set_input(self,inpt,inpt_dropout):
        self.inpt=inpt
        self.inpt_dropout=inpt_dropout
        self.set_input_flag=False

class Conv(Layer):
    def __init__(self,dim_input,dim_filter,stride,padding,poolsize,ativ_func,dropout_rate,funcao_erro=None,funcao_acuracia=None):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Convolution layer'
        self.dim_input=dim_input
        self.n_inputs=dim_input[0]*dim_input[1]*dim_input[2]
        
        self.dim_filter=dim_filter
        self.stride=stride
        self.padding=padding
        self.poolsize=poolsize
        
        self.ativ_func=ativ_func
        self.dropout_rate=dropout_rate
        
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(self.dim_filter[1]*self.dim_filter[2])), size=(self.dim_filter[0:1]+self.dim_input[0:1]+self.dim_filter[1:])),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
                np.zeros(self.dim_filter[0],dtype=theano.config.floatX),
            borrow=True)
        self.params=[self.w,self.b]
        
        self.dim_output=[self.dim_filter[0],int(np.ceil(((self.dim_input[1]+2*self.padding[0])-(self.dim_filter[1]-1))/(self.stride[0]*self.poolsize[0]))),int(np.ceil(((self.dim_input[2]+2*self.padding[1])-(self.dim_filter[2]-1))/(self.stride[1]*self.poolsize[1])))]
        self.n_outputs=self.dim_output[0]*self.dim_output[1]*self.dim_output[2]
        
    def set_input(self,inpt,inpt_dropout):
        Layer.set_input(self,inpt,inpt_dropout)
        self.inputs=self.inpt.reshape(self.dim_input+[self.inpt.shape[1]]).transpose(3,0,1,2)
        self.inputs_dropout=dropout(self.inpt_dropout.reshape(self.dim_input+[self.inpt_dropout.shape[1]]).transpose(3,0,1,2), self.dropout_rate)
        input_filtered=conv2d(input=self.inputs,
                              filters=self.w,
                              filter_shape=(self.dim_filter[0:1]+self.dim_input[0:1]+self.dim_filter[1:]),
                              border_mode=self.padding,
                              subsample=self.stride
                              )
        input_filtered=pool.pool_2d(input=input_filtered, ws=self.poolsize, ignore_border=True)
            
        input_filtered_dropout=conv2d(input=self.inputs_dropout,
                              filters=self.w,
                              filter_shape=(self.dim_filter[0:1]+self.dim_input[0:1]+self.dim_filter[1:]),
                              border_mode=self.padding,
                              subsample=self.stride
                              )
        input_filtered_dropout=pool.pool_2d(input=input_filtered_dropout, ws=self.poolsize, ignore_border=True)
        
        self.outputs=self.ativ_func(input_filtered+self.b.dimshuffle('x', 0, 'x', 'x')).transpose(1,2,3,0).reshape([self.n_outputs,self.inpt.shape[1]])
        self.outputs_dropout=self.ativ_func((1-self.dropout_rate)*input_filtered_dropout+self.b.dimshuffle('x', 0, 'x', 'x')).transpose(1,2,3,0).reshape([self.n_outputs,self.inpt_dropout.shape[1]])
        
class FC(Layer):
    def __init__(self,n_input,n_neurons,ativ_func,dropout_rate=0,funcao_erro=None,funcao_acuracia=None):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Fully connected layer'
        self.n_inputs,self.n_neurons,self.ativ_func,self.dropout_rate=n_input,n_neurons,ativ_func,dropout_rate
        
        self.n_outputs=n_neurons
        
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(n_input)), size=[n_neurons,self.n_inputs]),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
                np.zeros((n_neurons,),dtype=theano.config.floatX),
            borrow=True)
        self.params=[self.w,self.b]
        
    def set_input(self,inpt,inpt_dropout):
        Layer.set_input(self,inpt,inpt_dropout)
        self.inputs=self.inpt
        self.inputs_dropout=dropout(self.inpt_dropout, self.dropout_rate)
        
        self.outputs=self.ativ_func(((1-self.dropout_rate)*tensor.dot(self.w,self.inputs)+self.b.dimshuffle(0,'x')).transpose()).transpose()
        self.outputs_dropout=self.ativ_func((tensor.dot(self.w,self.inputs_dropout)+self.b.dimshuffle(0,'x')).transpose()).transpose()
        self.label=tensor.round(self.outputs)
        
        
class Sparce(Layer):
    def __init__(self,n_input,funcao_erro=None,funcao_acuracia=None):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Sparce converter layer'
        self.n_input=n_input
        

    def set_input(self,inpt,inpt_dropout):
        Layer.set_input(self,inpt,inpt_dropout)
        self.inputs=self.inpt.toarray()
        self.inputs_dropout=self.inpt_dropout.toarray()
        
        self.outputs=self.inputs
        self.outputs_dropout=self.inputs_dropout
        
def dropout(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*tensor.cast(mask, theano.config.floatX)  

class BN(Layer):
    def __init__(self,n_input,update_rate,funcao_erro=None,funcao_acuracia=None):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Batch normalization layer'
        self.update_rate=update_rate
        self.n_inputs=n_input
        self.n_outputs=n_input
        
        self.mean=theano.shared(np.zeros([n_input,],dtype=theano.config.floatX))
        self.std=theano.shared(np.zeros([n_input,],dtype=theano.config.floatX)+1)
        
        self.scale_mean=theano.shared(np.cast[theano.config.floatX](1.0))
        self.scale_std=theano.shared(np.cast[theano.config.floatX](1.0))
        
        self.params=[self.scale_mean,self.scale_std]
    def set_input(self,inpt,inpt_dropout):
        Layer.set_input(self,inpt,inpt_dropout)
        self.inputs=self.inpt
        self.inputs_dropout=self.inpt_dropout
        
        self.updates=[(self.mean,(self.mean*(1-self.update_rate)+self.update_rate*self.inputs_dropout.mean(axis=1)).astype(theano.config.floatX)),
                      (self.std,((self.std*(1-self.update_rate))+(self.inputs_dropout.std(ddof=1,axis=1)*self.update_rate)).astype(theano.config.floatX))]
        self.outputs_dropout=self.scale_std*((self.inputs_dropout-self.inputs_dropout.mean(axis=1).astype(theano.config.floatX).dimshuffle(0,'x'))/((self.inputs_dropout.std(ddof=1,axis=1).astype(theano.config.floatX).dimshuffle(0,'x')+10**(-10))))+self.scale_mean
        self.outputs=self.scale_std*((self.inputs-self.mean.dimshuffle(0,'x'))/(self.std.dimshuffle(0,'x')+10**(-10)))+self.scale_mean
        
class integration(Layer):
    def __init__(self,output,funcao_erro=None,funcao_acuracia=None):
        self.type='Integration layer'
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.output_var=output
    def set_input(self,inpt,inpt_dropout):
        Layer.set_input(self,inpt,inpt_dropout)
        self.outputs,self.outputs_dropout=self.output_var        
        
class DeConv(Layer):
    def __init__(self,dim_input,dim_filter,stride,padding,ativ_func,dropout_rate,funcao_erro=None,funcao_acuracia=None):
        Layer.__init__(self,funcao_erro,funcao_acuracia)
        self.type='Deconvolution layer'
        self.dim_input=dim_input
        self.n_inputs=dim_input[0]*dim_input[1]*dim_input[2]
        
        self.dim_filter=dim_filter
        self.stride=stride
        self.padding=padding
        
        self.ativ_func=ativ_func
        self.dropout_rate=dropout_rate
        
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(self.dim_filter[2]*self.dim_filter[3])), size=self.dim_filter),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
                np.zeros(self.dim_filter[1],dtype=theano.config.floatX),
            borrow=True)
        self.params=[self.w,self.b]
        
        self.dim_output=[self.dim_filter[1],int(np.ceil(self.stride[0]*self.dim_input[1]+self.dim_filter[2]-1-2*self.padding[0])),int(np.ceil(self.stride[1]*self.dim_input[2]+self.dim_filter[3]-1-2*self.padding[1]))]
        self.n_outputs=self.dim_output[0]*self.dim_output[1]*self.dim_output[2]
        
    def set_input(self,inpt,inpt_dropout):
        Layer.set_input(self,inpt,inpt_dropout)
        self.inputs=self.inpt.reshape(self.dim_input+[self.inpt.shape[1]]).transpose(3,0,1,2)
        
        self.inputs_dropout=dropout(self.inpt_dropout.reshape(self.dim_input+[self.inpt_dropout.shape[1]]).transpose(3,0,1,2), self.dropout_rate)
        input_filtered=conv2d_transpose(input=self.inputs,
                              filters=self.w,
                              output_shape=[None]+self.dim_output,
                              filter_shape=self.dim_filter,
                              border_mode=self.padding,
                              input_dilation=self.stride
                              )
            
        input_filtered_dropout=conv2d_transpose(input=self.inputs_dropout,
                              filters=self.w,
                              output_shape=[None]+self.dim_output,
                              filter_shape=self.dim_filter,
                              border_mode=self.padding,
                              input_dilation=self.stride
                              )
        
        self.outputs=self.ativ_func(input_filtered+self.b.dimshuffle('x', 0, 'x', 'x')).transpose(1,2,3,0).reshape([self.n_outputs,self.inpt.shape[1]])
        self.outputs_dropout=self.ativ_func((1-self.dropout_rate)*input_filtered_dropout+self.b.dimshuffle('x', 0, 'x', 'x')).transpose(1,2,3,0).reshape([self.n_outputs,self.inpt_dropout.shape[1]])