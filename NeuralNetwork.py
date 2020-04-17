import time

import dill
import numpy as np
import theano
from theano import sparse
import theano.tensor as tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from IPython.display import clear_output
from theano.tensor import shared_randomstreams

print('Theano Version = ',theano.__version__)
print('Network Version = ','1.1.0')

class NeuralNetwork:
    def __init__(self,dataset,layers,validation_dataset,sparse_flag=False,output_layer=-1,transformacoes=None):
        
        self.version='CNN_V1_1_0'
        
        self.output_layer=output_layer
        
        self.transformacoes=transformacoes
        
        self.layers=layers
        if sparse_flag=='csr':
            self.x = sparse.csr_matrix("x")
        elif sparse_flag=='csc':
            self.x = sparse.csc_matrix("x")
        elif not(sparse_flag):
            self.x = tensor.fmatrix("x")
        self.y = tensor.fmatrix("y")
        
        self.params = [param for layer in self.layers for param in layer.params]
            
        self.input=self.x
        layers[0].set_input(self.input,self.input)
        
        for i in range(len(layers)-1):
            layers[i+1].set_input(layers[i].outputs,layers[i].outputs_dropout)
        
        self.classificar=theano.function([self.x],layers[-1].outputs)
        if self.output_layer!=-1:
            self.encode=theano.function([self.x],layers[self.output_layer].outputs)
            self.decode=theano.function([self.x],layers[-1].outputs, givens={layers[self.output_layer+1].inputs:self.x})
        
        if validation_dataset is not None and dataset is not None:
            
            self.flag_shared=(type(dataset[0])==theano.tensor.sharedvar.TensorSharedVariable) #falta colocar suporter para gpu nesta linha
            
            self.dataset_x,self.dataset_y=dataset[0],dataset[-1]
            self.validation_x,self.validation_y=validation_dataset[0],validation_dataset[-1]
            
            if self.flag_shared:
                
                self.len_validation=len(self.validation_y.get_value()[0])
                self.len_dataset=len(self.dataset_y.get_value()[0])
            else:
                self.len_dataset=len(self.dataset_y[0])
                self.len_validation=len(self.validation_y[0])
            
            self.len_subsets=theano.shared(1)
            self.check_size_intern=theano.shared(1)
            self.check_size_validation=theano.shared(1)
            
            self.n_subsets=self.len_dataset//self.len_subsets.get_value()
            self.check_intern_subsets_len=self.len_dataset//self.check_size_intern.get_value()
            self.validation_subsets_len=self.len_validation//self.check_size_validation.get_value()
            
            self.historico_treino=[[],[]]
            self.historico_params=[]

            self.weight_penalty=theano.shared(0.0, borrow=True)
            self.no_sparcity_penalty=theano.shared(0.0, borrow=True)
            self.learnrate=theano.shared(1.0, borrow=True)

            l2_norm_squared = sum([(layer.params[0]**2).sum() for layer in self.layers if len(layer.params)>1])
            accuracy=self.layers[-1].accuracy_check(self)   
            if self.output_layer==-1:
                regularization=0.5*self.weight_penalty*l2_norm_squared/self.n_subsets
                cost = self.layers[-1].cost(self)+regularization
                error_list=[accuracy[0]+regularization,accuracy[1]]
            else:
                regularization=self.no_sparcity_penalty*self.layers[self.output_layer].cost(self)+\
                       0.5*self.weight_penalty*l2_norm_squared/self.n_subsets
                cost = self.layers[-1].cost(self)+regularization
                error_list=[accuracy[0]+regularization,accuracy[1]]
            grads = tensor.grad(cost, self.params)
            updates = [(param, (param-self.learnrate*grad).astype(theano.config.floatX))
                       for param, grad in zip(self.params, grads)]
            
            if self.flag_shared:
                
                i=tensor.lscalar()

                self.erros=theano.function([i],error_list,givens={
                        self.x:
                        self.validation_x[:,i*self.check_size_validation:(i+1)*self.check_size_validation],
                        self.y:
                        self.validation_y[:,i*self.check_size_validation:(i+1)*self.check_size_validation]})

                self.erros_interno=theano.function([i],error_list,givens={
                        self.x:
                        self.dataset_x[:,i*self.check_size_intern:(i+1)*self.check_size_intern],
                        self.y:
                        self.dataset_y[:,i*self.check_size_intern:(i+1)*self.check_size_intern]
                    })

                self.train_mb = theano.function(
                    [i], cost, updates=updates,
                    givens={
                        self.x:
                        self.dataset_x[:,i*self.len_subsets:(i+1)*self.len_subsets],
                        self.y:
                        self.dataset_y[:,i*self.len_subsets:(i+1)*self.len_subsets]
                    })
            else:
                
                self.erros=theano.function([self.x,self.y],error_list)
            
                self.train_mb = theano.function(
                    [self.x,self.y], cost, updates=updates)

    def otimize(self,no_better_interval,weight_penalty,learnrate,batch_sizes,no_sparcity_penalty=0,dataset_check_flag=True):
        self.weight_penalty.set_value(weight_penalty)
        self.no_sparcity_penalty.set_value(no_sparcity_penalty)
        self.learnrate.set_value(learnrate)
        
        self.check_size_intern.set_value(batch_sizes[2])
        self.check_size_validation.set_value(batch_sizes[1])
        self.len_subsets.set_value(batch_sizes[0])
        
        self.n_subsets=self.len_dataset//self.len_subsets.get_value()
        self.check_intern_subsets_len=self.len_dataset//self.check_size_intern.get_value()
        self.validation_subsets_len=self.len_validation//self.check_size_validation.get_value()
        
        loading_symbols=['\\','|','/','-']
        loading_index=0
        
        no_better_count=0
        times=len(self.historico_treino[1])
        try:
            while True:
                times+=1
                tempo=time.time()
                if self.flag_shared:
                    for j in range(self.n_subsets):
                        indice=np.random.randint(self.n_subsets)
                        iteration=self.train_mb(indice)
                    if self.validation_subsets_len>1:
                        erros=np.mean([self.erros(i) for i in range(self.validation_subsets_len)],axis=0)
                    else:
                        erros=self.erros(0)
                else:
                    
                    indices=np.asarray(range(self.len_dataset))
                    np.random.shuffle(indices)
                    self.dataset_x,self.dataset_y=self.dataset_x[:,indices],self.dataset_y[:,indices]
                    for j in range(self.n_subsets):
                        indice=j
                        iteration=self.train_mb(self.dataset_x[:,int(indice*batch_sizes[0]):int((indice+1)*batch_sizes[0])],
                                                self.dataset_y[:,int(indice*batch_sizes[0]):int((indice+1)*batch_sizes[0])])
                    if self.validation_subsets_len>1:
                        erros=[]
                        for i in range(self.validation_subsets_len):
                            erros.append(self.erros(self.validation_x[:,i*batch_sizes[1]:(i+1)*batch_sizes[1]],
                                                    self.validation_y[:,i*batch_sizes[1]:(i+1)*batch_sizes[1]]))
                        erros=np.mean(erros,axis=0)
                    else:
                        erros=self.erros(self.dataset_x,self.dataset_y)

                self.historico_treino[0].append(erros[0])
                self.historico_treino[1].append(erros[1])

                self.historico_params+=[[k.get_value() for k in self.params]]
                clear_output(wait=True)
                if dataset_check_flag:
                    if self.flag_shared:
                        erros_interno=np.mean([self.erros_interno(k) for k in range(self.check_intern_subsets_len)],axis=0)
                    else:
                        erros_interno=[]
                        for i in range(self.check_intern_subsets_len):
                            erros_interno.append(self.erros(self.dataset_x[:,i*batch_sizes[2]:(i+1)*batch_sizes[2]],
                                                            self.dataset_y[:,i*batch_sizes[2]:(i+1)*batch_sizes[2]]))
                        erros_interno=np.mean(erros_interno,axis=0)
                    print('Dados de treino',' ',loading_symbols[loading_index%4],'\n',
                          'Função de erro - ',erros_interno[0],'\n',
                          'Função de acurácia - ',erros_interno[1])
                print('Função de acurácia',' ',loading_symbols[loading_index%4],'\n',
                      'Atual - ',self.historico_treino[1][-1],'\n',
                      'Melhor - ',max(self.historico_treino[1]))
                print('Função de erro',' ',loading_symbols[loading_index%4],'\n',
                      'Atual - ',self.historico_treino[0][-1],'\n',
                      'Learnrate - ',learnrate,' - Weight Penalty - ',weight_penalty,'\n',
                      'Batch size - ',batch_sizes[0],' - No Sparcity Penalty - ',no_sparcity_penalty,'\n',
                      'Tempo - ',time.time()-tempo,'\n',
                      'Iteração - ',times,' - Shared - ',self.flag_shared)
                loading_index+=1
                if times%no_better_interval==0 and times>2*no_better_interval:
                    if np.mean(self.historico_treino[1][-no_better_interval:])>=np.mean(self.historico_treino[1][-2*no_better_interval:-no_better_interval]):
                        print('Early Stop')
                        break
        except KeyboardInterrupt:
            print('Early Stop')
            
    def export_network(self,nome='default',diretorio='',tipo='layers'):
        if nome=='default':
            date=time.localtime()
            nome=self.version+'_'+str(date.tm_hour)+'h_'+str(date.tm_mday)+'_'+str(date.tm_mon)+'_'+str(date.tm_year)+'_'+str(max(self.historico_treino[0])).ljust(6,'0')[2:6]+'.'+tipo
        arquivo=open(diretorio+nome,'wb')
        if tipo=='layers':
            dill.dump(self.layers, arquivo)
        elif tipo=='cnn':
            dill.dump(self,arquivo)
        else:
            arquivo.close()
            raise TypeError('Tipo de exportação inválida, valor informado diferente de "cnn" e "layers".')
        arquivo.close()
    
    def update_parametros(self,parametros='default'):
        if parametros=='default':
            parametros=self.historico_params[np.argmin(np.asarray(self.historico_treino[0]))]
        for i,j in zip(self.params,parametros):
            i.set_value(j)
            
class ConvLayer:
    def __init__(self,dim_input,dim_filter,poolsize,ativ_func,dropout_rate):
        self.dim_input=dim_input
        
        self.dim_filter=dim_filter
        self.poolsize=poolsize
        
        self.ativ_func=ativ_func
        self.dropout_rate=dropout_rate
        
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(self.dim_filter[1]*self.dim_filter[2])), size=self.dim_filter),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=self.dim_filter[0]),
                dtype=theano.config.floatX),
            borrow=True)
        self.params=[self.w,self.b]
        
        self.n_outputs=int(self.dim_filter[0]*((self.dim_input[1]-(self.dim_filter[2]-1))/self.poolsize[0])*((self.dim_input[2]-(self.dim_filter[3]-1))/self.poolsize[1]))
        
    def set_input(self,inpt,inpt_dropout):
        self.inputs=inpt.reshape(self.dim_input+[inpt.shape[1]]).transpose(3,0,1,2)
        self.inputs_dropout=dropout_layer(inpt_dropout.reshape(self.dim_input+[inpt.shape[1]]).transpose(3,0,1,2), self.dropout_rate)
        input_filtered=conv2d(input=self.inputs,filters=self.w,filter_shape=self.dim_filter)
        input_filtered=pool.pool_2d(input=input_filtered, ws=self.poolsize, ignore_border=True)
            
        input_filtered_dropout=conv2d(input=self.inputs_dropout,filters=self.w,filter_shape=self.dim_filter)
        input_filtered_dropout=pool.pool_2d(input=input_filtered_dropout, ws=self.poolsize, ignore_border=True)
        
        self.outputs=self.ativ_func(input_filtered+self.b.dimshuffle('x', 0, 'x', 'x')).transpose(1,2,3,0).reshape([self.n_outputs,inpt.shape[1]])
        self.outputs_dropout=self.ativ_func((1-self.dropout_rate)*input_filtered_dropout+self.b.dimshuffle('x', 0, 'x', 'x')).transpose(1,2,3,0).reshape([self.n_outputs,inpt.shape[1]])
        
class FCLayer:
    def __init__(self,n_input,n_neurons,ativ_func,dropout_rate=0,funcao_erro=None,funcao_acuracia=None):
        self.n_input,self.n_neurons,self.ativ_func,self.dropout_rate=n_input,n_neurons,ativ_func,dropout_rate
        self.funcao_erro=funcao_erro
        self.funcao_acuracia=funcao_acuracia
        
        self.n_outputs=n_neurons
        
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/(n_input)), size=[n_neurons,self.n_input]),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(n_neurons,)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params=[self.w,self.b]
    
    def cost(self,network):
        return self.funcao_erro(self.outputs_dropout,network.y,network)
    def accuracy_check(self,network):
        return [self.funcao_erro(self.outputs_dropout,network.y,network),self.funcao_acuracia(self.outputs,network.y,network)]
    def set_input(self,inpt,inpt_dropout):
        self.inputs=inpt
        self.inputs_dropout=dropout_layer(inpt_dropout, self.dropout_rate)
        
        self.outputs=self.ativ_func(((1-self.dropout_rate)*tensor.dot(self.w,self.inputs)+self.b.dimshuffle(0,'x')).transpose()).transpose()
        self.outputs_dropout=self.ativ_func((tensor.dot(self.w,self.inputs_dropout)+self.b.dimshuffle(0,'x')).transpose()).transpose()
        self.label=tensor.round(self.outputs)
        
class SparceLayer:
    def __init__(self,n_input,batch_size):
        identidade=lambda x:x
        self.n_input,self.n_neurons,self.ativ_func,self.dropout_rate=n_input,n_input,identidade,0
        self.batch_size=batch_size
        self.params=[]

    def set_input(self,inpt,inpt_dropout):
        self.inputs=inpt.toarray()
        self.inputs_dropout=dropout_layer(inpt_dropout.toarray(), self.dropout_rate)
        
        self.outputs=self.inputs
        self.outputs_dropout=self.inputs_dropout
        
def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*tensor.cast(mask, theano.config.floatX)  
