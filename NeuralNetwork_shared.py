import time

import dill
import numpy as np
import theano
from theano import sparse
import theano.tensor as tensor
from Functions import report as default

print('Theano Version = ',theano.__version__)
print('Network Version = ','1.2.0 Shared')

class NeuralNetwork:
    def __init__(self,dataset,layers,validation_dataset,report_func=default,sparse_flag=False,output_layer=-1,transformacoes=None):
        
        self.version='NN_V1_2_0_shared'
        
        self.output_layer=output_layer
        self.report=report_func
        
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
            
            if not(self.flag_shared):
                raise(TypeError("Este código é para dataset's no formato Tensor Shared, mude o formato dos dados ou use o outro códgio"))
            
            self.dataset_x,self.dataset_y=dataset[0],dataset[-1]
            self.validation_x,self.validation_y=validation_dataset[0],validation_dataset[-1]
            
            self.len_validation=len(self.validation_y.get_value()[0])
            self.len_dataset=len(self.dataset_y.get_value()[0])
            
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
                
            self.fric=theano.shared(np.cast[theano.config.floatX](0))
            
            self.update_rate_1=theano.shared(np.cast[theano.config.floatX](1))
            self.update_rate_2=theano.shared(np.cast[theano.config.floatX](1))
            
            grads = tensor.grad(cost, self.params)
            self.grads_old = [theano.shared((param.get_value()*0).astype(theano.config.floatX)) for param in self.params]
            
            self.adam_moment_1 = [theano.shared((param.get_value()*0).astype(theano.config.floatX)) for param in self.params]
            self.adam_moment_2 = [theano.shared((param.get_value()*0).astype(theano.config.floatX)) for param in self.params]
            self.adam_counter = theano.shared(np.int16(1))
            
            updates_sgd = [(param, (param-self.learnrate*(grad_old*self.fric+(1-self.fric)*grad)).astype(theano.config.floatX))
                       for param, grad,grad_old in zip(self.params, grads,self.grads_old)]+\
                       sum([layer.updates for layer in self.layers],[])+\
                       [(grad_old,grad_old*self.fric+(1-self.fric)*grad)
                       for grad_old,grad in zip(self.grads_old,grads)]
            
            updates_adam = [(param, (param-self.learnrate*(((moment_1*self.update_rate_1+(1-self.update_rate_1)*grad)/(1-self.update_rate_1**self.adam_counter))/((((moment_2*self.update_rate_2+(1-self.update_rate_2)*(grad**2))**0.5)/(1-self.update_rate_2**self.adam_counter))+10**(-40)))).astype(theano.config.floatX))
                       for param, grad,moment_1,moment_2 in zip(self.params, grads,self.adam_moment_1,self.adam_moment_2)]+\
                       [(moment_1,(moment_1*self.update_rate_1+(1-self.update_rate_1)*grad).astype(theano.config.floatX))
                       for moment_1,grad in zip(self.adam_moment_1,grads)]+\
                       sum([layer.updates for layer in self.layers],[])+\
                       [(moment_2,(moment_2*self.update_rate_2+(1-self.update_rate_2)*(grad**2)).astype(theano.config.floatX))
                       for moment_2,grad in zip(self.adam_moment_2,grads)]+\
                       [(self.adam_counter,self.adam_counter+1)]
            
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

            self.train_mb_SGD = theano.function(
                [i], cost, updates=updates_sgd,
                givens={
                    self.x:
                    self.dataset_x[:,i*self.len_subsets:(i+1)*self.len_subsets],
                    self.y:
                    self.dataset_y[:,i*self.len_subsets:(i+1)*self.len_subsets]
                })
            
            self.train_mb_ADAM = theano.function(
                [i], cost, updates=updates_adam,
                givens={
                    self.x:
                    self.dataset_x[:,i*self.len_subsets:(i+1)*self.len_subsets],
                    self.y:
                    self.dataset_y[:,i*self.len_subsets:(i+1)*self.len_subsets]
                })

    def otimize(self,no_better_interval,weight_penalty,learnrate,batch_sizes,no_sparcity_penalty=0):
        self.weight_penalty.set_value(weight_penalty)
        self.no_sparcity_penalty.set_value(no_sparcity_penalty)
        
        self.otimization=learnrate
        
        if learnrate[0]=='SGD':
            self.learnrate.set_value(learnrate[1])
            self.fric.set_value(learnrate[2])
            self.otimization_func=self.train_mb_SGD
            for i in range(len(self.params)):
                self.grads_old[i].set_value(self.params[i].get_value()*0)
                
        elif learnrate[0]=='ADAM':
            self.learnrate.set_value(learnrate[1])
            self.update_rate_1.set_value(learnrate[2])
            self.update_rate_2.set_value(learnrate[3])
            self.otimization_func=self.train_mb_ADAM
            
            self.adam_counter.set_value(np.int16(1))
            for i in range(len(self.params)):
                self.adam_moment_1[i].set_value(self.params[i].get_value()*0)
                self.adam_moment_2[i].set_value(self.params[i].get_value()*0)
        
        self.check_size_intern.set_value(batch_sizes[2])
        self.check_size_validation.set_value(batch_sizes[1])
        self.len_subsets.set_value(batch_sizes[0])
        
        self.n_subsets=self.len_dataset//self.len_subsets.get_value()
        self.check_intern_subsets_len=self.len_dataset//self.check_size_intern.get_value()
        self.validation_subsets_len=self.len_validation//self.check_size_validation.get_value()
        
        no_better_count=0
        self.times=len(self.historico_treino[1])
        try:
            while True:
                self.times+=1
                self.tempo=time.time()
                
                for j in range(self.n_subsets):
                    indice=np.random.randint(self.n_subsets)
                    iteration=self.otimization_func(indice)
                if self.validation_subsets_len>1:
                    erros=np.mean([self.erros(i) for i in range(self.validation_subsets_len)],axis=0)
                else:
                    erros=self.erros(0)

                self.historico_treino[0].append(erros[0])
                self.historico_treino[1].append(erros[1])

                self.historico_params+=[[k.get_value() for k in self.params]]
                self.loading_symbols=['\\','|','/','-']
                self.loading_index=0
                self.report(self)
                self.loading_index+=1
                if self.times%no_better_interval==0 and self.times>2*no_better_interval:
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