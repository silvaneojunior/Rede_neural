import time
from pygpu.gpuarray import GpuArrayException

import dill
import numpy as np
import theano
from theano import sparse
import theano.tensor as tensor
from Functions import report as default,report_GAN as default_GAN
import Layers

print('Theano Version = ',theano.__version__)
print('Network Version = ','1.2.0')

class NeuralNetwork:
    def __init__(self,dataset,layers,report_func=default,sparse_flag=False,transformacoes=None):
        
        self.version='NN_V1_2_0'
        
        if sparse_flag=='csr':
            self.x = sparse.csr_matrix("x")
        elif sparse_flag=='csc':
            self.x = sparse.csc_matrix("x")
        elif not(sparse_flag):
            self.x = tensor.fmatrix("x")
        self.y = tensor.fmatrix("y")
        
        self.report=report_func
        self.layers=layers
        self.params = [param for layer in layers for param in layer.params]
        self.transformacoes=transformacoes
        self.historico={'cost':[],'accuracy':[],'params':[]}
        
        if dataset is not None:
            self.times=0
            self.len_subsets=1
            self.check_size=1
            
            self.update_dataset(dataset)
        
    def update_dataset(self,dataset):
        self.dataset=dataset
        self.dataset_x,self.dataset_y=dataset('train')
        self.validation_x,self.validation_y=dataset('validation')

        self.len_dataset=self.dataset_y.shape[1]
        self.len_validation=self.validation_y.shape[1]

        self.flag_shared=type(self.dataset_y)==theano.tensor.sharedvar.TensorSharedVariable if theano.config.device=='cpu' else type(self.dataset_y)==theano.gpuarray.type.GpuArraySharedVariable

        if self.flag_shared:
            raise(TypeError('Este código não tem suporte para Tensor Shared como dataset, use o outro código.'))


        self.n_subsets=self.len_dataset//self.len_subsets
        self.check_intern_subsets_len=self.len_dataset//self.check_size
        self.validation_subsets_len=self.len_validation//self.check_size
        
    def export_network(self,nome='default',diretorio='',tipo='layers'):
        if nome=='default':
            date=time.localtime()
            nome=self.version+'_'+str(date.tm_hour)+'h_'+str(date.tm_mday)+'_'+str(date.tm_mon)+'_'+str(date.tm_year)+'.'+tipo
        arquivo=open(diretorio+nome,'wb')
        if tipo=='layers':
            dill.dump(self.layers, arquivo)
        elif tipo=='dnn':
            dill.dump(self,arquivo)
        else:
            arquivo.close()
            raise TypeError('Tipo de exportação inválida, valor informado diferente de "dnn" e "layers".')
        arquivo.close()
    
    def update_parametros(self,parametros='default'):
        if parametros=='default':
            parametros=self.historico['params'][np.argmin(np.asarray(self.historico['cost']))]
        for i,j in zip(self.params,parametros):
            i.set_value(j)
            
            
class FowardNetwork(NeuralNetwork):
    def __init__(self,dataset,layers,report_func=default,sparse_flag=False,output_layer=-1,transformacoes=None):
        
        self.version='FN_V1_2_0'
                
        NeuralNetwork.__init__(self,dataset,layers,report_func,sparse_flag,transformacoes)
        
        self.output_layer=output_layer
        
        givens={}
        for i in layers:
            givens.update(i.givens)
        
        self.inputs=self.x
        if layers[0].set_input_flag:
            layers[0].set_input(self.inputs,self.inputs)
        else:
            givens.update({layers[0].inpt:self.x,layers[0].inpt_dropout:self.x})
        layers[0].set_input(self.inputs,self.inputs)
        for i in range(len(layers)-1):
            if layers[i+1].set_input_flag:
                layers[i+1].set_input(layers[i].outputs,layers[i].outputs)
        
        pre_classificar_list=theano.function([self.x],layers[-1].outputs,givens=givens)
        if output_layer!=-1:
            pre_encode_list=theano.function([self.x],layers[output_layer].outputs,givens=givens)
            decode_givens={layers[output_layer].outputs:self.x}
            decode_givens.update(givens)
            pre_decode_list=theano.function([self.x],layers[-1].outputs, givens=decode_givens)
        
        
        if dataset is not None:
            
            ADAM=create_otimizer_ADAM(self,layers,self.params,givens)
            SGD=create_otimizer_SGD(self,layers,self.params,givens)
            
            self.train_mb_ADAM,self.update_params_ADAM,self.reset_hist_ADAM=ADAM
            self.train_mb_SGD,self.update_params_SGD,self.reset_hist_SGD=SGD
            
            accuracy=layers[-1].accuracy_check(self)
            pre_erros_list=theano.function([self.x,self.y],accuracy,givens=givens)
            
            if theano.config.device=='cpu':
                length=int(10**(np.ceil(np.log10(self.len_dataset))))
                flag_concluido=False
                while flag_concluido!=True and length>=1:
                    try:
                        pre_classificar_list(self.dataset_x[:,:min(length,self.len_dataset)])
                        flag_concluido=True
                    except (GpuArrayException,RuntimeError) as error:
                        if 'BaseCorrMM: Failed to allocate output of' in ''.join(error.args):
                            length=int(length//10)
                        else:
                            raise error
            else:
                data_size=int(theano.config.floatX[-2:])/8
                size_layer=max([layer.n_outputs for layer in layers if layer.type!='Integration layer'])
                single_example_size=data_size*size_layer
                length=int(10**(np.floor(np.log10(2*(1024**3)//single_example_size))))

            if length<1:
                raise ValueError('Esta combinação de dados e arquitetura exigem mais memória do que o disponível. Modifique a arquitetura, mude os dados ou, caso esteja usando a GPU, troque para a CPU.')
            else:
                self.check_size=min(length,self.len_dataset)
                self.check_size=10000
                
            texto='self.{0}=lambda *inputs:[pre_{0}(*inputs)] \
if inputs[0].shape[1]<=self.check_size \
else np.concatenate([[pre_{0}( \
*[k[:,i*self.check_size:(i+1)*self.check_size] for k in inputs]\
)] \
for i in range(inputs[0].shape[1]//self.check_size)] + \
([] if inputs[0].shape[1]%self.check_size==0 \
else [[pre_{0}(*[k[:,(inputs[0].shape[1]//self.check_size)*self.check_size:] for k in inputs])]]),\
axis=2)'
            exec(texto.format('classificar_list'),{'self':self,'np':np,'pre_classificar_list':pre_classificar_list})
            self.classificar=lambda *inputs:self.classificar_list(*inputs)[0]
            exec(texto.format('erros_list'),{'self':self,'np':np,'pre_erros_list':pre_erros_list})
            self.erros=lambda *inputs:np.mean(self.erros_list(*inputs),axis=0).flatten()
            
            if output_layer!=-1:
                
                exec(texto.format('encode_list'),{'self':self,'np':np,'pre_encode_list':pre_encode_list})
                self.encode=lambda *inputs:self.encode_list(*inputs)[0]
                exec(texto.format('decode_list'),{'self':self,'np':np,'pre_decode_list':pre_decode_list})
                self.decode=lambda *inputs:self.decode_list(*inputs)[0]
    def otimize(self,number_of_epochs,no_better_interval,weight_penalty,learning_method,batch_size,update_data=False):
        self.otimization=learning_method
        self.weight_penalty=weight_penalty
        
        if update_data:
            self.update_dataset(dataset)
        
        exec(('self.otimization_func=self.train_mb_{0} \n'+\
             'self.reset_hist=self.reset_hist_{0} \n'+\
             'self.update_params=self.update_params_{0}').format(learning_method[0]),{'self':self})
            
        self.reset_hist()
        self.update_params(weight_penalty,*learning_method[1:])

        self.len_subsets=batch_size
        
        self.n_subsets=self.len_dataset//self.len_subsets
        self.check_intern_subsets_len=self.len_dataset//self.check_size
        self.validation_subsets_len=self.len_validation//self.check_size
        
        no_better_count=0
        initial_epoch=self.times
        
        self.loading_symbols=['\\','|','/','-']
        self.loading_index=0
        
        try:
            while self.times-initial_epoch<number_of_epochs:
                self.times+=1
                self.tempo=time.time()
                    
                indices=np.asarray(range(self.len_dataset))
                np.random.shuffle(indices)
                self.dataset_x,self.dataset_y=self.dataset_x[:,indices],self.dataset_y[:,indices]
                for indice in range(self.n_subsets):
                    iteration=self.otimization_func(self.dataset_x[:,int(indice*self.len_subsets):int((indice+1)*self.len_subsets)],
                                            self.dataset_y[:,int(indice*self.len_subsets):int((indice+1)*self.len_subsets)])
                    
                erros=self.erros(self.validation_x,self.validation_y)

                self.historico['cost'].append(erros[0])
                self.historico['accuracy'].append(erros[1])

                self.historico['params'].append([k.get_value() for k in self.params])
                
                self.report(self)
                
                self.loading_index+=1
                if self.times%no_better_interval==0 and self.times>2*no_better_interval:
                    if np.mean(self.historico['accuracy'][-no_better_interval:])>=np.mean(self.historico['accuracy'][-2*no_better_interval:-no_better_interval]):
                        print('Finished')
                        break
        except KeyboardInterrupt:
            print('Early Stop')
        
class GANetwork(NeuralNetwork):
    def __init__(self,dataset_class,dataset_gen,layers_gen,layers_class,report_func=default_GAN,sparse_flag=False,transformacoes=None):
        
        self.version='GAN_V1_2_0'
        
        layers=layers_gen+layers_class
        NeuralNetwork.__init__(self,dataset_class,layers,report_func,sparse_flag,transformacoes)
        
        self.params_gen = [param for layer in layers_gen for param in layer.params]
        self.params_class = [param for layer in layers_class for param in layer.params]
        
        self.inputs=self.x
        
        self.len_dataset=dataset_class('train')[1].shape[1]
        self.len_validation=dataset_class('validation')[1].shape[1]
        self.dataset_gen=dataset_gen
        
        labels_train_class=np.zeros([layers_class[-1].n_outputs,self.len_dataset*2],dtype=theano.config.floatX)
        labels_val_class=np.zeros([layers_class[-1].n_outputs,self.len_validation*2],dtype=theano.config.floatX)

        dataset_y=np.zeros([layers_class[0].n_inputs,self.len_dataset*2],dtype=theano.config.floatX)
        validation_y=np.zeros([layers_class[0].n_inputs,self.len_validation*2],dtype=theano.config.floatX)
        
        layers_gen[-1].cost_flag=False
        
        self.dataset_class=lambda label: [dataset_y,labels_train_class] if label=='train' else [validation_y,labels_val_class]
        
        layers=[]
        
        self.inputs=self.x
        
        layers_gen[0].set_input(self.inputs,self.inputs)
        layers.append(layers_gen[0])
        for i in range(len(layers_gen)-1):
            layers_gen[i+1].set_input(layers_gen[i].outputs,layers_gen[i].outputs_dropout)
            layers.append(layers_gen[i+1])
        
        layers.append(layers_class[0])
        layers[-1].set_input(layers_gen[-1].outputs,layers_gen[-1].outputs_dropout)
        for i in range(1,len(layers_class)):
            layers_class[i].set_input(layers[-1].outputs,layers[-1].outputs_dropout)
            layers.append(layers_class[i])
                
        layers.append(Layers.integration(output=[layers_class[-1].outputs,layers_class[-1].outputs_dropout],
                                    funcao_erro=layers_gen[-1].funcao_erro,
                                    funcao_acuracia=layers_gen[-1].funcao_acuracia))
        
        layers_gen[-1].cost_flag=False
        
        self.classifier=FowardNetwork(dataset=self.dataset_class,
                                    layers=layers_class,
                                    report_func=lambda x: None)
        
        self.generator=FowardNetwork(dataset=self.dataset_gen,
                                    layers=layers_gen+[layers[-1]],
                                    report_func=lambda x: None,
                                    output_layer=-2)
        
        self.check_size=self.generator.check_size
            
        def pre_dataset_class(label):
            generator_data=self.dataset_gen(label)
            observed_data=dataset_class(label)
            
            generator_x=self.generator.encode(generator_data[0])
            
            classifier_x=np.concatenate([generator_x,observed_data[0]],axis=1)
            classifier_y=np.concatenate([np.zeros(observed_data[1].shape,theano.config.floatX),observed_data[1]],axis=1)
            
            return classifier_x,classifier_y
        
        self.dataset_class=pre_dataset_class
        self.classifier.update_dataset(self.dataset_class)
        
        self.gerar=self.generator.encode
        self.classificar=self.classifier.classificar
        
    def otimize(self,number_of_epochs,weight_penalty,learning_method_gen,learning_method_class,batch_size):
        
        initial_epoch=self.times
        try:
            self.generator.otimize(0,
                                   learning_method_gen[0],
                                   weight_penalty[0],
                                   learning_method_gen[1:],
                                   batch_size)
            
            while self.times<=number_of_epochs+initial_epoch:
                for network,other_network,learning_method,index in zip((self.classifier,self.generator),
                                                                       (self.generator,self.classifier),
                                                                       (learning_method_class,learning_method_gen),
                                                                       (1,0)):
                    self.times+=0.5
                    self.tempo=time.time()

                    network.otimize(number_of_epochs,
                                    learning_method[0],
                                    weight_penalty[index],
                                    learning_method[1:],
                                    batch_size)

                    self.classifier.update_dataset(self.classifier.dataset)
                    self.generator.update_dataset(self.generator.dataset)

                    erros=(other_network.erros(other_network.validation_x,other_network.validation_y))

                    other_network.historico['cost'].append(erros[0])
                    other_network.historico['accuracy'].append(erros[1])

                    self.loading_symbols=['\\','|','/','-']
                    self.loading_index=0

                    self.report(self)

                    self.loading_index+=1
            
            print('Finished')
        except KeyboardInterrupt:
            print('Early Stop')
            
def create_otimizer_SGD(network,layers,params,givens):
    weight_penalty=theano.shared(0.0, borrow=True)
    learnrate=theano.shared(1.0, borrow=True)

    l2_norm_squared = sum([(layer.params[0]**2).sum() for layer in layers if len(layer.params)>1])

    regularization=0.5*weight_penalty*l2_norm_squared/network.n_subsets
    cost = sum([layer.cost_weight*layer.cost(network) for layer in layers if layer.cost_flag])+regularization

    fric=theano.shared(np.cast[theano.config.floatX](0))

    grads = tensor.grad(cost, params)
    grads_old = [theano.shared((param.get_value()*0).astype(theano.config.floatX)) for param in params]

    updates_sgd = [(param, (param-learnrate*(grad_old*fric+(1-fric)*grad)).astype(theano.config.floatX))
               for param, grad,grad_old in zip(params, grads,grads_old)]+\
               sum([layer.updates for layer in layers],[])+\
               [(grad_old,grad_old*fric+(1-fric)*grad)
               for grad_old,grad in zip(grads_old,grads)]
    
    update_parameter=lambda x_1,x_2,x_3: (weight_penalty.set_value(x_1),learnrate.set_value(x_2),fric.set_value(x_3))
    reset_hist=lambda: [i.set_value(i.get_value()*0) for i in grads_old]

    train_mb_SGD = theano.function(
        [network.x,network.y], cost, updates=updates_sgd, givens=givens)

    return (train_mb_SGD,update_parameter,reset_hist)

def create_otimizer_ADAM(network,layers,params,givens):
    weight_penalty=theano.shared(0.0, borrow=True)
    learnrate=theano.shared(1.0, borrow=True)

    l2_norm_squared = sum([(layer.params[0]**2).sum() for layer in layers if len(layer.params)>1])

    regularization=0.5*weight_penalty*l2_norm_squared/network.n_subsets
    cost = sum([layer.cost_weight*layer.cost(network) for layer in layers if layer.cost_flag])+regularization

    update_rate_1=theano.shared(np.cast[theano.config.floatX](1))
    update_rate_2=theano.shared(np.cast[theano.config.floatX](1))

    grads = tensor.grad(cost, params)
    
    adam_moment_1 = [theano.shared((param.get_value()*0).astype(theano.config.floatX)) for param in params]
    adam_moment_2 = [theano.shared((param.get_value()*0).astype(theano.config.floatX)) for param in params]
    adam_counter = theano.shared(np.int16(1))

    updates_adam = [(param, (param-learnrate*(((moment_1*update_rate_1+(1-update_rate_1)*grad)/(1-update_rate_1**adam_counter))/((((moment_2*update_rate_2+(1-update_rate_2)*(grad**2))**0.5)/(1-update_rate_2**adam_counter))+10**(-40)))).astype(theano.config.floatX))
               for param, grad,moment_1,moment_2 in zip(params, grads,adam_moment_1,adam_moment_2)]+\
               [(moment_1,(moment_1*update_rate_1+(1-update_rate_1)*grad).astype(theano.config.floatX))
               for moment_1,grad in zip(adam_moment_1,grads)]+\
               sum([layer.updates for layer in layers],[])+\
               [(moment_2,(moment_2*update_rate_2+(1-update_rate_2)*(grad**2)).astype(theano.config.floatX))
               for moment_2,grad in zip(adam_moment_2,grads)]+\
               [(adam_counter,adam_counter+1)]

    train_mb_ADAM = theano.function(
        [network.x,network.y], cost, updates=updates_adam, givens=givens)
    
    update_parameter=lambda x_1,x_2,x_3,x_4: (weight_penalty.set_value(x_1),learnrate.set_value(x_2),update_rate_1.set_value(x_3),update_rate_2.set_value(x_4))
    reset_hist=lambda: [i.set_value(i.get_value()*0) for i in adam_moment_1]+[i.set_value(i.get_value()*0) for i in adam_moment_2]+[adam_counter.set_value(np.int16(1))]

    return (train_mb_ADAM,update_parameter,reset_hist)