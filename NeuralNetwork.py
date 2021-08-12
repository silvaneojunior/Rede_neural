import time
import tensorflow as tf
import dill
import config
import Otimizers

from Functions import report,report_GAN
import Layers as Layers

__version__='2.4.1'

print('TensorFlow version = {}'.format(tf.__version__))
print('Network version = {}'.format(__version__))
tf.config.optimizer.set_jit(config.XLA_opt)

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)
    
class NeuralNetwork:
    def __init__(self,dataset,layers,check_size,report_func,transformations=None):
        self.version='NN_'+__version__
        
        self.report=report_func
        self.check_size=check_size
        self.transformations=transformations
        self.recurrent_flag=False
        self.layers=layers
        #Armazenando os parâmetros de todas as camadas da rede
        self.params = []
        self.params_ref=[]
        for layer in layers:
            for param in layer.params:
                if param.ref() not in self.params_ref:
                    self.params.append(param)
                    self.params_ref.append(param.ref())
        self.historico={'cost':[],'accuracy':[],'params':[]}
        if dataset is not None:
            self.times=0
            self.len_subsets=1
            self.update_dataset(dataset)
        
    def update_dataset(self,dataset):
        self.dataset=dataset

        self.len_dataset,self.len_validation=dataset('sizes')[:2]
        
        self.check_intern_subsets_len=self.len_dataset//self.check_size+1
        self.validation_subsets_len=self.len_validation//self.check_size+1
        
    def export_network(self,name='default',directory='',layer_type='layers'):
        if name=='default':
            date=time.localtime()
            name=self.version+'_'+str(date.tm_hour)+'h_'+str(date.tm_mday)+'_'+str(date.tm_mon)+'_'+str(date.tm_year)+'.'+layer_type
        with open(directory+name,'wb') as file:
            if layer_type=='layers':
                dill.dump(self.layers, file)
            else:
                raise TypeError('Invalid exportation type.')
        
    def update_parameters(self,parameters):
        for i,j in zip(self.params,parameters):
            i.assign(j)
        
class FowardNetwork(NeuralNetwork):
    def __init__(self,dataset,layers,check_size,output_layer=-1,report_func=report,transformations=None,sparse_flag=False):
        self.version='FN_'+__version__
        
        NeuralNetwork.__init__(self,dataset,layers,check_size,report_func,transformations)
        
        self.output_layer=output_layer
        self.report=report_func
        self.change_freq=tf.Variable(1000,dtype='int32')
        self.sparse_flag=sparse_flag
        
        self.otimizers={}
        self.best_accuracy=None

    @tf.function(experimental_compile=config.XLA_func)
    def classify(self,inputs,return_mask=False):
        activation=inputs
        mask=None
        for layer in self.layers:
            activation,mask=layer.predict(activation,mask),layer.compute_mask(activation,mask)
        if return_mask:
            return activation,mask
        else:
            return activation

    def accuracy_eval(self,x,y):
        expected_value,mask=self.classify(x,return_mask=True)
        if self.sparse_flag:
            return self.layers[-1].accuracy_check(expected_value,tf.sparse.to_dense(y),mask)
        else:
            return self.layers[-1].accuracy_check(expected_value,y,mask)

    @tf.function(experimental_compile=config.XLA_func)
    def encode(self,inputs,return_mask=False):
        activation=inputs
        mask=None
        for layer in self.layers[:self.output_layer+1]:
            activation,mask=layer.execute(activation,mask),layer.compute_mask(activation,mask)
        if return_mask:
            return activation,mask
        else:
            return activation

    @tf.function(experimental_compile=config.XLA_func)
    def decode(self,inputs,return_mask=False):
        assert self.output_layer>=0 and isinstance(self.output_layer,int),'Esta rede não é um auto-encoder. Caso você queira um auto-encoder, defina output_layer como um inteiro não-negativo.'
        activation=inputs
        mask=None
        for layer in self.layers[self.output_layer+1:]:
            activation,mask=layer.execute(activation,mask),layer.compute_mask(activation,mask)
        if return_mask:
            return activation,mask
        else:
            return activation

    def foward_prop(self,x,y):
        cost = 0
        
        current_value=x
        mask=None
        for layer in self.layers:
            current_value,mask=layer.execute(current_value,mask),layer.compute_mask(current_value,mask)
            if layer.cost_flag:
                cost=cost+layer.cost_weight*layer.cost(current_value,y,mask)
        return cost
        
    def otimize(self,number_of_epochs,no_better_interval,weight_penalty,learning_method,batch_size):
        self.otimization=learning_method
        self.weight_penalty=weight_penalty
        
        self.best_param=self.params
        
        if learning_method[0] not in self.otimizers.keys():
            exec('self.otimizers["{0}"]=Otimizers.create_otimizer_{0}(self)'.format(learning_method[0]))
        
        self.otimization_func,self.update_params,self.reset_hist=self.otimizers[learning_method[0]]
        self.reset_hist()
        self.update_params(weight_penalty,*learning_method[1:])
        no_better_count=0
        initial_epoch=self.times
        self.len_subsets=batch_size
        self.n_subsets=self.len_dataset//self.len_subsets+1
        
        self.loading_symbols=['\\','|','/','-']
        self.loading_index=0
        
        try:
            while self.times-initial_epoch<number_of_epochs:
                self.times+=1
                self.tempo=time.time()                
                self.initial_time=time.time()
                index_range=tf.random.shuffle(tf.range(self.len_dataset))
                cost=0
                grad_range=[0,0]
                last_time=time.time()
                current_time=time.time()
                for index in range(self.n_subsets):
                    assert tf.math.greater(39,self.grad_scale),'Gradient scale is too high.'

                    slices=index_range[index*self.len_subsets:(index+1)*self.len_subsets]
                    update,cost,grad_range=self.otimization_func(*self.dataset('train',index_range=slices))
                    last_time=current_time
                    current_time=time.time()
                    
                    print('Progression: '+\
                          str(int(10000*index/self.n_subsets)/100)+\
                          '% - Time spent: '+\
                          str(int((current_time-last_time)*1000)/1000)+\
                          ' - Last error: '+\
                          str(cost.numpy())+\
                          ' - Grad range : {0}~{1}'.format(grad_range[0],grad_range[1])+\
                          '                                                              ',
                          end='\r')
                print('Progression: '+\
                      str(int(10000*index/self.n_subsets)/100)+\
                      '% - Time spent: '+
                      str(int((current_time-last_time)*1000)/1000)+\
                      ' - Last error: '+str(cost.numpy())+\
                      ' - Grad range : {0}~{1}'.format(grad_range[0],grad_range[1])+\
                      '                                                                  ',
                      end='\r')
                self.end_time=time.time()

                error_list=[]
                index_range=tf.random.shuffle(tf.range(self.len_validation))
                
                for index in range(self.validation_subsets_len):
                    slices=index_range[index*self.check_size:(index+1)*self.check_size]
                    x,y=self.dataset('validation',index_range=slices)
                    instant_erro=self.accuracy_eval(x,y)
                    error_list.append([instant_erro])
                errors=tf.squeeze(tf.math.reduce_mean(error_list,axis=0))

                if self.best_accuracy is None:
                    self.best_accuracy=errors[1]
                    self.best_cost=errors[0]
                    self.best_param=[param.numpy() for param in self.params]
                elif errors[1]>self.best_accuracy:
                    self.best_accuracy=errors[1]
                    self.best_cost=errors[0]
                    self.best_param=[param.numpy() for param in self.params]
                    no_better_count=0
                else:
                    no_better_count+=1
                self.historico['cost']=self.historico['cost'][-5:]+[errors[0]]
                self.historico['accuracy']=self.historico['accuracy'][-5:]+[errors[1]]

                self.report(self)
                
                self.loading_index+=1

                if no_better_count>no_better_interval:
                    self.update_parametros(self.best_param)
                    print('Early Stop')
                    break
            if self.times-initial_epoch>=number_of_epochs:
                self.update_parametros(self.best_param)
                print('Finished')
        except KeyboardInterrupt:
            self.update_parametros(self.best_param)
            print('Forced Stop')
            
class GANetwork(NeuralNetwork):
    def __init__(self,dataset_class,dataset_gen,layers_gen,layers_class,check_size,tolerance=0,report_func=report_GAN,transformations=None):        
        self.version='GAN_'+__version__
        
        self.tolerance=tolerance

        layers=layers_gen+layers_class
        NeuralNetwork.__init__(self,dataset_class,layers,check_size,report_func,transformations)

        self.params_gen = [param for layer in layers_gen for param in layer.params]
        self.params_class = [param for layer in layers_class for param in layer.params]

        self.len_dataset,self.len_validation=dataset_gen('sizes')
        self.dataset_gen=dataset_gen
        
        layers.append(Layers.integration(layers=layers_class,
                                         error_function=layers_gen[-1].funcao_erro,
                                         accur_function=layers_gen[-1].funcao_acuracia))
        
        layers_gen[-1].cost_flag=False

        self.generator=FowardNetwork(dataset=self.dataset_gen,
                                    layers=layers_gen+[layers[-1]],
                                    check_size=check_size,
                                    report_func=lambda x: None,
                                    output_layer=len(layers_gen)-1)
        
        self.pre_dataset_class=dataset_class

        self.classifier=FowardNetwork(dataset=self.dataset_class,
                                      layers=layers_class,
                                      check_size=check_size,
                                      report_func=lambda x: None)
        
        self.gerar=self.generator.encode
        self.classificar=self.classifier.classificar
        
        with tf.device('/device:CPU:0'):
            self.pre_data_x,self.pre_data_y=dataset_class('train')
            self.pre_test_x,self.pre_test_y=dataset_class('validation')

        self.classifier_train_x=tf.zeros([self.len_dataset*2,self.pre_data_x.shape[1]],dtype=self.pre_data_x.dtype)
        self.classifier_train_y=tf.zeros([self.len_dataset*2,self.pre_data_y.shape[1]],dtype=self.pre_data_y.dtype)
        self.classifier_val_x=tf.zeros([self.len_validation*2,self.pre_test_x.shape[1]],dtype=self.pre_test_x.dtype)
        self.classifier_val_y=tf.zeros([self.len_validation*2,self.pre_test_y.shape[1]],dtype=self.pre_test_y.dtype)
        
        self.renew_dataset_class()
    
    def renew_dataset_class(self):
        chuncks=[self.pre_data_x]
        index_range=tf.range(self.len_dataset)
        for index in range(self.len_dataset//self.check_size+1):
            chuncks.append(self.generator.encode(self.dataset_gen('train',index_range[index*self.check_size:(index+1)*self.check_size])[0]))
        self.classifier_train_x=tf.concat(chuncks,axis=0)

        self.classifier_train_y=tf.concat([self.pre_data_y,
                                                  tf.zeros(self.pre_data_y.shape,
                                                           dtype=config.float_type)+self.tolerance],
                                                 axis=0)
        
        chuncks=[self.pre_test_x]
        index_range=tf.range(self.len_validation)
        for index in range(self.len_validation//self.check_size+1):
            chuncks.append(self.generator.encode(self.dataset_gen('validation',index_range[index*self.check_size:(index+1)*self.check_size])[0]))
        self.classifier_val_x=tf.concat(chuncks,axis=0)

        self.classifier_val_y=tf.concat([self.pre_test_y,
                                                tf.zeros(self.pre_test_y.shape,
                                                         dtype=config.float_type)+self.tolerance],
                                               axis=0)
        
    def dataset_class(self,label,index_range=None):
        if label=='sizes':
            return self.len_dataset*2,self.len_validation*2
        elif label=='train':
            if index_range is None:
                index_range=tf.range(self.len_dataset*2)
            data_x=tf.gather(self.classifier_train_x,index_range,axis=0)
            data_y=tf.gather(self.classifier_train_y,index_range,axis=0)
            return data_x,data_y
        elif label=='validation':
            if index_range is None:
                index_range=tf.range(self.len_validation*2)
            data_x=tf.gather(self.classifier_val_x,index_range,axis=0)
            data_y=tf.gather(self.classifier_val_y,index_range,axis=0)
            return data_x,data_y
        
    def otimize_network(self,network):
        index_range=tf.range(network.len_dataset)
        for step in range(network.number_of_epochs):
                index_range=tf.random.shuffle(index_range)
                for index in range(network.n_subsets):
                    slices=index_range[index*self.batch_size:(index+1)*self.batch_size]
                    network.otimization_func(*network.dataset('train',index_range=slices))
                    print('Progression: '+str(int(10000*((step*network.n_subsets+index)/(network.number_of_epochs*network.n_subsets)))/100)+'%                         ',end='\r')
        
    def otimize(self,number_of_epochs,weight_penalty,learning_method_gen,learning_method_class,batch_size):
        initial_epoch=self.times
        
        self.weight_penalty=weight_penalty
        self.classifier.otimization,self.generator.otimization=learning_method_class,learning_method_gen
        self.batch_size=batch_size
        
        self.classifier.weight_penalty=weight_penalty[0]
        self.generator.weight_penalty=weight_penalty[1]
        
        for index,network in enumerate([self.classifier,self.generator]):
            network.number_of_epochs=number_of_epochs[index+1]
            network.n_subsets=network.len_dataset//batch_size+1
            if network.otimization[0] not in network.otimizers.keys():
                exec('network.otimizers["{0}"]=create_otimizer_{0}(network)'.format(network.otimization[0]))
        
            network.otimization_func,network.update_params,network.reset_hist=network.otimizers[network.otimization[0]]
            network.reset_hist()
            network.update_params(weight_penalty[index],*network.otimization[1:])
            
        try:           
            while self.times<=(number_of_epochs[0])+initial_epoch:
                for network,name in zip((self.classifier,self.generator),('Classificador','Gerador')):
                    
                    self.times+=0.5
                    self.tempo=time.time()
                    self.last_name=name

                    self.otimize_network(network)
                    if name=='Gerador':
                        self.renew_dataset_class()
                    
                    for other_network in (self.classifier,self.generator):
                        error_list=[]
                        indices=tf.range(other_network.len_validation)
                        indices=tf.random.shuffle(indices)
                        for indice in range(other_network.validation_subsets_len):
                            x,y=other_network.dataset('validation',indices[indice*self.check_size:(indice+1)*self.check_size])
                            instant_erro=other_network.erros(x,y)
                            error_list.append([[i*x.shape[0] for i in instant_erro]])
                        errors=tf.math.reduce_sum(error_list,axis=0)/other_network.len_validation

                        other_network.historico['cost'].append(errors[0])
                        other_network.historico['accuracy'].append(errors[1])

                    self.loading_symbols=['\\','|','/','-']
                    self.loading_index=0

                    self.report(self)

                    self.loading_index+=1
            print('Finished')
        except KeyboardInterrupt:
            print('Forced Stop')