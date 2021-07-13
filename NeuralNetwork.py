import time
import tensorflow as tf
import dill
import config
from Otimizers import *

from Functions import report,report_GAN
import Layers as Layers

__version__='2.3.0'

print('TensorFlow version = {}'.format(tf.__version__))
print('Network version = {}'.format(__version__))
tf.config.optimizer.set_jit(config.XLA_opt)

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)

'''
Este código é o código usado para a criação rede neurais. Atualmente há suporte para 3 tipos de redes neurais:

Foward Network - é a rede neural padrão e mais básica, para cria-la use a classe FowardNetwork com o parâmetro output_layer=-1 (é o valor padrão).

Auto-encoder Network - é a rede neural que codifica os dados e decodifica, tentando minimizar alguma medida de erro na decodificação, para cria-la use a classe FowardNetwork com o parâmetro ouptut_layer sendo a camada que retorna os dados codificados (lembre-se de começar a contar a partir do 0).

Generative Adversaril Network (GAN) - esta classe cria duas redes neurais e as treina fazendo com que uma rede compita com a outra, para mais informações acesse o Google. Para criar esta rede use a classe GANNetwork.
'''
    
class NeuralNetwork:
    '''
    Classe base das redes neurais, as outras classes são subclasse desta e herdam todos os seus atributos e métodos.
    '''
    def __init__(self,dataset,layers,check_size,report_func,transformacoes=None):
        '''
        Este método inicializa a rede neural.
        
        dataset = função ou None
        layers = lista ou tupla
        check_size = int > 0
        report_func = função
        transformacoes = Qualquer coisa
        
        - dataset deve ser uma função que retorna os dados para treino e para teste, os dados devem ser do tipo tensorflow.Tensor ou algum formato compatível. Vale observar que a rede espera que os elementos dos dados estejam nas colunas da matriz. Além disso, esta função deve receber apenas um argumento, durante a execução da rede neural, esta função será chamada com o argumento 'train' para pegar os dados de treino e 'validation' para pegar os dados de teste. Por último, caso o usuário não queira que esta rede tenha as funções destinadas ao treino, basta passar None como dataset, daí só serão criadas as funções necessárias para executar a rede.
        
        - layers deve ser uma lista ou tupla de objetos da classe layer (para mais informações veja o arquivo 'Layers.py').
        
        - check_size é a quantidade máxima de inputs que a rede neural irá computar por vez, caso seja requisitado que ela calcule mais inputs do que o check_size ela dividirá os dados inseridos em pacotes com o tamanho do check_size e computará um pacote por vez. Informar um check_size muito grande pode ocasionar em erros de falta de memória (OOM).
        
        - report_func deve ser uma função que exibe um relatório do treino a cada rodada de treino, o único argumento que ela deve receber é a rede neural.
        
        - transformacoes é um argumento opcional que serve para armazenar instruções sobre o tratamento dos dados antes deles serem inseridos na rede neural.
        '''
        
        self.version='NN_'+__version__
        
        self.report=report_func
        self.check_size=check_size
        self.transformacoes=transformacoes
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
        #Este dicionário armazena informações sobre o histórico de treino. Apesar de ser viável, não é recomendado que se armazene o historico dos paramentros, pois isso pode consumir muita memória.
        self.historico={'cost':[],'accuracy':[],'params':[]}
        
        #Caso o dataset não seja None as funções destinadas a treino serão criadas, por isso, no bloco a seguir são criadas as variáveis auxiliares necessárias.
        if dataset is not None:
            #self.times registra a quantidade de etapas de treino já executas.
            self.times=0
            #self.len_subsets armazenas o tamanho dos pacotes de treino.
            self.len_subsets=1
            #Atualizando os dados de treino
            self.update_dataset(dataset)
        
    def update_dataset(self,dataset):
        '''
        Este método atualiza as informações sobre o banco de dados da rede neural
        
        inputs:
        dataset = função
        
        outputs:
        Nenhum
        
        Comentários:
        dataset deve ser uma função que retorna os dados para treino e para teste, os dados devem ser do tipo tensorflow.Tensor ou algum formato compatível. Vale observar que a rede espera que os elementos dos dados estejam nas colunas da matriz. Além disso, esta função deve receber apenas um argumento, durante a execução da rede neural, esta função será chamada com o argumento 'train' para pegar os dados de treino e 'validation' para pegar os dados de teste. Por último, caso o usuário não queira que esta rede tenha as funções destinadas ao treino, basta passar None como dataset, daí só serão criadas as funções necessárias para executar a rede.
        '''
        
        #Atualizando informações do banco de dados da rede e as variáveis associadas a ele.
        self.dataset=dataset

        self.len_dataset,self.len_validation=dataset('sizes')[:2]
        
        self.check_intern_subsets_len=self.len_dataset//self.check_size+1
        self.validation_subsets_len=self.len_validation//self.check_size+1
        
    def export_network(self,nome='default',diretorio='',tipo='layers'):
        '''
        Este método salva a rede neural no computador.
        
        inputs:
        nome = sting
        diretorio = string
        tipo= sting
        
        outputs:
        Nenhum
        
        Comentários:
        - nome é o nome do arquivo a ser salvo, caso o nome seja 'default', o nome será "versão da rede _ hora _ dia _ mês _ ano de criação"
        
        - diretorio é o local onde a rede deve ser salva, caso seja '', a rede será salva no diretório em que está sendo executado o código.
       
        - tipo é o tipo do arquivo, atualmente apenas o valor 'layers' é suportado, o que permite apenas que sejam salvos as camadas da rede neural.
        '''
        if nome=='default':
            date=time.localtime()
            nome=self.version+'_'+str(date.tm_hour)+'h_'+str(date.tm_mday)+'_'+str(date.tm_mon)+'_'+str(date.tm_year)+'.'+tipo
        arquivo=open(diretorio+nome,'wb')
        if tipo=='layers':
            dill.dump(self.layers, arquivo)
        else:
            arquivo.close()
            raise TypeError('Tipo de exportação inválida, valor informado diferente de "dnn" e "layers".')
        arquivo.close()
        
    def update_parametros(self,parametros):
        '''
        Este métodos atualiza os parâmetros das camadas da rede com os parâmetros inseridos como argumento.
        
        inputs:
        parametros = lista ou tupla
        
        outputs:
        Nenhum
        
        Comentário:
        parametros deve ser uma lista ou tupla contendo apenas tensorflow.Tensor ou objetos compatíveis, sendo que estes devem estar ordenados na mesma ordem que os parâmetros de cada camada, na ordem das camadas, por exemplo, se a rede neural tem apenas duas camadas do tipo FC, então a lista deve vir na seguinte ordem:
            [ pesos para a primeira camada , viés para a primeira camada , pesos para a segunda camada , viés para a segunda camada]
        Para mais informações sobre a ordem dos parâmetros em cada tipo de camada, consulte o arquivo 'Layers.py'.
        '''
        for i,j in zip(self.params,parametros):
            i.assign(j)
        
class FowardNetwork(NeuralNetwork):
    '''
    Esta classe é destinada a redes neurais cujo treino é direto, ou seja, redes neurais padrões. Além disso, há suporte para auto-encoders, pois o treino deste tipo de rede também é direto.
    '''
    def __init__(self,dataset,layers,check_size,output_layer=-1,report_func=report,transformacoes=None,sparse_flag=False):
        '''
        Este método inicializa a rede neural.
        
        dataset = função ou None
        layers = lista ou tupla
        check_size = int > 0
        output_layer = int
        report_func = função
        transformacoes = Qualquer coisa
        
        - dataset deve ser uma função que retorna os dados para treino e para teste (o retorno da função deve ser uma lista cuja primeira coordenada são os inputs da rede neural e a segunda são os labels), os dados devem ser do tipo tensorflow.Tensor ou algum formato compatível. Vale observar que a rede espera que os elementos dos dados estejam nas colunas da matriz. Além disso, esta função deve receber apenas um argumento, durante a execução da rede neural, esta função será chamada com o argumento 'train' para pegar os dados de treino e 'validation' para pegar os dados de teste. Por último, caso o usuário não queira que esta rede tenha as funções destinadas ao treino, basta passar None como dataset, daí só serão criadas as funções necessárias para executar a rede.
        
        - layers deve ser uma lista ou tupla de objetos da classe layer (para mais informações veja o arquivo 'Layers.py').
        
        - check_size é a quantidade máxima de inputs que a rede neural irá computar por vez, caso seja requisitado que ela cálcule mais inputs do que o check_size ela dividirá os dados inseridos em pacotes com o tamanho do check_size e computará um pacote por vez. Informar um check_size muito grande pode ocasionar em erros de falta de memória (OOM).
        
        - output_layer é o índice da camada que possuí as saídas da rede, quando este valor é diferente de -1, cria-se uma função que executa a rede neural apenas até a camada que possuí a saída da rede e outra função que executa a rede desta camada em diante.
        
        - report_func deve ser uma função que exibe um relatório do treino a cada rodada de treino, o único argumento que ela deve receber é a rede neural.
        
        - transformacoes é um argumento opcional que serve para armazenar instruções sobre o tratamento dos dados antes deles serem inseridos na rede neural.
        '''
        self.version='FN_'+__version__
        
        NeuralNetwork.__init__(self,dataset,layers,check_size,report_func,transformacoes)
        
        self.output_layer=output_layer
        self.report=report_func
        self.change_freq=tf.Variable(1000,dtype='int32')
        self.sparse_flag=sparse_flag
        
        self.otimizers={}
        self.best_accuracy=None

    @tf.function(experimental_compile=config.XLA_func)
    def classificar(self,entrada,previous_value=None):
        processamento=entrada
        for layer in self.layers:
            processamento=layer.execute(processamento,False)
        return processamento

    def erros(self,x,y):
        expected_value=self.classificar(x)
        if self.sparse_flag:
            return self.layers[-1].accuracy_check(expected_value,tf.sparse.to_dense(y))
        else:
            return self.layers[-1].accuracy_check(expected_value,y)

    #Caso a saída da rede não seja na última camada, cria-se duas funções que executam pedaços separados da rede, a função self.encode executa a rede do inicio até a camada de saída e a self.decode executa a rede da camada posterior à saída até o final.
    @tf.function(experimental_compile=config.XLA_func)
    def encode(self,entrada):
        #assert self.output_layer>=0 and isinstance(self.output_layer,int),'Esta rede não é um auto-encoder. Caso você queira um auto-encoder, defina output_layer como um inteiro não-negativo.'
        processamento=entrada
        for layer in self.layers[:self.output_layer+1]:
            processamento=layer.execute(processamento,False)
        return processamento

    @tf.function(experimental_compile=config.XLA_func)
    def decode(self,entrada):
        assert self.output_layer>=0 and isinstance(self.output_layer,int),'Esta rede não é um auto-encoder. Caso você queira um auto-encoder, defina output_layer como um inteiro não-negativo.'
        processamento=entrada
        for layer in self.layers[self.output_layer+1:]:
            processamento=layer.execute(processamento,False)
        return processamento

    def foward_prop(self,x,y):
        #calculando regularização.
        cost = 0
        #Adicionando o custo de cada camada ao custo total. Cada camada tem uma flag que informa se o custo da camada deve ser somado ao custo total.
        current_value=x
        for layer in self.layers:
            current_value=layer.execute(current_value,True)
            if layer.cost_flag:
                cost=cost+layer.cost_weight*layer.cost(current_value,y)
        return [cost]
        
    def otimize(self,number_of_epochs,no_better_interval,weight_penalty,learning_method,batch_size,do_something=lambda x:x):
        '''
        Esta função otimiza a rede neural.
        
        inputs:
        number_of_epochs = int >= 0
        no_better_interval = int, entre 0 e 25
        weight_penalty = float
        learning_method = lista ou tupla
        batch_size = int > 0 
        update_data= bool
        
        outputs:
        Nenhum
        
        Comentários:
        number_of_epochs é o número máximo de rodadas de treino a serem executadas
        
        no_better_interval é o tamanho dos intervalos a serem avaliados para finalizar o treino. Seja no_better_interval = n, então a cada n rodadas de treino verifica-se se a média da acurácia nas últimas n rodadas é menor do que a das n rodadas anteriores, se sim, continua-se a treinar, do contrário, o treino é interrompido. A rede neural mantém registro da acurácia apenas dos 50 últimos treinos, logo no_better_interval tem de ser menor ou igual a 25.
        
        weight_penalty é o peso a ser dado a regularização dos parâmetros das camadas.
        
        learning_method é uma lista ou tupla cujo primeiro elemento é o método de otimização (SGD, ADAM ou RMSProp) e os outros elementos são os parâmetros do método escolhido, para mais informações sobre quais são os parâmetros de cada método de otimização, consulte o arquivo 'Otimizers.py'.
        
        batch_size é o tamanho dos pacotes de treino, caso a quantidade de elementos nos dados de treino não seja múltiplo do batch_size, os elementos que estiverem sobrando no final não serão usados no treino.
        
        update_data é uma flag que informa se os dados devem ser atualizados no início do treino ou não. Isto é útil ao usar dados que devem ser re-gerados a cada treino.
        '''
        self.otimization=learning_method
        self.weight_penalty=weight_penalty
        
        best_param=self.params
        
        #Caso o otimizador ainda não tenha sido criado, cria-se o otimizador e o armazena no dicionário de otimizadores.
        if learning_method[0] not in self.otimizers.keys():
            exec('self.otimizers["{0}"]=create_otimizer_{0}(self)'.format(learning_method[0]))
        
        #Preparando variáveis para treino.
        self.otimization_func,self.update_params,self.reset_hist=self.otimizers[learning_method[0]]
            
        self.reset_hist()
        self.update_params(weight_penalty,*learning_method[1:])
        
        no_better_count=0
        initial_epoch=self.times
        
        self.len_subsets=batch_size
        self.n_subsets=self.len_dataset//self.len_subsets+1
        
        
        self.loading_symbols=['\\','|','/','-']
        self.loading_index=0
        
        #O usuário tem a opção de interromper o treino antecipadamente, para isto, basta gerar o KeyboardInterrupt (basta apertar Ctrl + C no Idle do Python).
        try:
            indices=tf.range(self.len_dataset)
            while self.times-initial_epoch<number_of_epochs:
                self.times+=1
                #self.tempo armazena o tempo no início da execução, isto é útil para medir o tempo gasto para executar uma rodada de treino.
                self.tempo=time.time()                
                self.initial_time=time.time()
                #Treinando a rede para cada pacote de dados.
                indices=tf.random.shuffle(indices)
                iterantion=[[],[tf.constant(0)]]
                grad_range=[0,0]
                last_time=time.time()
                current_time=time.time()
                for indice in range(self.n_subsets):
                    assert tf.math.greater(39,self.grad_scale),'A escala do gradiente está muito alta (maior que 10^39), verifique se não há um erro no código. A princípio, a variável que controla a escala do gradiente só aumenta se a derivada de alguma variável for infinita, porém, se houver algum erro no código ou nos dados, de modo que a função de custo retorne NaN, então a escala do gradiente irá aumentar indefinidamente, nestes casos é necessário revisar o código e o conjunto de dados usado.'
                    #Amostrando indices dos dados
                    slices=indices[indice*self.len_subsets:(indice+1)*self.len_subsets]
                    iteration=self.otimization_func(*self.dataset('train',indices=slices))
                    grad_range=iteration[1:3]
                    last_time=current_time
                    current_time=time.time()
                    
                    do_something(self)
                    
                    print('Batch: '+str(int(10000*indice/self.n_subsets)/100)+'% - Time spent: '+str(int((current_time-last_time)*1000)/1000)+' - Último erro: '+str(iteration[4].numpy())+\
                          ' - Grad range : {0}~{1}'.format(grad_range[1],grad_range[0])+'              ',end='\r')
                print('Batch: '+str(int(10000*indice/self.n_subsets)/100)+'% - Time spent: '+str(int((current_time-last_time)*1000)/1000)+' - Último erro: '+str(iteration[4].numpy())+\
                      ' - Grad range : {0}~{1}'.format(grad_range[1],grad_range[1])+'              ',end='\r')
                self.end_time=time.time()
                #Avaliando a função de custo e a função de acurácia depois da rodada atual de treino.
                
                erros_list=tf.zeros([0,2],dtype=config.float_type)
                indices=tf.range(self.len_validation)
                indices=tf.random.shuffle(indices)
                
                for indice in range(self.validation_subsets_len):
                    slices=indices[indice*self.check_size:(indice+1)*self.check_size]
                    x,y=self.dataset('validation',slices)
                    instant_erro=self.erros(x,y)
                    erros_list=tf.concat([erros_list,[instant_erro]],axis=0)
                erros=tf.math.reduce_mean(erros_list,axis=0)
                
                #Registrando os valores obtidos.
                if self.best_accuracy is None:
                    self.best_accuracy=erros[1]
                    self.best_cost=erros[0]
                    self.best_param=[param.numpy() for param in self.params]
                elif erros[1]>self.best_accuracy:
                    self.best_accuracy=erros[1]
                    self.best_cost=erros[0]
                    #self.export_network('backup\\NN_{}.layers'.format(erros[1]))
                    self.best_param=[param.numpy() for param in self.params]
                    no_better_count=0
                else:
                    no_better_count+=1
                self.historico['cost']=self.historico['cost'][-5:]+[erros[0]]
                self.historico['accuracy']=self.historico['accuracy'][-5:]+[erros[1]]
                
                #Gerando report.
                self.report(self)
                
                self.loading_index+=1
                
                #Verificando o critério de paragem antecipada.
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
    def __init__(self,dataset_class,dataset_gen,layers_gen,layers_class,check_size,tolerance=0,report_func=report_GAN,transformacoes=None):        
        self.version='GAN_'+__version__
        
        self.tolerance=tolerance

        layers=layers_gen+layers_class
        NeuralNetwork.__init__(self,dataset_class,layers,check_size,report_func,transformacoes)

        self.params_gen = [param for layer in layers_gen for param in layer.params]
        self.params_class = [param for layer in layers_class for param in layer.params]

        self.len_dataset,self.len_validation=dataset_gen('sizes')
        self.dataset_gen=dataset_gen
        
        layers.append(Layers.integration(layers=layers_class,
                                         funcao_erro=layers_gen[-1].funcao_erro,
                                         funcao_acuracia=layers_gen[-1].funcao_acuracia))
        
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
        indices=tf.range(self.len_dataset)
        for indice in range(self.len_dataset//self.check_size+1):
            chuncks.append(self.generator.encode(self.dataset_gen('train',indices[indice*self.check_size:(indice+1)*self.check_size])[0]))
        self.classifier_train_x=tf.concat(chuncks,axis=0)

        self.classifier_train_y=tf.concat([self.pre_data_y,
                                                  tf.zeros(self.pre_data_y.shape,
                                                           dtype=config.float_type)+self.tolerance],
                                                 axis=0)
        
        chuncks=[self.pre_test_x]
        indices=tf.range(self.len_validation)
        for indice in range(self.len_validation//self.check_size+1):
            chuncks.append(self.generator.encode(self.dataset_gen('validation',indices[indice*self.check_size:(indice+1)*self.check_size])[0]))
        self.classifier_val_x=tf.concat(chuncks,axis=0)

        self.classifier_val_y=tf.concat([self.pre_test_y,
                                                tf.zeros(self.pre_test_y.shape,
                                                         dtype=config.float_type)+self.tolerance],
                                               axis=0)
        
    def dataset_class(self,label,indices=None):
        if label=='sizes':
            return self.len_dataset*2,self.len_validation*2
        elif label=='train':
            if indices is None:
                indices=tf.range(self.len_dataset*2)
            data_x=tf.gather(self.classifier_train_x,indices,axis=0)
            data_y=tf.gather(self.classifier_train_y,indices,axis=0)
            return data_x,data_y
        elif label=='validation':
            if indices is None:
                indices=tf.range(self.len_validation*2)
            data_x=tf.gather(self.classifier_val_x,indices,axis=0)
            data_y=tf.gather(self.classifier_val_y,indices,axis=0)
            return data_x,data_y
        
    def otimize_network(self,network):
        indices=tf.range(network.len_dataset)
        for time in range(network.number_of_epochs):
                indices=tf.random.shuffle(indices)
                iterantion=[[],[tf.constant(0)]]
                for indice in range(network.n_subsets):
                    slices=indices[indice*self.batch_size:(indice+1)*self.batch_size]
                    iteration=network.otimization_func(*network.dataset('train',indices=slices))
                    print('Progresso: '+str(int(10000*((time*network.n_subsets+indice)/(network.number_of_epochs*network.n_subsets)))/100)+'%         ',end='\r')
        
    def otimize(self,number_of_epochs,weight_penalty,learning_method_gen,learning_method_class,batch_size):
        initial_epoch=self.times
        
        self.weight_penalty=weight_penalty
        self.classifier.otimization,self.generator.otimization=learning_method_class,learning_method_gen
        self.batch_size=batch_size
        
        self.classifier.weight_penalty=weight_penalty[0]
        self.generator.weight_penalty=weight_penalty[1]
        
        for indice,network in enumerate([self.classifier,self.generator]):
            network.number_of_epochs=number_of_epochs[indice+1]
            network.n_subsets=network.len_dataset//batch_size+1
            if network.otimization[0] not in network.otimizers.keys():
                exec('network.otimizers["{0}"]=create_otimizer_{0}(network)'.format(network.otimization[0]))
        
            network.otimization_func,network.update_params,network.reset_hist=network.otimizers[network.otimization[0]]
            network.reset_hist()
            network.update_params(weight_penalty[indice],*network.otimization[1:])
            
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
                        erros_list=tf.zeros([0,2],dtype=config.float_type)
                        indices=tf.range(other_network.len_validation)
                        indices=tf.random.shuffle(indices)
                        for indice in range(other_network.validation_subsets_len):
                            x,y=other_network.dataset('validation',indices[indice*self.check_size:(indice+1)*self.check_size])
                            instant_erro=other_network.erros(x,y)
                            erros_list=tf.concat([erros_list,[[i*x.shape[0] for i in instant_erro]]],axis=0)
                        erros=tf.math.reduce_sum(erros_list,axis=0)/other_network.len_validation

                        other_network.historico['cost'].append(erros[0])
                        other_network.historico['accuracy'].append(erros[1])

                    self.loading_symbols=['\\','|','/','-']
                    self.loading_index=0

                    self.report(self)

                    self.loading_index+=1
            print('Finished')
        except KeyboardInterrupt:
            print('Forced Stop')