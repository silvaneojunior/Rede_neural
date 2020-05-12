import time
import tensorflow as tf
import numpy as np
import dill
import config
from Otimizers import *

from Functions import report,report_GAN
import Layers as Layers

__version__='V2.0.0'

print('TensorFlow version = {}'.format(tf.__version__))
print('Network version = {}'.format(__version__))

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
        
        self.version='NN_V2.0.0'
        
        self.report=report_func
        self.check_size=check_size
        self.transformacoes=transformacoes
        self.layers=layers
        #Armazenando os parâmetros de todas as camadas da rede
        self.params = [param for layer in layers for param in layer.params]
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
        
        #Definindo qual dispositivo deve ser usado no escopo do with. Ao explicitar que este bloco de código deve ser executado na CPU, evitamos que estes dados sejam armazenados na memória da GPU, economizando preciosos MB de memória.
        with tf.device('/CPU:0'):
            #Atualizando informações do banco de dados da rede e as variáveis associadas a ele.
            self.dataset=dataset
            self.dataset_x,self.dataset_y=dataset('train')
            self.validation_x,self.validation_y=dataset('validation')

            self.len_dataset=self.dataset_y.shape[1]
            self.len_validation=self.validation_y.shape[1]

            self.n_subsets=self.len_dataset//self.len_subsets
            self.check_intern_subsets_len=self.len_dataset//self.check_size
            self.validation_subsets_len=self.len_validation//self.check_size
        
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
        Este métodos atualiza os parâmetros das camadas da rede com os parâmetrosinseridos como argumento.
        
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
    def __init__(self,dataset,layers,check_size,output_layer=-1,report_func=report,transformacoes=None):
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
        self.version='FN_V2.0.0'
                
        NeuralNetwork.__init__(self,dataset,layers,check_size,report_func,transformacoes)
        
        self.output_layer=output_layer
        self.report=report_func
        
        #A função divide_data recebe uma função (pre_encode, pre_decode ou pre_classificar) e retorna uma função que divide os dados que recebe um pacotes do tamanho do check_size e depois executa a função original em cada pacote, no final a função concatena os pacotes e os retorna.  
        def divide_data(func):
            def divided_func(entrada):
                n=entrada.shape[1]//self.check_size
                if n>0:
                    saida=tf.concat([func(entrada[:,indice*self.check_size:(indice+1)*self.check_size]) for indice in range(n)],axis=1)
                    if entrada.shape[1]%self.check_size>0:
                        saida=tf.concat([saida,func(entrada[:,(n)*self.check_size:])],axis=1) if n>0 else func(entrada)
                else:
                    saida=func(entrada)
                return saida
            return divided_func
        
        #Criando a função preliminar que executa a rede do início até o final, mais a frente esta função será usada para criar a função de classificação
        #O decorator @tf.function serve para sinalizar ao tensorflow que esta função deve ter a execução otimizada, consulte a documentação do tensorflow para mais informações a respeito.
        @tf.function
        def pre_classificar(entrada):
            processamento=entrada
            for layer in self.layers:
                processamento=layer.execute(processamento,False)
            return processamento
        
        #Criando função de classificação
        self.classificar=divide_data(pre_classificar)
        
        #Caso a saída da rede não seja na última camada, cria-se duas funções que executam pedaços separados da rede, a função self.encode executa a rede do inicio até a camada de saída e a self.decode executa a rede da camada posterior à saída até o final.
        if output_layer!=-1:
            
            @tf.function
            def pre_encode(entrada):
                processamento=entrada
                for layer in self.layers[:self.output_layer+1]:
                    processamento=layer.execute(processamento,False)
                return processamento
                
            @tf.function
            def pre_decode(entrada):
                processamento=entrada
                for layer in self.layers[self.output_layer+1:]:
                    processamento=layer.execute(processamento,False)
                return processamento
            
            self.encode=divide_data(pre_encode)
            self.decode=divide_data(pre_decode)
            
        #Este dicionário armazena os otimizadores já criados. Para mais informações sobre os otimizadores consulte o arquivo "Otimizers.py"   
        self.otimizers={}
        
        #Criando a função que calcula o erro cometido na classificação de um certo dado. Esta função recebe um dado e o label desse dado.
        @tf.function
        def pre_erros(x,y):
            expected_value=self.classificar(x)
            return layers[-1].accuracy_check(expected_value,y)

        self.erros=pre_erros
        
    def otimize(self,number_of_epochs,no_better_interval,weight_penalty,learning_method,batch_size,update_data=False):
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
        
        if update_data:
            self.update_dataset(dataset)
        
        #Caso o otimizador ainda não tenha sido criado, cria-se o otimizador e o armazena no dicionário de otimizadores.
        if learning_method[0] not in self.otimizers.keys():
            exec('self.otimizers["{0}"]=create_otimizer_{0}(self)'.format(learning_method[0]))
        
        #Preparando variáveis para treino.
        self.otimization_func,self.update_params,self.reset_hist=self.otimizers[learning_method[0]]
            
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
        
        #O usuário tem a opção de interromper o treino antecipadamente, para isto, basta gerar o KeyboardInterrupt (basta apertar Ctrl + C no Idle do Python).
        try:
            while self.times-initial_epoch<number_of_epochs:
                self.times+=1
                #self.tempo armazena o tempo no início da execução, isto é útil para medir o tempo gasto para executar uma rodada de treino.
                self.tempo=time.time()
                
                #Gerando índices do banco de dados.
                indices=np.asarray(range(self.len_dataset))
                
                #Embaralhando índices.
                np.random.shuffle(indices)
                
                #Reordenando o banco de dados na ordem dos índices embaralhados.
                self.dataset_x,self.dataset_y=tf.gather(self.dataset_x,indices,axis=1),tf.gather(self.dataset_y,indices,axis=1)
                
                #Treinando a rede para cada pacote de dados.
                for indice in range(self.n_subsets):
                    iteration=self.otimization_func(self.dataset_x[:,int(indice*self.len_subsets):int((indice+1)*self.len_subsets)],
                                            self.dataset_y[:,int(indice*self.len_subsets):int((indice+1)*self.len_subsets)])
                
                #Avaliando a função de custo e a função de acurácia depois da rodada atual de treino.
                erros=self.erros(self.validation_x,self.validation_y)
                
                #Registrando os valores obtidos.
                self.historico['cost'].append(erros[0])
                self.historico['accuracy'].append(erros[1])
                
                #Caso mais de 50 valores tenham sido armazenados, deleta-se os mais antigos até que existam apenas 50 valores.
                if len(self.historico['cost'])>50:
                    self.historico['cost']=self.historico['cost'][-50:]
                    self.historico['accuracy']=self.historico['accuracy'][-50:]

                #self.historico['params'].append([k.numpy() for k in self.params])
                
                #Gerando report.
                self.report(self)
                
                self.loading_index+=1
                
                #Verificando o critério de paragem antecipada.
                if self.times%no_better_interval==0 and self.times>2*no_better_interval:
                    if tf.math.reduce_mean(self.historico['accuracy'][-no_better_interval:])<=tf.math.reduce_mean(self.historico['accuracy'][-2*no_better_interval:-no_better_interval]):
                        print('Early Stop')
                        break
            if self.times-initial_epoch>=number_of_epochs:
                print('Finished')
        except KeyboardInterrupt:
            print('Early Stop')
            
class GANetwork(NeuralNetwork):
    '''
    Esta classe é destinada a redes neurais que são treinadas através da competição entre si (suporte para no máximo duas redes). São criadas duas redes neurais, uma chamada de generator e outra chamada classifier, a classifier é treinada para classificar um vetor como pertencente ao banco de dados ou não, já a generator é uma rede que recebe um ruído aleatório e retornar um vetor com as mesmas dimensões de um vetor do banco de dados da classifier, sendo que a generator é treinada para enganar a classifier, ou seja, ela tenta imitar um vetor do banco de dados. As redes são treinadas de maneira alternada, de modo que elas competem entre si, pois diminuir o erro da classifier implica em aumentar o erro da generator e vice-versa. As redes estão "prontas" quando se obtêm um equilíbrio de Nash entre as duas redes.
    '''
    def __init__(self,dataset_class,dataset_gen,layers_gen,layers_class,check_size,report_func=report_GAN,transformacoes=None):
        '''
        Este método inicializa a rede neural.
        
        dataset_class = função ou None
        dataset_gen = função ou None
        layers_class = lista ou tupla
        layers_gen = lista ou tupla
        check_size = int > 0
        report_func = função
        transformacoes = Qualquer coisa
        
        - Os dataset's devem seguir as mesmas condições do dataset de uma FowardNetwork, porém, com algumas condições adicionais:
            1 - dataset_gen deve retornar uma lista cujo primeiro elemento é o ruído e o segundo são os labels que a generator deve tentar alcançar.
            2 - dataset_class deve retornar uma lista cujo primeiro elemento é o conjunto dos dados que pertencem aos bancos de dados e a segunda coordenada é o label que a classifier deve dar a eles.
        
        - layers devem ser listas ou tuplas de objetos da classe layer (para mais informações veja o arquivo 'Layers.py'), layers_gen são as camadas do gerador e layers_class são as camadas do classificador. Vale lembrar que a saída da última camada do gerador deve ter o mesmo tamanho da entrada da primeira camada do classificador.
        
        - check_size é a quantidade máxima de inputs que a rede neural irá computar por vez, caso seja requisitado que ela cálcule mais inputs do que o check_size ela dividirá os dados inseridos em pacotes com o tamanho do check_size e computará um pacote por vez. Informar um check_size muito grande pode ocasionar em erros de falta de memória (OOM).
        
        - report_func deve ser uma função que exibe um relatório do treino a cada rodada de treino, o único argumento que ela deve receber é a rede neural.
        
        - transformacoes é um argumento opcional que serve para armazenar instruções sobre o tratamento dos dados antes deles serem inseridos na rede neural.
        '''
        
        self.version='GAN_V2.0.0'
        
        #Salvando todos os layers.
        layers=layers_gen+layers_class
        NeuralNetwork.__init__(self,dataset_class,layers,check_size,report_func,transformacoes)
        
        #Armazenando uma lista com os parâmetros de cada rede.
        self.params_gen = [param for layer in layers_gen for param in layer.params]
        self.params_class = [param for layer in layers_class for param in layer.params]
        
        #Armazenando informações sobre os dados.
        self.len_dataset=dataset_class('train')[1].shape[1]
        self.len_validation=dataset_class('validation')[1].shape[1]
        self.dataset_gen=dataset_gen
        
        #Criando a camada de integração. Esta camada integra o classificador com o gerador, permitindo que o gerador treine usando os labels do classificador.
        layers.append(Layers.integration(layers=layers_class,
                                    funcao_erro=layers_gen[-1].funcao_erro,
                                    funcao_acuracia=layers_gen[-1].funcao_acuracia))
        
        #Registrando que a função de custo inserida na última cada do gerador não deve ser usada para calcular o erro, pois a função de erro foi herdada pela camada de integração.
        layers_gen[-1].cost_flag=False
        
        #Criando rede geradora.
        self.generator=FowardNetwork(dataset=self.dataset_gen,
                                    layers=layers_gen+[layers[-1]],
                                    check_size=check_size,
                                    report_func=lambda x: None,
                                    output_layer=-2)
        
        #Criando função preliminar para o dataset da rede classificadora.
        def pre_dataset_class(label):
            generator_data=self.dataset_gen(label)
            observed_data=dataset_class(label)
            
            generator_x=self.generator.encode(generator_data[0])
            with tf.device('/CPU:0'):
                classifier_x=np.concatenate([generator_x,observed_data[0]],axis=1)
                classifier_y=np.concatenate([np.zeros(observed_data[1].shape,config.float_type),observed_data[1]],axis=1)
            
            return classifier_x,classifier_y
        
        self.dataset_class=pre_dataset_class
        
        #Gerando rede classificadora.
        self.classifier=FowardNetwork(dataset=self.dataset_class,
                                      layers=layers_class,
                                      check_size=check_size,
                                      report_func=lambda x: None)
        
        self.gerar=self.generator.encode
        self.classificar=self.classifier.classificar
        
    def otimize(self,number_of_epochs,weight_penalty,learning_method_gen,learning_method_class,batch_size):
        '''
        Esta função otimiza as redes neurais.
        
        inputs:
        number_of_epochs = int >= 0
        weight_penalty = float
        learning_method_gen = lista ou tupla
        learning_method_class = lista ou tupla
        batch_size = int > 0
        
        outputs:
        Nenhum
        
        Comentários:        
        number_of_epochs é o número máximo de rodadas de treino a serem executadas
        
        weight_penalty é o peso a ser dado a regularização dos parâmetros das camadas.
        
        Os learning_method's são o método de aprendizado a ser usado em cada rede neural, eles devem ser da mesma forma do learning_method usado na FowardNetwork.
        
        batch_size é o tamanho dos pacotes de treino, caso a quantidade de elementos nos dados de treino não seja múltiplo do batch_size, os elementos que estiverem sobrando no final não serão usados no treino.
        '''
        initial_epoch=self.times
        #O usuário tem a opção de interromper o treino antecipadamente, para isto, basta gerar o KeyboardInterrupt (basta apertar Ctrl + C no Idle do Python).
        try:
            #Inicializando os parâmetros de otimização do gerador.
            self.generator.otimize(0,
                                   learning_method_gen[0],
                                   weight_penalty[0],
                                   learning_method_gen[1:],
                                   batch_size)
            
            while self.times<=number_of_epochs+initial_epoch:
                #Este for treina cada rede neural uma vez, sendo que primeiro treina-se a rede classificadora e depois a geradora
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

                    erros=other_network.erros(other_network.validation_x,other_network.validation_y)

                    other_network.historico['cost'].append(erros[0])
                    other_network.historico['accuracy'].append(erros[1])

                    self.loading_symbols=['\\','|','/','-']
                    self.loading_index=0

                    self.report(self)

                    self.loading_index+=1
            
            print('Finished')
        except KeyboardInterrupt:
            print('Early Stop')