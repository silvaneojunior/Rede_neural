import time
import config
import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy import stats

'''
Este módulo contém as funções padrões que são usada na rede neural.
'''

tf.config.optimizer.set_jit(config.XLA_opt)

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)
softmax,sigmoid,tanh,relu,leaky_relu=lambda x: tf.nn.softmax(x,axis=1),tf.nn.sigmoid,tf.nn.tanh,tf.nn.relu,tf.nn.leaky_relu
identity=tf.identity

#Esta Sigmoid e esta SoftMax foram criadas para evitar erros na Cross Entropy, pois em alguns casos de saturação de ativação as funções originais podiam atingir o valor 1 ou 0, assim estas novas Sigmoid e SoftMax tem a imagem restringia ao intervalo (0.0000001,0.9999999).
SoftMax=lambda x: (softmax(x)*0.99998)+0.00001
Sigmoid=lambda x: (sigmoid(x)*0.99998)+0.00001
#Esta versão da tanh é feita para que a derivada nos pontos onde a imagem da função é -1 ou 1 tenha o módulo da derivada no valor máximo possível.
TanH=lambda v:1.7159*tanh(2*v/3)

#As funções a seguir são funções de custo que podem ser usadas nas redes neurais.

def L2(output_expected,output_observed):
    return tf.math.reduce_mean((output_expected-output_observed)**2)

def Cross_Entropy(output_expected,output_observed):
    return -tf.math.reduce_mean((tf.math.log(output_expected)*output_observed)+(tf.math.log(1-output_expected)*(1-output_observed)))

rho=0.001
def KL_divergence_autoencoder(classificacao,outputs):
    media=tf.math.reduce_mean(classificacao,axis=0)
    return tf.math.reduce_sum(rho*tf.math.log(rho/media)+(1-rho)*tf.math.log((1-rho)/(1-media)))

def L2_autoencoder(classificacao,outputs):
    return tf.math.reduce_mean(classificacao**2)/2

def L1_autoencoder(classificacao,outputs):
    return tf.math.reduce_mean(tf.math.abs(classificacao))

#As funções a seguir são funções de acurácia que podem ser usadas nas redes neurais.
def L2_accuracy(output_expected,output_observed):
    return -tf.math.reduce_mean((output_expected-output_observed)**2)

def Abs_accuracy(output_expected,output_observed):
    return -tf.math.reduce_mean(tf.math.abs(output_expected-output_observed))

def Cross_Entropy_accuracy(output_expected,output_observed):
    return tf.math.reduce_mean((tf.math.log(output_expected)*output_observed)+(tf.math.log(1-output_expected)*(1-output_observed)))

def multi_label_accuracy(output_expected,output_observed):
    return tf.math.reduce_mean(tf.cast(tf.math.equal(tf.round(output_observed),tf.round(output_expected)),config.float_type))

def unique_label_accuracy(output_expected,output_observed):
    return tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(output_observed,axis=1),tf.math.argmax(output_expected,axis=1)),config.float_type))

def Sparse_Cross_Entropy(x,y):
    new_y=tf.one_hot(y,x.shape[1],dtype=config.float_type)[:,0,:]
    return Cross_Entropy(x,new_y)
    
def Sparse_unique_label_accuracy(x,y):
    new_y=tf.one_hot(y,x.shape[1],dtype=config.float_type)[:,0,:]
    return unique_label_accuracy(x,new_y)

def Temporal_Sparse_Cross_Entropy(x,y):
    new_y=tf.one_hot(y,x.shape[2],dtype=config.float_type)[:,:,0,:]
    return Cross_Entropy(x,new_y)
    
def Temporal_unique_label_accuracy(x,y):
    new_y=tf.one_hot(y,x.shape[2],dtype=config.float_type)[:,:,0,:]
    return unique_label_accuracy(x,new_y)

def Sparse_masked_Cross_Entropy(x,y):
    new_y=tf.one_hot(y[:,:,0],x.shape[2],dtype=config.float_type)[:,:,0,:]
    pre_cross_entropy=(tf.math.log(output_expected)*output_observed)+(tf.math.log(1-output_expected)*(1-output_observed))
    pre_cross_entropy=(pre_cross_entropy*y[:,:,0])
    n=tf.math.reduce_sum(y[:,:,0])
    return tf.math.reduce_sum(pre_cross_entropy)/n
    
def Sparse_masked_unique_label_accuracy(x,y):
    new_y=tf.one_hot(y,x.shape[2],dtype=config.float_type)[:,:,0,:]
    pre_accuracy=tf.cast(tf.math.equal(tf.math.argmax(output_observed,axis=1),tf.math.argmax(output_expected,axis=1)),config.float_type)
    pre_accuracy=(pre_accuracy*y[:,:,0])
    n=tf.math.reduce_sum(y[:,:,0])
    return tf.math.reduce_sum(pre_accuracy)/n

#As funções a seguir são funções de regulariazação que podem ser usadas nas redes neurais.

def L2_regularization(x):
    return tf.math.reduce_sum(x**2)/2

def L1_regularization(x):
    return tf.math.reduce_sum(tf.math.abs(x))

#As funções a seguir são as funções de report padrão para redes neurais.

def got_the_time(tempo):
    milisec,tempo=(tempo%1)*1000,tempo//1
    sec,tempo=int(tempo%60),tempo//60
    minutes,tempo=int(tempo%60),tempo//60
    hours,tempo=int(tempo%24),tempo//24
    
    if tempo>0:
        tempo='{0} dias, {1} horas, {2} minutos, {3} segundos e ~{4} milésimos de segundo'.format(str(tempo),str(hours),str(minutes),str(sec),str(int(milisec)))
    elif hours>0:
        tempo='{1} horas, {2} minutos, {3} segundos e ~{4} milésimos de segundo'.format(str(tempo),str(hours),str(minutes),str(sec),str(int(milisec)))
    elif minutes>0:
        tempo='{2} minutos, {3} segundos e ~{4} milésimos de segundo'.format(str(tempo),str(hours),str(minutes),str(sec),str(int(milisec)))
    elif sec>0:
        tempo='{3} segundos e ~{4} milésimos de segundo'.format(str(tempo),str(hours),str(minutes),str(sec),str(int(milisec)))
    else:
        tempo='~{4} milésimos de segundo'.format(str(tempo),str(hours),str(minutes),str(sec),str(int(milisec)))
    return tempo

dataset_check_flag=True

def report(network):
    global dataset_check_flag
    
    clear_output(wait=True)
    if dataset_check_flag:
        erros_list=tf.zeros([0,2],dtype=config.float_type)
        indices=tf.range(network.len_dataset)
        indices=tf.random.shuffle(indices)
        for indice in range(network.check_intern_subsets_len):
            slices=indices[indice*network.check_size:(indice+1)*network.check_size]
            x,y=network.dataset('train',slices)
            instant_erro=network.erros(x,y)
            erros_list=tf.concat([erros_list,[instant_erro]],axis=0)
        erros_interno=tf.math.reduce_mean(erros_list,axis=0)
            
        print('Dados de treino ',network.loading_symbols[network.loading_index%4],'                                                  \n',
              'Função de acurácia - ',erros_interno[1].numpy(),'\n',
              'Função de erro     - ',erros_interno[0].numpy())
              
    tempo=got_the_time(network.end_time-network.initial_time)
    check_time=got_the_time(time.time()-network.end_time)
    
    
    print('Função de acurácia ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual  - ',network.historico['accuracy'][-1].numpy(),'\n',
          'Melhor - ',network.best_accuracy.numpy())
    print('Função de erro ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual  - ',network.historico['cost'][-1].numpy(),'\n',
          'Melhor - ',network.best_cost.numpy())
    # Aqui consideramos que o melhor valor da função de erro é onde temos a melhor acurácia com os pesos que possuem melhor acurácia.
    print('Configuração de treino ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - ',network.otimization,' - Weight Penalty - ',network.weight_penalty,'\n',
          'Batch size - ',network.len_subsets,' - Weights - ',[(i,float(network.layers[i].cost_weight)) for i in range(len(network.layers)) if network.layers[i].cost_flag],'\n',
          'Tempo de treino - ',tempo,'\n',
          'Tempo de checagem - ',check_time,'\n',
          'Iteração - ',network.times)
    
def report_GAN(network):
    global dataset_check_flag
    
    clear_output(wait=True)
    if dataset_check_flag:
        erros_intern=[]
        for net in (network.classifier,network.generator):
            erros_list=tf.zeros([0,2],dtype=config.float_type)
            for indice in range(net.check_intern_subsets_len):
                x,y=net.dataset('train',tf.range(indice*net.check_size,(indice+1)*net.check_size))
                instant_erro=net.erros(x,y)
                erros_list=tf.concat([erros_list,[instant_erro]],axis=0)
            erros_interno=tf.math.reduce_mean(erros_list,axis=0)
            erros_intern.append(erros_list)

        erros_intern_class=erros_intern[0]
        erros_intern_gen=erros_intern[1]
        
        print('Dados de treino ',network.loading_symbols[network.loading_index%4],'\n',
              'Função de erro - Classifador - ',erros_intern_class[0].numpy(),'\n',
              'Função de erro - Gerador     - ',erros_intern_gen[0].numpy(),'\n',
              'Função de acurácia -  Classificar - ',erros_intern_class[1].numpy(),'\n',
              'Função de acurácia -  Gerador     - ',erros_intern_gen[1].numpy())
    tempo=got_the_time(network.end_time-network.initial_time)
    
    print('Função de acurácia ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - Classificador - ',network.classifier.historico['accuracy'][-1].numpy(),'\n',
          'Atual - Gerador       - ',network.generator.historico['accuracy'][-1].numpy())
    print('Função de erro ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - Classificador',network.classifier.historico['cost'][-1].numpy(),'\n',
          'Atual - Gerador      ',network.generator.historico['cost'][-1].numpy())
    print('Configuração de treino ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - Classificador - ',network.classifier.otimization,' - Weight Penalty - ',network.classifier.weight_penalty,'\n',
          'Learning method - Gerador       - ',network.generator.otimization,' - Weight Penalty - ',network.generator.weight_penalty,'\n',
          'Batch size - ',network.batch_size,'\n',
          'Tempo - ',tempo,'\n',
          'Iteração - ',network.times,' - ',network.last_name)
    
def verifica_distr(data):
    plt.figure(figsize=(8*2,6))

    plt.subplot(1, 2, 1)

    x=tf.reshape(data,[-1])
    var=10**-40+(tf.math.reduce_variance(x)*x.shape[0])/(x.shape[0]-1)
    ordered_x=tf.sort((x-tf.math.reduce_mean(x))/(var**0.5))
    empiric_q=tf.range(1,x.shape[0]+1)/(x.shape[0])
    teoric_q=stats.t.cdf(tf.range(-50,50)/10,x.shape[0]-1)
    plt.plot(ordered_x.numpy(),empiric_q.numpy())
    plt.plot((tf.range(-50,50)/10).numpy(),teoric_q)
    plt.title('Função de densidade acumulada')

    plt.subplot(1, 2, 2)
    plt.hist(ordered_x.numpy())
    plt.title('Distribuição')
    plt.show()
    
    M1=tf.math.reduce_max(tf.abs(tf.range(x.shape[0])/(x.shape[0])-stats.t.cdf(ordered_x,x.shape[0]-1)))
    M2=tf.math.reduce_max(tf.abs(tf.range(1,x.shape[0]+1)/(x.shape[0])-stats.t.cdf(ordered_x,x.shape[0]-1)))
    
    stat_test=tf.math.reduce_max(M1,M2)
    
    g=lambda t,i: ((-1)**(i-1))*tf.exp(-2*(i**2)*(t**2))

    k=sum([g((x.shape[0]**0.5)*stat_test,i) for i in range(1,10001)])
    G=1-2*k
    print('Valor do teste de Kolmogorov-Smirnov: ',1-G)