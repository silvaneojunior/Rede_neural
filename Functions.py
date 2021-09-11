import time
import config
import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy import stats

tf.config.optimizer.set_jit(config.XLA_opt)

if config.custom_opt:
    optimizations=['layout_optimizer','memory_optimizer','constant_folding','shape_optimization','remapping','arithmetic_optimization','dependency_optimization','loop_optimization','function_optimization','debug_stripper','disable_model_pruning','scoped_allocator_optimization','pin_to_host_optimization','implementation_selector','auto_mixed_precision','disable_meta_optimizer','min_graph_nodes']
    opt_dict={}
    for opt in optimizations:
        exec("opt_dict['{0}']=config.{0}".format(opt))
    tf.config.optimizer.set_experimental_options(opt_dict)
softmax,sigmoid,tanh,relu,leaky_relu=lambda x: tf.nn.softmax(x,axis=1),tf.nn.sigmoid,tf.nn.tanh,tf.nn.relu,tf.nn.leaky_relu
identity=tf.identity

SoftMax=lambda x: (softmax(x)*0.99998)+0.00001
Sigmoid=lambda x: (sigmoid(x)*0.99998)+0.00001
TanH=lambda v:1.7159*tanh(2*v/3)

# Cost functions

def Norm_Ln(n):
    def func(output_expected,output_observed,mask=None):
        error=tf.math.abs(output_expected-output_observed)**n/n
        if mask is not None:
            error=error*mask
        return tf.math.reduce_mean(error)
    return func

L1=Norm_Ln(1)
L2=Norm_Ln(2)

def Cross_Entropy(output_expected,output_observed,mask=None):
    error=-(tf.math.log(output_expected)*output_observed)-(tf.math.log(1-output_expected)*(1-output_observed))
    if mask is not None:
        error=error*mask
    return tf.math.reduce_mean(error)

def Sparse_Cross_Entropy(x,y,mask=None):
    new_y=tf.squeeze(tf.one_hot(y,x.shape[-1],dtype=config.float_type),axis=[-2])
    return Cross_Entropy(x,new_y,mask)

# Regularizer for autoencoders
rho=0.001
def KL_divergence_autoencoder(output_expected,output_observed,mask=None):
    return Cross_Entropy(output_expected,rho,mask)

def L2_autoencoder(output_expected,output_observed,mask=None):
    return L2(output_expected,0,mask)

def L1_autoencoder(output_expected,output_observed,mask=None):
    return L1(output_expected,0,mask)

#Accuracy functions
def L2_accuracy(output_expected,output_observed,mask=None):
    return -L2(output_expected,output_observed,mask)

def Abs_accuracy(output_expected,output_observed,mask=None):
    return -L1(output_expected,output_observed,mask)

def Cross_Entropy_accuracy(output_expected,output_observed,mask=None):
    return -Cross_Entropy(output_expected,output_observed,mask)

def multi_label_accuracy(output_expected,output_observed,mask=None):
    error=tf.cast(tf.math.equal(tf.round(output_observed),tf.round(output_expected)),config.float_type)
    if mask is not None:
        error=error*mask
    return tf.math.reduce_mean(error)

def unique_label_accuracy(output_expected,output_observed,mask=None):
    error=tf.cast(tf.math.equal(tf.math.argmax(output_observed,axis=1),tf.math.argmax(output_expected,axis=1)),config.float_type)
    if mask is not None:
        error=error*mask
    return tf.math.reduce_mean(error)

def Sparse_unique_label_accuracy(x,y,mask=None):
    new_y=tf.squeeze(tf.one_hot(y,x.shape[-1],dtype=config.float_type),axis=[-2])
    return unique_label_accuracy(x,new_y)

#Weigth regularization functions

def L2_regularization(x):
    return tf.math.reduce_sum(x**2)/2

def L1_regularization(x):
    return tf.math.reduce_sum(tf.math.abs(x))

#Report functions

def got_the_time(obs_time):
    milisec,obs_time=(obs_time%1)*1000,obs_time//1
    sec,obs_time=int(obs_time%60),obs_time//60
    minutes,obs_time=int(obs_time%60),obs_time//60
    hours,obs_time=int(obs_time%24),obs_time//24
    
    if obs_time>0:
        obs_time='{0} dias, {1} horas, {2} minutos, {3} segundos e ~{4} milésimos de segundo'.format(str(obs_time),str(hours),str(minutes),str(sec),str(int(milisec)))
    elif hours>0:
        obs_time='{0} horas, {1} minutos, {2} segundos e ~{3} milésimos de segundo'.format(str(hours),str(minutes),str(sec),str(int(milisec)))
    elif minutes>0:
        obs_time='{0} minutos, {1} segundos e ~{2} milésimos de segundo'.format(str(minutes),str(sec),str(int(milisec)))
    elif sec>0:
        obs_time='{0} segundos e ~{1} milésimos de segundo'.format(str(sec),str(int(milisec)))
    else:
        obs_time='~{0} milésimos de segundo'.format(str(int(milisec)))
    return obs_time

dataset_check_flag=True

def report(network):
    global dataset_check_flag
    
    clear_output(wait=True)
    if dataset_check_flag:
        error_list=[]
        indices=tf.range(network.len_dataset)
        indices=tf.random.shuffle(indices)
        for indice in range(network.check_intern_subsets_len):
            slices=indices[indice*network.check_size:(indice+1)*network.check_size]
            x,y=network.dataset('train',slices)
            instant_erro=network.accuracy_eval(x,y)
            error_list.append([instant_erro])
        error_inside=tf.squeeze(tf.math.reduce_mean(error_list,axis=0))
            
        print('Train data          ',network.loading_symbols[network.loading_index%4],'                                                  \n',
              'Accuracy - ',error_inside[1].numpy(),'\n',
              'Cost     - ',error_inside[0].numpy())
              
    obs_time=got_the_time(network.end_time-network.initial_time)
    check_time=got_the_time(time.time()-network.end_time)
    
    print('Accuracy            ',network.loading_symbols[network.loading_index%4],'\n',
          'Current  - ',network.historico['accuracy'][-1].numpy(),'\n',
          'Best     - ',network.best_accuracy.numpy())
    print('Error ',network.loading_symbols[network.loading_index%4],'\n',
          'Current  - ',network.historico['cost'][-1].numpy(),'\n',
          'Best     - ',network.best_cost.numpy())
    # Aqui consideramos que o melhor valor da função de erro é onde temos a melhor acurácia com os pesos que possuem melhor acurácia.
    print('Train configuration ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - ',network.otimization,' - Weight Penalty - ',network.weight_penalty,'\n',
          'Batch size      - ',network.len_subsets,' - Weights - ',[(i,float(network.layers[i].cost_weight)) for i in range(len(network.layers)) if network.layers[i].cost_flag],'\n',
          'Train time      - ',obs_time,'\n',
          'Check time      - ',check_time,'\n',
          'Iteration       - ',network.times)
    
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
    
def check_distr(data):
    plt.figure(figsize=(8*2,6))

    plt.subplot(1, 2, 1)

    x=tf.reshape(data,[-1])
    var=10**-40+(tf.math.reduce_variance(x)*x.shape[0])/(x.shape[0]-1)
    ordered_x=tf.sort((x-tf.math.reduce_mean(x))/(var**0.5))
    empiric_q=tf.range(1,x.shape[0]+1)/(x.shape[0])
    teoric_q=stats.t.cdf(tf.range(-50,50)/10,x.shape[0]-1)
    plt.plot(ordered_x.numpy(),empiric_q.numpy())
    plt.plot((tf.range(-50,50)/10).numpy(),teoric_q)
    plt.title('Accumulated density function')

    plt.subplot(1, 2, 2)
    plt.hist(ordered_x.numpy())
    plt.title('Distribuition')
    plt.show()
    
    M1=tf.math.reduce_max(tf.abs(tf.range(x.shape[0])/(x.shape[0])-stats.t.cdf(ordered_x,x.shape[0]-1)))
    M2=tf.math.reduce_max(tf.abs(tf.range(1,x.shape[0]+1)/(x.shape[0])-stats.t.cdf(ordered_x,x.shape[0]-1)))
    
    stat_test=tf.math.reduce_max(M1,M2)
    
    g=lambda t,i: ((-1)**(i-1))*tf.exp(-2*(i**2)*(t**2))

    k=sum([g((x.shape[0]**0.5)*stat_test,i) for i in range(1,10001)])
    G=1-2*k
    print('KS-Test p-value: ',1-G)