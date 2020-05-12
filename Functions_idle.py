import time
import config
import tensorflow as tf
import numpy as np

'''
Este módulo contém as funções padrões que são usada na rede neural.
'''

softmax,sigmoid,tanh,relu=lambda x: tf.nn.softmax(x,axis=0),tf.nn.sigmoid,tf.nn.tanh,tf.nn.relu

#Esta Sigmoid e esta SoftMax foram criadas para evitar erros na Cross Entropy, pois em alguns casos de saturação de ativação as funções originais podiam atingir o valor 1 ou 0, assim estas novas Sigmoid e SoftMax tem a imagem restringia ao intervalo (0.0000001,0.9999999).
SoftMax=lambda x: (softmax(x)*0.9999998)+0.0000001
Sigmoid=lambda x: (sigmoid(x)*0.9999998)+0.0000001
#Esta versão da tanh é feita para que a derivada nos pontos onde a imagem da função é -1 ou 1 tenha o módulo da derivada no valor máximo possível.
TanH=lambda v:1.7159*tanh(2*v/3)

#As funções a seguir são funções de custo que podem ser usadas nas redes neurais.

def L2(output_expected,output_observed):
    return tf.math.reduce_mean((output_expected-output_observed)**2)

def Cross_Entropy(output_expected,output_observed):
    return -tf.math.reduce_mean((tf.math.log(output_expected)*output_observed)+(tf.math.log(1-output_expected)*(1-output_observed)))

rho=0.001
def KL_divergence_autoencoder(classificacao,outputs):
    media=tf.math.reduce_mean(classificacao,axis=1)
    return tf.math.reduce_sum(rho*tf.math.log(rho/media)+(1-rho)*tf.math.log((1-rho)/(1-media)))

def L2_autoencoder(classificacao,outputs):
    return tf.math.reduce_mean(classificacao**2)

#As funções a seguir são funções de acurácia que podem ser usadas nas redes neurais.

def L2_accuracy(output_expected,output_observed):
    return -tf.math.reduce_mean((output_expected-output_observed)**2)

def Abs_accuracy(output_expected,output_observed):
    return -tf.math.reduce_mean(tf.math.abs(output_expected-output_observed))

def multi_label_accuracy(output_expected,output_observed):
    return tf.math.reduce_mean(tf.cast(tf.math.equal(output_observed,tf.round(output_expected)),config.float_type))

def unique_label_accuracy(output_expected,output_observed):
    return tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(output_observed,axis=0),tf.math.argmax(output_expected,axis=0)),config.float_type))

#As funções a seguir são as funções de report padrão para redes neurais.

dataset_check_flag=False

def report(network):
    global dataset_check_flag
    
    if dataset_check_flag:
        erros_interno=(network.erros(network.dataset_x,network.dataset_y))
            
        print('Dados de treino ',network.loading_symbols[network.loading_index%4],'\n',
              'Função de erro - ',erros_interno[0].numpy(),'\n',
              'Função de acurácia - ',erros_interno[1].numpy())
    print('Função de acurácia ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - ',network.historico['accuracy'][-1].numpy(),'\n',
          'Melhor - ',max(network.historico['accuracy']).numpy())
    print('Função de erro ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - ',network.historico['cost'][-1].numpy(),'\n',
          'Melhor - ',min(network.historico['cost']).numpy())
    print('Configuração de treino ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - ',network.otimization,' - Weight Penalty - ',network.weight_penalty,'\n',
          'Batch size - ',network.len_subsets,' - Weights - ',[(i,float(network.layers[i].cost_weight)) for i in range(len(network.layers)) if network.layers[i].cost_flag],'\n',
          'Tempo - ',time.time()-network.tempo,'\n',
          'Iteração - ',network.times)
    
def report_GAN(network):
    global dataset_check_flag
    
    if dataset_check_flag:

        erros_intern_class=network.classifier.erros(network.classifier.dataset_x,network.classifier.dataset_y)
        erros_intern_gen=network.generator.erros(network.generator.dataset_x,network.generator.dataset_y)
        
        print('Dados de treino ',network.loading_symbols[network.loading_index%4],'\n',
              'Função de erro - Gerador - ',erros_intern_gen[0].numpy(),'\n',
              'Função de erro - Classifador - ',erros_intern_class[0].numpy(),'\n',
              'Função de acurácia -  Gerador - ',erros_intern_gen[1].numpy(),'\n',
              'Função de acurácia -  Classificar - ',erros_intern_class[1].numpy())
    print('Função de acurácia ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - Gerador - ',network.generator.historico['accuracy'][-1].numpy(),'\n',
          'Atual - Classificador - ',network.classifier.historico['accuracy'][-1].numpy())
    print('Função de erro ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - Gerador',network.generator.historico['cost'][-1].numpy(),'\n',
          'Atual - Classificador',network.classifier.historico['cost'][-1].numpy())
    print('Configuração de treino ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - Gerador - ',network.generator.otimization,' - Weight Penalty - Gerador - ',network.generator.weight_penalty,'\n',
          'Learning method - Classificador - ',network.classifier.otimization,' - Weight Penalty - Classificador - ',network.classifier.weight_penalty,'\n',
          'Batch size - ',network.classifier.len_subsets,'\n',
          'Tempo - ',time.time()-network.tempo,'\n',
          'Iteração - ',network.times)