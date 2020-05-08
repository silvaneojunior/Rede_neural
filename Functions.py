import time

import theano
from theano import tensor
from theano.tensor import tanh
from theano.tensor.nnet import softmax,relu,sigmoid
import numpy as np
from IPython.display import clear_output

SoftMax=lambda x: (softmax(x)*0.9999998)+0.0000001
Sigmoid=lambda x: (sigmoid(x)*0.9999998)+0.0000001
TanH=lambda v:1.7159*tanh(2*v/3)

def L2(output_expected,output_observed,network):
    return tensor.mean((output_expected-output_observed)**2)

def Cross_Entropy(output_expected,output_observed,network):
    return -tensor.mean((tensor.log(output_expected)*output_observed)+(tensor.log(1-output_expected)*(1-output_observed)))

def L2_accuracy(output_expected,output_observed,network):
    return -tensor.mean((output_expected-output_observed)**2)**0.5

def Abs_accuracy(output_expected,output_observed,network):
    return -tensor.mean(tensor.abs_(output_expected-output_observed))

def multi_label_accuracy(output_expected,output_observed,network):
    return tensor.mean(tensor.eq(output_observed,tensor.round(output_expected)))

def unique_label_accuracy(output_expected,output_observed,network):
    return tensor.mean(tensor.eq(tensor.argmax(output_observed,axis=0),tensor.argmax(output_expected,axis=0)))

def contractive_cost_alt(output_expected,output_observed,network):
    return tensor.mean(tensor.grad(network.layers[-1].cost(network),network.x)**2)

def contractive_cost(output_expected,output_observed,network):
    r,u=theano.scan(lambda i:tensor.sum(theano.gradient.jacobian(network.layers[network.output_layer].outputs[:,i],network.x)[i]**2)/network.x.shape[1],sequences=tensor.arange(network.x.shape[1]))

    return tensor.mean(r)

rho=0.001
def KL_divergence_autoencoder(classificacao,outputs,network):
    media=theano.tensor.mean(classificacao,axis=1)
    return theano.tensor.sum(rho*theano.tensor.log(rho/media)+(1-rho)*theano.tensor.log((1-rho)/(1-media)))

def L2_autoencoder(classificacao,outputs,network):
    return theano.tensor.mean(classificacao**2)

dataset_check_flag=False

def report(network):
    global dataset_check_flag
    
    clear_output(wait=True)
    if dataset_check_flag:
        erros_interno=(network.erros(network.dataset_x,network.dataset_y))
            
        print('Dados de treino ',network.loading_symbols[network.loading_index%4],'\n',
              'Função de erro - ',erros_interno[0],'\n',
              'Função de acurácia - ',erros_interno[1])
    print('Função de acurácia ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - ',network.historico['accuracy'][-1],'\n',
          'Melhor - ',max(network.historico['accuracy']))
    print('Função de erro ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - ',network.historico['cost'][-1],'\n',
          'Melhor - ',min(network.historico['cost']))
    print('Configuração de treino ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - ',network.otimization,' - Weight Penalty - ',network.weight_penalty,'\n',
          'Batch size - ',network.len_subsets,' - Weights - ',[(i,float(network.layers[i].cost_weight.get_value())) for i in range(len(network.layers)) if network.layers[i].cost_flag],'\n',
          'Tempo - ',time.time()-network.tempo,'\n',
          'Iteração - ',network.times,' - Shared - ',network.flag_shared)
    
def report_GAN(network):
    global dataset_check_flag
    
    clear_output(wait=True)
    if dataset_check_flag:

        erros_intern_class=network.classifier.erros(network.classifier.dataset_x,network.classifier.dataset_y)
        erros_intern_gen=network.generator.erros(network.generator.dataset_x,network.generator.dataset_y)
        
        print('Dados de treino ',network.loading_symbols[network.loading_index%4],'\n',
              'Função de erro - Gerador - ',erros_intern_gen[0],'\n',
              'Função de erro - Classifador - ',erros_intern_class[0],'\n',
              'Função de acurácia -  Gerador - ',erros_intern_gen[1],'\n',
              'Função de acurácia -  Classificar - ',erros_intern_class[1])
    print('Função de acurácia ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - Gerador - ',network.generator.historico['accuracy'][-1],'\n',
          'Atual - Classificador - ',network.classifier.historico['accuracy'][-1])
    print('Função de erro ',network.loading_symbols[network.loading_index%4],'\n',
          'Atual - Gerador',network.generator.historico['cost'][-1],'\n',
          'Atual - Classificador',network.classifier.historico['cost'][-1])
    print('Configuração de treino ',network.loading_symbols[network.loading_index%4],'\n',
          'Learning method - Gerador - ',network.generator.otimization,' - Weight Penalty - Gerador - ',network.generator.weight_penalty,'\n',
          'Learning method - Classificador - ',network.classifier.otimization,' - Weight Penalty - Classificador - ',network.classifier.weight_penalty,'\n',
          'Batch size - ',network.classifier.len_subsets,'\n',
          'Tempo - ',time.time()-network.tempo,'\n',
          'Iteração - ',network.times,' - Shared - ',network.flag_shared)