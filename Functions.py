import theano
from theano import tensor
from theano.tensor import tanh
from theano.tensor.nnet import softmax,relu,sigmoid
import numpy as np

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