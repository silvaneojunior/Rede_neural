import numpy as np
import tensorflow as tf
import NeuralNetwork as rn
import Layers
import Functions as default
import Normalize

import _pickle as cPickle
import gzip

f = gzip.open("mnist.pkl.gz", 'rb')
training_data, validation_data, test_data = cPickle.load(f,encoding='latin1')
f.close()

training_data, validation_data, test_data

train_x=tf.convert_to_tensor(
    np.concatenate([training_data[0], validation_data[0]],axis=0).T,
dtype='float32')

validation_x=tf.convert_to_tensor(test_data[0].T,dtype='float32')

train_y=np.zeros([10,60000])
validation_y=np.zeros([10,10000])

for i,j in zip(range(60000),training_data[1].tolist()+validation_data[1].tolist()):
    train_y[j-1,i]=1
for i,j in zip(range(10000),test_data[1].tolist()):
    validation_y[j-1,i]=1
    
train_y=tf.convert_to_tensor(train_y,dtype='float32')
validation_y=tf.convert_to_tensor(validation_y,dtype='float32')

def dataset(label):
    if label=='train':
        return [train_x,train_y]
    elif label=='validation':
        return [validation_x,validation_y]
    
    
    
layers=[]


layers.append(Layers.FC(n_inputs=784,
                        n_neurons=64,
                        ativ_func=default.relu))

layers.append(Layers.FC(n_inputs=layers[-1].n_outputs,
                        n_neurons=64,
                        ativ_func=default.relu))

layers.append(Layers.FC(n_inputs=64,
                        n_neurons=10,
                        ativ_func=default.SoftMax,
                        funcao_erro=default.Cross_Entropy,
                        funcao_acuracia=default.unique_label_accuracy))

Network=rn.FowardNetwork(dataset,
                   layers,
                   transformacoes=None,
                   check_size=1000,
                   report_func=default.report)

trainpack=[[5,['RMSProp',10**(-3),0.1],False]]

default.dataset_check_flag=False

for n,learning_method,flag_data in trainpack:
    Network.otimize(no_better_interval=n,
                    number_of_epochs=10,
                    weight_penalty=0,
                    learning_method=learning_method,
                    batch_size=64,
                    update_data=flag_data)
