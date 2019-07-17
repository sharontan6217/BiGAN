#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:07:51 2019

@author: sharontan
"""

from PIL import Image


import os
import math
import inspect
import sys
import importlib
import random

import numpy as np
from numpy import log10, log2, exp2

import pandas as pd

import datetime
from datetime import timedelta

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as bkend
from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, convolutional, pooling, Reshape, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import metrics
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, Adamax
from keras.utils.generic_utils import Progbar

import tensorflow as tf
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt

from plotnine import *
import plotnine


# init data
num_generations = 1000 
PopulationSize = 215
PredictSize =15
currentTime=datetime.datetime.now()

os.environ["KERAS_BACKEND"] = "tensorflow"
importlib.reload(bkend)
print(device_lib.list_local_devices())




class BiGAN(BaseEstimator,
            TransformerMixin):
    def __init__(self,
                 z_size=None,
                 iterations=None,
                 batch_size=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        
        for arg, val in values.items():
            setattr(self, arg, val)
            
        optimizer1=Adamax(0.002,0.5)
        
        # Build the discriminator.
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=optimizer1,
                                   loss="mean_squared_error",
                                   metrics=["accuracy"])

        # Build the generator to fool the discriminator.
        # Freeze the discriminator here.
        self.discriminator.trainable = False
        self.generator = self.build_generator()
        self.encoder = self.build_encoder()
        
        noise = Input(shape=(self.z_size, ))
        generated_data = self.generator(noise)
        fake = self.discriminator([noise, generated_data])

        real_data = Input(shape=(1,))
        encoding = self.encoder(real_data)
        valid = self.discriminator([encoding, real_data])

        # Set up and compile the combined model.
        # Trains generator to fool the discriminator.
        self.bigan_generator = Model([noise, real_data], [fake, valid])
        self.bigan_generator.compile(loss=["mean_squared_error", "mean_squared_error"],
                                     optimizer=Adamax(0.004,0.5))
 
    def fit(self,
            X,
            y=None):
        num_train = X.shape[0]
        start = 0
        
        # Adversarial ground truths.
        valid = np.ones((self.batch_size, 1)) 
        fake = np.zeros((self.batch_size, 1))        
        
        for step in range(self.iterations):
            # Generate a new batch of noise...
            noise = np.random.uniform(low=-1.0, high=1.0, size=(self.batch_size, self.z_size))
            # ...and generate a batch of synthetic returns data.
            generated_data = self.generator.predict(noise)
            
            # Get a batch of real returns data...
            stop = start + self.batch_size
            real_batch = X[start:stop]
            # ...and encode them.
            encoding = self.encoder.predict(real_batch)
            # Train the discriminator.
            d_loss_real = self.discriminator.train_on_batch([encoding, real_batch], valid)
            d_loss_fake = self.discriminator.train_on_batch([noise, generated_data], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator.
            g_loss = self.bigan_generator.train_on_batch([noise, real_batch], [valid, fake])
            
            start += self.batch_size
            if start > num_train - self.batch_size:
                start = 0
            
            if step % 100 == 0:
                # Plot the progress.
                print("[Discriminator loss: %f, Discriminator accuracy: %.2f%%] [Generator loss: %f]" % (d_loss[0], 100 * d_loss[1], g_loss[0]))
                
        return self

    def transform(self,
                  X):
        return self.feature_extractor.predict(X)

    def build_encoder(self):
        encoder_input = Input(shape=(1,))

        encoder_model = Dense(units=90)(encoder_input)
        encoder_model = LeakyReLU(alpha=0.2)(encoder_model)
        encoder_model = BatchNormalization()(encoder_model)
        encoder_model = Dense(units=90)(encoder_model)
        encoder_model = LeakyReLU(alpha=0.2)(encoder_model)
        encoder_output = Dense(units=self.z_size, activation="tanh")(encoder_model)
        
        self.feature_extractor = Model(encoder_input, encoder_output)
        
        return Model(encoder_input, encoder_output)
    
    def build_generator(self):
        # We will map z, a latent vector, to continuous returns data space (..., 1).
        latent = Input(shape=(self.z_size,))

        # This produces a (..., 100) shaped tensor.
        generator_model = Dense(units=90, activation="elu")(latent)
        generator_model = BatchNormalization()(generator_model)
        generator_model = Dense(units=90, activation="elu")(generator_model)
        generator_model = BatchNormalization()(generator_model)

        generator_output = Dense(units=1, activation="linear")(generator_model)
        
        return Model(latent, generator_output)
    
    def build_discriminator(self):
        z = Input(shape=(self.z_size,))
        ret_data = Input(shape=(1,))
        discriminator_inputs = concatenate([z, ret_data], axis=1)

        discriminator_model = Dense(units=180)(discriminator_inputs)
        discriminator_model = LeakyReLU(alpha=0.2)(discriminator_model)
        discriminator_model = Dropout(rate=0.5)(discriminator_model)


        discriminator_output = Dense(units=1, activation="sigmoid")(discriminator_model)
        
        return Model([z, ret_data], discriminator_output)
    
    def data_load():
        global realSize, log_dir, graph_dir_1, graph_dir_2, fileName
        
        fileName='LIBOR_example'
        data_dir='/data/example.csv'
        log_dir='/output_predict/'
        graph_dir_1='/graph/distribution/'
        graph_dir_2='/graph/line/'
        #get_ipython().magic("matplotlib inline")

        df = pd.read_csv(data_dir,na_values='null',na_filter=True)
        df['example']=pd.to_numeric(df['example'],errors='coerce')
        df['DATE']=df['DATE'].astype(str)

        n=random.randint(1,1000)
        #n=30
        
        df=df.iloc[:,0:2]
        df=df.loc[pd.notnull(df['example'])]
        df_train=df.iloc[-1001:,1:2]
        #df_test=df.iloc[-PopulationSize-n-1:-PredictSize-n,0:2]
        df_test=df.iloc[-PopulationSize-n-1:-PredictSize-n,0:2]
        df_test.reset_index()
        startDate=df_test.iloc[0]['DATE']
        predictDate=df_test.iloc[200]['DATE']
        df_test=df_test.iloc[:,1:2]
        df_actual=df.iloc[-PopulationSize-n-1:-n,1:2]  
        data=df_train.values
        data_test=df_test.values
        data_actual=df_actual.values
        data_mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        x_train=data[0:1000]
        y_train=data[1:1001]
        d_train=y_train-x_train
        z_train=[]
        for i in range(len(x_train)):
            d_train[i]=y_train[i]-x_train[i]
            if d_train[i]<0:
                z_train.append((-1)*log2((-1)*d_train[i]))
            if d_train[i]==0:
                z_train.append(0)
            if d_train[i]>0:
                z_train.append(log2(d_train[i]))
            i+=1
        z_train = np.asarray(z_train)
        x_test = data_test[0:200]
        y_test = data_test[1:201]
        d_test=y_test-x_test
        z_test=[]
        for i in range(len(x_test)):
            d_test[i]=y_test[i]-x_test[i]
            if d_test[i]<0:
                z_test.append((-1)*log2((-1)*d_test[i]))
            if d_test[i]==0:
                z_test.append(0)
            if d_test[i]>0:
                z_test.append(log2(d_test[i]))
            i+=1
        z_test = np.asarray(z_test)
        z_test=np.reshape(z_test,(len(z_test),1))
        x_sample=data_actual[0:215]
        y_sample=data_actual[1:216]
        return x_test, y_test, z_train, z_test, x_sample, y_sample, startDate, predictDate, n
    
    def predict (self):
        
        x_test, y_test, z_train, z_test, x_sample, y_sample, startDate, predictDate, n = BiGAN.data_load()
        
        z_size = 10
        bigan = BiGAN(z_size=z_size,
                      batch_size=128,
                      iterations=10000 )
        
        bigan.fit(X=z_train)
        
        n_sim = 1000
        noise_train = np.random.uniform(low=-1.0, high=1.0, size=(n_sim, z_size))
        trainPredict = np.zeros(shape=(n_sim,1))
        for i, xi in enumerate(noise_train):  
            trainPredict[i, :] = bigan.generator.predict(x=np.array([xi]))[0]
            i+=1
        
        bigan.fit(X=z_test)
        n_test = realSize
        noise_test = np.random.uniform(low=-1.0, high=1.0, size=(n_test, z_size))
        testPredict = np.zeros(shape=(n_test,1))
        for i, xi in enumerate(noise_test):  
            testPredict[i, :] = bigan.generator.predict(x=np.array([xi]))[0]
            i+=1

        
        n_predict = PopulationSize
        x=np.zeros(shape=(n_predict,1))
        x_test1=z_test

        for i in range(PredictSize):
            if len(x_test1)==200:
                x_new=[testPredict[realSize+i-1]]
            else:
                x_new=[x[realSize+i-1]]
            x_test1=np.append(x_test1,x_new,axis=0)        
            bigan.fit(X=x_test1)
            noise_predict = np.random.uniform(low=-1.0, high=1.0, size=(len(x_test1), z_size))    
            for j, tj in enumerate(noise_predict): 
                x[j, :]=bigan.generator.predict(x=np.array([tj]))[0]
                j+=1    
            i+=1

        
        x_mean = np.zeros(shape=(x.shape[0]))
        d_predict=np.zeros(shape=(x.shape[0],1))
        x_actual=np.zeros(shape=(x.shape[0],1))
        x_predict=np.zeros(shape=(x.shape[0],1))
        for i in range(x.shape[0]):
            if x[i, :]>0:
                d_predict[i]=(-1)*np.exp2((-1)*x[i, :])
            if x[i, :]==0:
                d_predict[i]=0
            if x[i, :]<0:
                d_predict[i]=np.exp2(x[i, :])
            
            if i<=199:
                x_actual[i]= x_test[i]
                x_predict[i] = d_predict[i]+ x_test[i]
            else:
                
                x_actual [i] = x_predict[i-1]
                x_predict[i]=d_predict[i]+x_actual[i]
            
            #x_actual[i] = x_sample[i]
            #x_predict[i] = d_predict[i]+ x_actual[i]
            x_mean[i] = np.average(a=x_predict[i])
            i+=1

        x_predict=np.asarray(x_predict)
        #print(x_actual,x_predict,d_predict)

        print("Test Dataset starts from ",n, "Test Dataset ends to ",n+PopulationSize, "Test Dataset StartDate",startDate,"Predict StartDate ",predictDate)
        
        
        act_mean = np.zeros(shape=y_sample.shape[0])
        for i in range(y_sample.shape[0]):
            act_mean[i] = np.average(a=(y_sample[i]))
            i+=1

        
        plotnine.options.figure_size = (12, 9)
        plot = ggplot(pd.melt(pd.concat([pd.DataFrame(x_mean, columns=["BiGAN Portfolio Returns Distribution"]).reset_index(drop=True),
                                         pd.DataFrame(act_mean, columns=["Actual Portfolio Returns Distribution"]).reset_index(drop=True)],
                                        axis=1))) + \
        geom_density(aes(x="value",
                         fill="factor(variable)"), 
                     alpha=0.5,
                     color="black") + \
        geom_point(aes(x="value",
                       y=0,
                       fill="factor(variable)"), 
                   alpha=0.5, 
                   color="black") + \
        xlab("Portfolio returns") + \
        ylab("Density") + \
        ggtitle("Trained Bidirectional Generative Adversarial Network (BiGAN) Portfolio Returns") + \
        theme_matplotlib()
        plot.save(filename='output_ga_'+str(object=predictDate)+'_'+str(object=currentTime)+'_bigan_ga_distribution.png', path=graph_dir_1)
        
        print("The VaR at 1%% estimate given by the BiGAN: %.2f%%" % (100 * np.percentile(a=x_mean, axis=0, q=1
        
        return (x_actual,x_predict,x_sample, y_sample, startDate, predictDate)
        
    def visualising(self):
    #generate output data
        x=[]
        y=[]
        d=[]
        from sklearn.metrics import mean_squared_error
        for i in range(PopulationSize):
            x=x_sample[i]
            #x=x_actual[i]
            y=x_predict[i]
            d=x-y
            df_output=pd.DataFrame({'actual':x,'predict':y,'difference':d})
            file_name='output_gabigan_'+fileName+str(object=currentTime)+'.csv'
            if not os.path.isfile(file_name):           
                df_output.to_csv(file_name,mode='a',header=True,index=True)
            else:
                       df_output.to_csv(file_name,mode='a',header=False,index=True)   
            self.x.append(x)
            self.y.append(y)
            self.d.append(d)
            i+=1  
    #generate plot    
       
        mse= mean_squared_error(self.x,self.y,multioutput='raw_values')
        avg_diff=np.average(self.d) 
        print('mse=',mse)
        print('average of difference=',avg_diff)
        a=np.std(np.array([self.y,self.x]))
        print('std=',a)

        plt.figure()
        plt.plot(self.x,color='red',label='real value')
        plt.plot(self.y,color='blue',label='predicted value')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Predict Start From {}'.format(testDate))
        plt.legend()
        png_name='test_scenario_'+str(object=testDate)+'_'+fileName+'_'+str(object=currentTime)+'_bigan_ga_line.png'
        plt.savefig(graph_dir_2+png_name)
        plt.close()
        
    #generate output log
        f= open(log_dir+'log.txt','a') 
        f.write('----------------------------------------------------\n')
        f.write('mse={}\n'.format(mse))
        f.write('average of difference={}\n'.format(avg_diff))
        f.write('std={}\n'.format(a))
        f.close()

    


    def clean(self):
        del self.x[:]
        del self.y[:]
        del x_actual[:]
        del x_predict[:]
        del self.d[:]
        del x_sample[:]
        #plt.cla()
        #plt.clf()
        #del self.testDate[:]
        #del self.plotDate[:]


if __name__ == '__main__':
    x=BiGAN()
    x.predict()
    x.visualising()
    x.clean()




    
        
