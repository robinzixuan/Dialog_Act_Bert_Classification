#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:49:32 2019

@author: rluo
"""

import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

history = pickle.load(open('history.p','rb'))
plt.plot(history['loss'])
#plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left');
plt.plot(history['acc'])
#plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left');



