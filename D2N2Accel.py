from mpi4py import MPI
import numpy as np
import time
import tensorflow as tf
import cv2
from imutils import paths
import os
import time
import random
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def data_split(fname):
    data=[]
    labels=[]
    i=1
    d=[]
    imagePaths = sorted(list(paths.list_images(fname)))
    for imagePath in imagePaths:
    #  print(i)
      i+=1
      image = cv2.imread(imagePath)
      image = cv2.resize(image,(50, 50))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = img_to_array(image)
      label = imagePath.split(os.path.sep)[-2]
      if label=="NORMAL":
        label=1
        d.append((image,label))
        data.append(image)
        labels.append(label)
      elif label=="PNEUMONIA":
        label=0
        d.append((image,label))
        data.append(image)
        labels.append(label)
    return (d)

def fn_modelfit(model,node):
    gw=[]
    gw1=[]
    for layer in model.layers:
        if node==0:
            if layer.name != 'flatten':
                print("Results from Master")
                print(layer.name)
                lw1=comm.recv(source=1,tag=10)
                lw2=comm.recv(source=2,tag=10)
                lw3=comm.recv(source=3,tag=10)
                lw4=comm.recv(source=4,tag=10)
                gw1=lw1+lw2+lw3+lw4
                gw1=gw1/4
                gw1=comm.bcast(gw1, root=0)
                gw.append(gw1)
                comm.Barrier()
        else:
            if layer.name != 'flatten':
                if node==5:
                    time.sleep(2)
                lw =layer.get_weights()[0]
                comm.send(lw, dest=0, tag=10)
                gw1=comm.bcast(gw1, root=0)
                np_gw1=np.array(gw1)
                layer.set_weights([np_gw1,np.ones(layer.get_weights()[1].shape)])
                comm.Barrier()
           
    if rank==0:
        return gw
    else:
        return None
        
def exec(dt,node):
    labels=[]
    data=[]
    i=1
    a_size = 11
    local_weight = np.zeros(a_size,dtype=np.double)
    for i in dt:
        data.append(i[0])
        labels.append(i[1])
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
#data=np.array([i[np.newaxis,...] for i in data])
    train_X = data
    train_y = labels
    VAL_PCT = 0.1  # lets reserve 10% of our data for validation
    val_size = int(len(data)*VAL_PCT)
    model=Sequential() 
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu',data_format = "channels_last",
                           input_shape = (1,50,50,1)))
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu', data_format = "channels_last"))
    #model.add(tf.keras.layers.ConvLSTM2D(filters = 128, kernel_size = (5, 5), activation = 'relu', data_format = "channels_last"))
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))
    train_X = data[:-val_size]
    train_y = labels[:-val_size]
    test_X = data[-val_size:]
    test_y = labels[-val_size:]
    EPOCHS = 1
    BATCH_SIZE = 8
    Iterations = 9
    steps_per_epoch = 1
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
    for i in range(Iterations):
        history = model.fit(x=train_X,y=train_y,steps_per_epoch=steps_per_epoch, epochs=EPOCHS,validation_data=(test_X,test_y),batch_size=BATCH_SIZE)
        weights=fn_modelfit(model,rank)
    #print(weights)
    if rank==0:
        print("hello")
        #print(weights[0].shape)
        #print(weights[1].shape)
        #print(weights[2].shape)
        
        
     #   model.set_weights(wt)

if rank==0:
    d1=[]
    d1=data_split('data')
    print("master working")
    exec(d1,0)
elif rank==1:
    print("Worker1 working")
    d2=[]
    d2=data_split('data1')
    exec(d2,1)
elif rank==2:
    print("Worker2 working")
    d3=[]
    d3=data_split('data2')
    exec(d3,2) 
elif rank==3:
    print("Worker3 working")
    d4=[]
    d4=data_split('data3')
    exec(d4,3)
elif rank==4:
    print("Worker4 working")
    d1=[]
    d1=data_split('data')
    exec(d1,4)
    
#model.layers[1].set_weights(data)
