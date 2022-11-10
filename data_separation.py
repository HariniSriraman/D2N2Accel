# mpiexec -np 4 python data_separation.py

from mpi4py import MPI
import os
from imutils import paths
import shutil

comm = MPI.COMM_WORLD
NODES = comm.Get_size()
RANK = comm.Get_rank()

DIR = "all_data" #name of the folder where dataset is present. Separate directory for separate labels
OUTPUT_DIR = "data" #name of the folder where the separated data is copied

classes = os.listdir(DIR)
len_arr = []
qt = []
rem = []

#finding the number of images belonging to each class and calculating the number of images
#to be copied in each folder
for c in classes:
    l = len(list(paths.list_images(DIR+'/'+c)))

    qt.append(l//NODES)
    rem.append(l%NODES) #last remaining images lies in range 0 to NODES-1
    len_arr.append(l)

print(qt,rem)

if RANK==0:
    os.mkdir(OUTPUT_DIR) #creating the output directory

comm.Barrier()

#creating separate directories and copying images using separate NODES
for i in range(NODES):
    if i==RANK:
        print("Rank: ",RANK)
        dir1 = OUTPUT_DIR+'/data'+str(i+1)
        os.mkdir(dir1)

        for j in range(len(classes)):
            start = i*qt[j]
            end = start+qt[j]

            img_paths = list(paths.list_images(DIR+'/'+classes[j])) #paths of all the images in 'DIR/class[j]'

            dest = dir1+'/'+classes[j]
            os.mkdir(dest) #creating the subclass directory [eg: 'data1/cats']

            for k in range(start,end):
                src = img_paths[k]
                shutil.copy(src, dest) #copying images from src to dest
            
            #adding remaining images from source to data1 folder
            if i==0:
                for k in range(-rem[j],0):
                    src = img_paths[k]
                    shutil.copy(src,dest)





