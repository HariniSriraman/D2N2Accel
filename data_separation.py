# mpiexec -np 4 python data_separation.py

from mpi4py import MPI
import os
from imutils import paths
import shutil

comm = MPI.COMM_WORLD
NODES = comm.Get_size()
RANK = comm.Get_rank()

DIR = "all_data"
OUTPUT_DIR = "data"

classes = os.listdir(DIR)
len_arr = []
qt = []
rem = []

for c in classes:
    l = len(list(paths.list_images(DIR+'/'+c)))

    qt.append(l//NODES)
    rem.append(l%NODES)
    len_arr.append(l)

print(qt,rem)

if RANK==0:
    os.mkdir(OUTPUT_DIR)

comm.Barrier()

for i in range(NODES):
    if i==RANK:
        print("Rank: ",RANK)
        dir1 = OUTPUT_DIR+'/data'+str(i+1)
        os.mkdir(dir1)

        for j in range(len(classes)):
            start = i*qt[j]
            end = start+qt[j]

            img_paths = list(paths.list_images(DIR+'/'+classes[j])) #source

            dest = dir1+'/'+classes[j]
            os.mkdir(dest)

            for k in range(start,end):
                src = img_paths[k]
                shutil.copy(src, dest)
            
            #adding remaining images to data1 folder
            if i==0:
                for k in range(-rem[j],0):
                    src = img_paths[k]
                    shutil.copy(src,dest)





