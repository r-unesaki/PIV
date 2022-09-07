import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


for i_tobasi in range(20,100,40):
    for kensa in range(32,96,32):
        for tansa_1 in range(16,kensa,16):
            for overlap in range(1,2):
                dir_r=f"20220826_143300/{i_tobasi}_{kensa}_{tansa_1}_{overlap}"
                dir_w="vel_images/"
                dir_w=dir_r+dir_w

                if not os.path.exists(dir_w):
                    os.makedirs(dir_w)

                start=1
                end=49

                images=[]

                for i in range(start,end+1):
                    print(dir_r+"data/v." + str(i).zfill(8) + ".dat")

                    datafname=dir_r+"/data/v." + str(i).zfill(8) + ".dat"
                    imgfname=dir_w + "v." + str(i).zfill(8) + ".png"

                    data = np.loadtxt(datafname)

                    fig, ax = plt.subplots()

                    ax.quiver(data[:,0],data[:,1],data[:,2],data[:,3])

                    ax.grid()
                    ax.set_aspect('equal')
                    plt.savefig(imgfname,format="png")
                    images.append(Image.open(imgfname))

                    plt.close()

                print(len(images))
                images[0].save(dir_r+'vel.gif',save_all=True,append_images=images[1:],optimize=False,duration=500,loop=0)