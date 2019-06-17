from PIL import Image
import numpy as np
import os, glob

# directory of image folder
basepath_train = "./UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/"

# folder for train
each_train = sorted(glob.glob("{}/*".format(basepath_train)))

# transform image data to numpy form and make dataset
def load_dataset():
    data = []
    
    for folder in each_train:
        files = sorted(glob.glob(folder + "/*.tif"))
        
        print("transform {}'s image to numpy form".format(folder))
        
        for f in files:
            img = Image.open(f).resize((227, 227))
            img = np.asarray(img).astype(np.float32) / 255
            data.append(img)
            
    return np.save("ucsd_ped1_train.npy", np.array(data).reshape((34, 200, 227, 227)))

if __name__ == '__main__':
    load_dataset()
