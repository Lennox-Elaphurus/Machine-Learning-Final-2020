import pandas as pd
import os,pickle
import numpy as np


# load from pickle file
def pickleLoad(filename):
    loadF = open(os.getcwd()+"/midData/"+filename+".dat", "rb")
    variable = pickle.load(loadF)
    loadF.close()
    return variable


def pickleDump(variable,filename):
    cwd=os.getcwd()
    if os.path.exists(cwd+"/midData") == False:
        os.mkdir(cwd+"/midData")
    loadF = open(cwd+"/midData/"+filename+".dat","wb")
    pickle.dump(variable,loadF)
    loadF.close()


def importTag(filename):
    tag=np.loadtxt(fname="rawData/"+filename+".txt",dtype=str, usecols=(1,4,5,6),encoding='utf-8')
    print(tag)
    pickleDump(tag,filename+"-tag")


def importDoc(filename):
    doc=np.loadtxt(fname="rawData/"+filename+".txt",dtype=str,encoding='utf-8')
    print(doc)
    pickleDump(doc,filename+"-doc")

#############
# main
# importTag("test")
importDoc("test")