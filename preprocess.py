import pandas as pd
import os,pickle
import numpy as np
import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec


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
    tag=np.loadtxt(fname="rawData/"+filename+".txt",dtype=str,delimiter='\t',comments='\n', usecols=(3,4,5),encoding='utf-8')
    print(tag)
    pickleDump(tag,filename+"-tag-noMid")


def importDoc(filename):
    doc=np.loadtxt(fname="rawData/"+filename+".txt",delimiter='\t',comments='\n', usecols=(6),dtype=str,encoding='utf-8')
    print(doc)
    pickleDump(doc,filename+"-doc-noMid")

def cut_sentence(data):
    stopList=[line[:-1] for line in open("中文停用词表.txt",'r',encoding='utf-8')]
    result=[]
    for i in range(data.shape[0]):
        doc=jieba.cut(data[i][1])
        doc=" ".join(doc).split()
        doc=[word for word in doc if word not in stopList]
        print(doc)
        data[i][1]=list(doc)


#############
# main weibo_train_data

# importTag("weibo_train_data")
# importDoc("weibo_train_data")

# data=pickleLoad("test-doc")
# cut_sentence(data)
# print(data)