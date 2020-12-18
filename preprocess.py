import pandas as pd
import os,pickle
import numpy as np
import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile

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
    tag=np.loadtxt(fname="rawData/"+filename+".txt",dtype=str,delimiter='\t',comments='\n', usecols=(1,3,4,5),encoding='utf-8')
    print(tag)
    pickleDump(tag,filename+"-tag")


def importDoc(filename):
    doc=np.loadtxt(fname="rawData/"+filename+".txt",delimiter='\t',comments='\n', usecols=(1,6),dtype=str,encoding='utf-8')
    doc=doc.tolist()
    print(doc)
    pickleDump(doc,filename+"-doc-list")

def cut_sentence(data,filename):
    stopList=[line[:-1] for line in open("中文停用词表.txt",'r',encoding='utf-8')]
    for i in range(len(data)):
        doc=jieba.cut(data[i][1])
        doc=" ".join(doc).split()
        doc=[word for word in doc if word not in stopList]
        # print(doc)
        data[i][1]=doc
    print(data)
    pickleDump(data,filename+"-doc-cut")


def getTrainSet(data,filename):
    data=np.array(data,dtype=object)
    TD=[]
    for i in range(data.shape[0]):
        TD.append(TaggedDocument(data[i][1],tags=data[i][0]))
    # print(TD)
    pickleDump(TD,filename+"-trainSet")
    pickleDump(data,filename+"-trainSet-ndarray")

def Doc2VecTrain(data,filename):
    # print(data)
    model=Doc2Vec(documents=data)
    model.train(data,total_examples=model.corpus_count,epochs=model.epochs)
    # print(model)
    fname = get_tmpfile(os.getcwd()+"/midData/"+filename+"-doc2vec.model")
    model.save(fname)

      # you can continue training with the loaded model!

def inferFromModel(inferSet,filename):
    fname = get_tmpfile(os.getcwd()+"/midData/"+filename+"-doc2vec.model")
    model = Doc2Vec.load(fname)
    # dataV = model.infer_vector(data[:,1])
    dataVec = [model.docvecs[z.tags[0]] for z in inferSet]
    print(len(dataVec))
    pickleDump(dataVec,filename+"-dataVec")

#############
# main weibo_train_data
# importTag("test")
# importDoc("test")
# importTag("weibo_train_data")
# importDoc("weibo_train_data")

# data=pickleLoad("weibo_train_data-doc-list")
# cut_sentence(data,"weibo_train_data")

dataL=pickleLoad("test-doc-cut")
getTrainSet(dataL,"test")

trainSet=pickleLoad("test-trainSet")
Doc2VecTrain(trainSet,"test")


inferFromModel(trainSet ,"test")
