import pandas as pd
import os,pickle
import numpy as np
import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile

# 从硬盘读取中间结果
def pickleLoad(filename):
    loadF = open(os.getcwd()+"/midData/"+filename+".dat", "rb")
    variable = pickle.load(loadF)
    loadF.close()
    return variable


# 将中间结果保存到硬盘中的midData文件夹，文件的扩展名为.dat
def pickleDump(variable,filename):
    cwd=os.getcwd()
    if os.path.exists(cwd+"/midData") == False:
        os.mkdir(cwd+"/midData")
    loadF = open(cwd+"/midData/"+filename+".dat","wb")
    pickle.dump(variable,loadF)
    loadF.close()


# 导入所有数据的标签（即转发，评论，点赞数）
def importTag(filename):
    # 读取训练集数据的第3，4，5列，原始数据放在rawData文件夹下
    tag=np.loadtxt(fname="rawData/"+filename+".txt",dtype=int,delimiter='\t',comments='\n', usecols=(3,4,5),encoding='utf-8')
    
    print(tag)

    #将导入的训练集数据标签另存为txt
    np.savetxt(os.getcwd()+"/midData/"+filename+"-tag.txt",tag)


# 导入训练集数据的博文id和博文内容
def importDoc(filename):
    # 导入训练集数据的第1，6行（即博文id和博文内容）
    doc=np.loadtxt(fname="rawData/"+filename+".txt",delimiter='\t',comments='\n', usecols=(1,6),dtype=str,encoding='utf-8')
    
    doc=doc.tolist() #将数据转为list变量
    print(doc)

    # 保存中间结果为“训练集名称-doc-list.dat”
    pickleDump(doc,filename+"-doc-list")


# 导入测试集数据的博文id和博文内容
def importDoc2(filename):
    doc=np.loadtxt(fname="rawData/"+filename+".txt",delimiter='\t',comments='\n', usecols=(3),dtype=str,encoding='utf-8')
    doc=doc.tolist()
    print(doc)
    pickleDump(doc,filename+"-doc-list")


# 将导入的训练集的博文内容分词
def cut_sentence(data,filename):
    # 创建停用词表
    stopList=[line[:-1] for line in open("中文停用词表.txt",'r',encoding='utf-8')]
    
    # 将博文内容用" "分词
    for i in range(len(data)):
        doc=jieba.cut(data[i][1])
        doc=" ".join(doc).split()
        doc=[word for word in doc if word not in stopList]
        # print(doc)
        data[i][1]=doc
    
    print(data)

    # 保存分词结果为“数据集名字-doc-cut.dat”
    pickleDump(data,filename+"-doc-cut")


# 同理，将导入的测试集的博文内容分词
def cut_sentence2(data,filename):
    stopList=[line[:-1] for line in open("中文停用词表.txt",'r',encoding='utf-8')]
    for i in range(len(data)):
        doc=jieba.cut(data[i])
        doc=" ".join(doc).split()
        doc=[word for word in doc if word not in stopList]
        # print(doc)
        data[i]=doc
    print(data)
    pickleDump(data,filename+"-doc-cut")


# 将训练集的博文id+博文内容转为gensim使用的TaggedDocument格式
def getTrainSet(data,filename):
    data=np.array(data,dtype=object)
    TD=[]
    for i in range(data.shape[0]):
        # TaggedDocument中不包含博文id，只包含博文内容
        TD.append(TaggedDocument(data[i][1],tags=[i]))
    print(TD)
    pickleDump(TD,filename+"-trainSet")

# 同理，将测试集的博文id+博文内容转为gensim使用的TaggedDocument格式，生成Doc2Vec的训练集
def getTrainSet2(data,filename):
    # data=np.array(data,dtype=object)
    TD=[]
    for i in range(len(data)):
        TD.append(TaggedDocument(data[i],tags=[i]))
    print(TD)
    pickleDump(TD,filename+"-testSet")


# 用训练集训练Doc2Vec模型
def Doc2VecTrain(data,filename):
    # print(data)
    model=Doc2Vec(documents=data) # 新建模型

    #训练模型
    model.train(data,total_examples=model.corpus_count,epochs=model.epochs) 

    # 保存模型到硬盘
    fname = get_tmpfile(os.getcwd()+"/midData/"+filename+"-doc2vec.model")
    model.save(fname)


# 通过模型获得训练集的向量化表示
def inferFromModel(trainSet,filename):
    # 从硬盘载入模型
    fname = get_tmpfile(os.getcwd()+"/midData/"+filename+"-doc2vec.model")
    model = Doc2Vec.load(fname)
    
    # 获得训练集的向量化表示
    dataVec = [model.docvecs[z.tags[0]] for z in trainSet]
    print("samples =",len(dataVec),", dimemsion =",dataVec[0].shape[0])
    # print(dataVec)

    # 将向量化表示另存为txt文件
    np.savetxt(os.getcwd()+"/midData/"+filename+"-docVector.txt",dataVec)


# 通过模型获得测试集的向量化表示
def inferFromModel2(testSet,filename):
    fname = get_tmpfile(os.getcwd()+"/midData/weibo_train_data-doc2vec.model")
    model = Doc2Vec.load(fname)

    # 获得测试集的向量化表示
    dataVec = [model.infer_vector(z.words) for z in testSet]

    print("samples =",len(dataVec),", dimemsion =",dataVec[0].shape[0])
    # print(dataVec)

    np.savetxt(os.getcwd()+"/midData/"+filename+"-docVector.txt",dataVec)


#############
# main 
# 由于我是一步步完成的，使用时前面的步骤我会注释掉。
# 现在我将对于weibo_train_data的一系列操作解除注释，方便观察调用顺序。

# importTag("test")
# importDoc("test")
importTag("weibo_train_data")
importDoc("weibo_train_data")

# importDoc2("weibo_predict_data")

data=pickleLoad("weibo_train_data-doc-list")
cut_sentence(data,"weibo_train_data")
# data=pickleLoad("weibo_predict_data-doc-list")
# cut_sentence2(data,"weibo_predict_data")

dataL=pickleLoad("weibo_train_data-doc-cut")
getTrainSet(dataL,"weibo_train_data")
# dataL=pickleLoad("weibo_predict_data-doc-cut")
# getTrainSet2(dataL,"weibo_predict_data")

trainSet=pickleLoad("weibo_train_data-trainSet")
Doc2VecTrain(trainSet,"weibo_train_data")


inferFromModel(trainSet ,"weibo_train_data")
# testSet=pickleLoad("weibo_predict_data-testSet")
# inferFromModel2(testSet ,"weibo_predict_data")

