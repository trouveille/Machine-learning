
# coding: utf-8

# In[1]:


from numpy import *

import operator     #导入需要用到的包

def creatdataset():

    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])   #四个样本的特征

    labels=['A','A','B','B']    #四个样本的标签

    return group,labels

def classify0(inx,dataset,labels,k):     #inX是你要输入的要分类的“坐标”，dataSet是上面createDataSet的array，

                                            #就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里面的k  

    datasetsize=dataset.shape[0]    #dataSetSize是dataSet的行数，用上面的举例就是4行

    diffmat=tile(inx,(datasetsize,1))-dataset      #前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，

                                                #dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），

                                                #然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减  

    sqdiffmat=diffmat**2

    sqdistances=sqdiffmat.sum(axis=1)   #axis=1是行相加，这样得到了(x1-x2)^2+(y1-y2)^2  

    distance=sqdistances**0.5     #这样求出来就是欧式距离

    

    sorteddistanceindicies=distance.argsort()    #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])  

    classcount={}

    for i in range(k):

        votelabel=labels[sorteddistanceindicies[i]]

        classcount[votelabel]=classcount.get(votelabel,0)+1        #求每个类别的个数，有‘A’就让'A'的计数加1

        

    sortclass=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)   #从大到小排序

    return sortclass[0][0]   #第一个就是最大的，返回最大的类别就是预测的类别

group,labels=creatdataset()

classre=classify0([0,0],group,labels,3)      #要预测的样本是【0,0】这个点

print(classre)


# In[3]:



from numpy import *

import operator    

def filematrix(filename):

    fr=open(filename)      #打开文件

    arrayoflines=fr.readlines()    #读取文件

    numberoflines=len(arrayoflines)     #得到文件的行数

    returnmat=zeros((numberoflines,3))      #returnmat是一个numberoflines行3列的0矩阵，用来存放feature

    classlabelvector=[]        #设置一个空list来存label

    index=0

    for line in arrayoflines:     #一行一行读数据

        line=line.strip()     #截取所有回车字符

        listfromline=line.split('\t')    #将上一步得到的整行数据分割成一个列表

        returnmat[index,:]=listfromline[0:3]             #1-3列是特征列，依次将特征存放到returnmat中

        classlabelvector.append(listfromline[-1])     #最后一列也就是第4列是label列，存放到classlabelvector中

        index+=1

    return returnmat,classlabelvector    #返回特征矩阵和标签列表

#将特征值归一化

def autonorm(dataset):

    minvalue=dataset.min(0)    #每列的最小值

    maxvalue=dataset.max(0)    #每列的最大值

    ranges=maxvalue-minvalue     

    normdataset=zeros(shape(dataset))    #数据集大小的0矩阵

    m=dataset.shape[0]     #数据集行数

    normdataset=dataset-tile(minvalue,(m,1))     #数据集-m行的3列最小值

    normdataset=normdataset/tile(ranges,(m,1))   #数据集-m行的3列最大值

    return normdataset,ranges,minvalue      #返回归一化之后的数据

#测试

def datingtest():

    horatio=0.10        

    datingdatamat,datinglabel=filematrix('datingTestSet2.txt')  #打开文件，并把特征和标签都存到相应的矩阵和列表中

    normmat,ranges,minvalue=autonorm(datingdatamat)     #得到归一化后的数据

    m=normmat.shape[0]                     #得到数据的行数

    numtestvecs=int(m*horatio)           #得到10%的测试数据

    errorcount=0                         #初始错误个数设为0

    for i in range(numtestvecs):

        classifierresult=classify0(normmat[i],normmat[numtestvecs:m,:],datinglabel[numtestvecs:m],3)    #利用分类函数得到预测类别

        print('the classifier came back with:',classifierresult,'the real answer is:',datinglabel[i])    

        if(classifierresult!=datinglabel[i]): 

            errorcount+=1                                 #如果预测类别和真实类别不等的话，错误个数+1

    print('the error rate:',(errorcount/float(numtestvecs)))         #输出错误率
datingtest()

