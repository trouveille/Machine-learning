
# coding: utf-8

# In[1]:


from numpy import *

import operator     #导入需要用到的包


# In[2]:



def creatdataset():

    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])   #四个样本的特征

    labels=['A','A','B','B']    #四个样本的标签

    return group,labels


# In[3]:



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


# In[4]:



group,labels=creatdataset()

classre=classify0([0,0],group,labels,3)      #要预测的样本是【0,0】这个点

print(classre)

