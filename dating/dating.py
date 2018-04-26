# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:08:51 2018

@author: YangRan
"""
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator
"""
进行相关的分类
创建分类器算法
"""
def Classify0(inx,dateset,label, K):
    #numpy 返回矩阵的行数，shpe[0]行数,shape[1]列数
    dateSetSize=dateset.shape[0]
    #返回二维矩阵的差值
    diffmat=np.tile(inx,(dateSetSize,1))-dateset
    #进行相关的平方
    sqDiffMat=diffmat**2
   #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDist=sqDiffMat.sum(axis=1)
   #开方，计算出距离
    distance=sqDist**0.5
   # 返回distances中元素从小到大排序后的索引值
    sortDistIndicied=distance.argsort()
    classCount={}
    for i in range(K):
       #取出前k个元素的类别
       voteILabel=label[sortDistIndicied[i]]
       #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
       #计算类别次数
       classCount[voteILabel]=classCount.get(voteILabel,0)+1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True )
   
    return sortedClassCount[0][0]        

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    NumOfLines=len(arrayOLines)
    returnMat=np.zeros((NumOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line=line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector
"""
对数据进行归一化处理，
"""
def autoNorm(dataSet):
    minvalue=dataSet.min(0)
    maxvalue=dataSet.max(0)
    ranges=maxvalue-minvalue
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minvalue,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minvalue
    
    

def showdatas(datingDataMat, datingLabels):
     #设置汉字格式
     font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
     #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
     fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
     numberOfLabels = len(datingLabels)
     LabelColors=[]
     for i in datingLabels:
         if i==1:
             LabelColors.append('black')
         if i==2:
             LabelColors.append('orange')
         if i==3:
             LabelColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
     axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelColors,s=15, alpha=.5)
       #设置标题,x轴label,y轴label
     axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
     axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
     axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
     plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
     plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
     plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
     #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
     axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
     axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
     axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
     axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
     plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
     plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
     plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
 
    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
     axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
     axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
     axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
     axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
     plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
     plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
     plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
     didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
     smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
     largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
     axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
     axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
     axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
     plt.show()
"""
函数说明:分类器测试函数
 
Parameters:
    无
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
 
Modify:
    2017-03-24
"""
def datingClassTest():
    #打开的文件名
    filename = "datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0
 
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = Classify0(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))
def classfyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games"))
    ffMiles= float(input("frequent flier miles earned per year"))
    icecream=float(input("liters of ice cream consumed per year"))
    filename = "datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr=np.array([ffMiles,percentTats,icecream])
    classifierResult=Classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("you will probably like this person: ",resultList[classifierResult-1])

if __name__ == '__main__':
    
    datingClassTest()
    classfyPerson()
    
    """
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)   
    showdatas(datingDataMat, datingLabels)
    normDataSet, ranges, minvalue = autoNorm(datingDataMat)
    print(normDataSet)
    print(ranges)
    print(minvalue)
    print(datingDataMat)
    print(datingLabels)
    """