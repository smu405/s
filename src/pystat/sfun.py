
"""
author: jsl
since: 20160325

collection of functions from *.ipynb
"""
#globals
global myhome
global mywd
global plantdir
global ch2wd
global ch2dd
global myV
global myVT4
global myVT0
global myX
global myL
global myLCol
global myTree
global mykL

def setupDirs():
    global myhome
    global mywd
    global plantdir
    global ch2wd
    global ch2dd
    import os
    myhome=os.path.expanduser('~')
    mywd=os.path.join(myhome,'Code/git/jsl/pystat/')
    plantdir=os.path.join(myhome,'Code/git/gh/s/lib/')
    ch2wd=os.path.join(myhome,'Code/git/else/machinelearninginaction/Ch02/')
    ch2dd=os.path.join(ch2wd,'trainingDigits')

def setupDt():
    global myV
    global myVT4
    global myVT0
    global myX
    global myL
    global myLCol
    global myTree
    global mykL
    # data breast cancer
    # https://archive.ics.uci.edu/ml/machine-learning-databases/
    # breast-cancer-wisconsin/breast-cancer-wisconsin.data
    url='http://mlearn.ics.uci.edu/databases/voting-records/house-votes-84.data'
    # visit this stie to find it in a csv format
    import requests
    r=requests.get(url)
    text=r.iter_lines()
    import csv
    reader=csv.reader(text,delimiter=',')

    import numpy as np
    myV=np.array([['young', 'false', 'false', 'fair', 'No'],
           ['young', 'false', 'false', 'good', 'No'],
           ['young', 'true', 'false', 'good', 'Yes'],
           ['young', 'true', 'true', 'fair', 'Yes'],
           ['young', 'false', 'false', 'fair', 'No'],
           ['middle', 'false', 'false', 'fair', 'No'],
           ['middle', 'false', 'false', 'good', 'No'],
           ['middle', 'true', 'true', 'good', 'Yes'],
           ['middle', 'false', 'true', 'excellent', 'Yes'],
           ['middle', 'false', 'true', 'excellent', 'Yes'],
           ['old', 'false', 'true', 'excellent', 'Yes'],
           ['old', 'false', 'true', 'good', 'Yes'],
           ['old', 'true', 'false', 'good', 'Yes'],
           ['old', 'true', 'false', 'excellent', 'Yes'],
           ['old', 'false', 'false', 'fair', 'No']], 
          dtype='|S9')
    myVT4=myV.T[4]
    myVT0=myV.T[0]
    myL=myV.tolist()
    myLCol=swapColRow(myL)
    myTree=createTree(myLCol)
    myX=[2,2,8,7,5,3,1,1,2,2,8,7,5,3,9,7]
    mykL=[
        ['young','middle','old'],
        ['true','false'],
        ['true','false'],
        ['excellent','good','fair']
    ]

# input: none
# output: none
# usage: sfun.reload()
def reload():
    import imp
    import sys
    if 'sfun' in sys.modules:
        imp.reload(sys.modules['sfun'])
    else:
        import sfun 

# >>> knn.ipynb
def getFileList(d):
    """
    in: directory path
    out: list of absolute file names
    usage: trainFiles=getFileList(ch2dd)
    """
    import os
    return [os.path.join(d, f) for f in os.listdir(d)]

# input: a file list with a format of 32 rows x 32 cols
# output: list of data (one column = 1934 x 32 x 32)
# usage: _dataAll=_readData(trainFiles)
def _readData(files):
    rows=32;
    cols=32;
    data = list()
    for f in files:
        fr = open(f, 'r')
        for i in range(rows):
            row=fr.readline()
            for j in range(cols):
                data.append(int(row[j]))
        fr.close()
    return data

# input: a file list
# output: list of data (with a 3d format: 1934 by 32 by 32)
# readlines -> '\r\n' -> splitlines
# eof problem -> no marker -> returns "" 
# convert string into integer problem
# usage: dataAll=readData(trainFiles)
def readData(files):
    data=list()
    for f in files:
        with open(os.path.join(dir,f)) as fr:
            data.append(fr.read().splitlines())
    return data

# input: list of files (one list of all files, each with a format of 32 x 32)
# output: 1) 32x32 vectors, 2) list of class labels, 3) list of file names
# dependency: _readData(files)
# usage: dataV,cLabels,fileNames = readDataVector(trainFiles)
def readDataVector(files):
    import numpy as np
    _dataAll=_readData(files)
    nFiles=len(files)
    nRowsCols=32*32
    dataV=np.zeros((nFiles,nRowsCols))
    cLabels=list()
    fileNames=list()
    for i in range(nFiles):
        filename=files[i]
        begin=i*nRowsCols
        end=begin+nRowsCols
        dataV[i,:]=_dataAll[begin:end]
        classNumStr=os.path.basename(filename).split('_')[0]
        cLabels.append(classNumStr)
        fileNames.append(filename)
    return dataV, cLabels, fileNames

# input: xs train data, x test data
# output: list of distances
def getDistanceUsingEq(xs,x):
    import math
    dMath=list()
    for item in xs:
        dtemp=0
        for i in range(len(x)):
           dtemp+=math.pow(item[i]-x[i],2)
        dtemp=math.sqrt(dtemp)
        dMath.append(dtemp)
    return dMath

# input:
# 1) sel: user selection -> one of 'norm', 'eucl" or else
# 2) xs: train data
# 3) x: test data
# output: list of distances
# dependency: getDistanceUsingEq(xs,x)
# usage:
# myV=np.array([[1.0,2.0],[2.0,3.0],[0,0],[4.0,5.0]])
# myP=np.array([0,0])
# normDistances=getDistance("norm",myV,myP)
# euclDistances=getDistance("eucl",myV,myP)
# mathDistances=getDistance("math",myV,myP)
def getDistance(sel,xs,x):
    import numpy as np
    d=list()
    if sel=="norm":
        d=[np.linalg.norm(i-x) for i in xs]
    elif sel=="eucl":
        d=np.sqrt(((xs-x)**2).sum(axis=1))
    else:
        d=getDistanceUsingEq(xs,x)
    return d

# input: data to sort, k neighbors
# output: sorted indices
# usage:
# knnIndices=selectKNeighbors(normDistances,k)
# print [myV[i] for i in knnIndices]
# print [myC[i] for i in knnIndices]
def selectKNeighbors(data,k):
    import numpy as np
    return np.array(data).argsort()[:k]

# input: data list, nMostCommon number of most common
# output: nMostCommon list of pair (class, freq)
# usage:
# knnClasses=[myC[i] for i in selectKNeighbors(normDistances,k)]
# print vote(knnClasses, 1)
def vote(data, nMostCommon):
    import collections
    return collections.Counter(data).most_common(nMostCommon)

# input
# 1) x: test data (features)
# 2) S: data set (list of features)
# 3) c: class
# 4) k: num of neighbhors
# output: list of majority
# usage:
# trainFiles=getFileList(ch2dd)
# dataV,cLabels,fileNames = readDataVector(trainFiles)
# majority, indices=knnClassify(dataV[100],dataV,cLabels,3)
def knnClassify(x, S, c, k):
    normDistances=getDistance("norm",S,x)
    knnIndices=selectKNeighbors(normDistances,k)
    print "\tNeighbor indices: {0}".format(knnIndices)
    knnClasses=[c[i] for i in knnIndices]
    majority=vote(knnClasses, 1)
    print "\tmajor class -> {0}".format(majority)
    return majority, knnIndices

# --->>> decision tree
def buildKeyCountVec(data):
    import numpy as np
    keys=np.unique(data)
    bins=keys.searchsorted(data)
    return np.vstack([keys,np.bincount(bins)])

def computeProbVec(kc):
    allFreq=kc[1,:].astype('int').sum()
    prob=kc[1,:].astype('float')/allFreq
    print "[from computeProbVec] prob {0} all freq {1}".format(prob,allFreq)
    return prob

def computeEntropy(prob):
    """
    in: list of prob
    out: float
    usage:
        prob=[0.125, 0.25, 0.125, 0.125, 0.1875, 0.125, 0.0625]
        computeEntropy(prob)
        2.70281953t1114783
    """
    import math
    return sum([-p*math.log(p,2) for p in prob])

def entropyVec(data):
    import math
    kc=buildKeyCountVec(data)
    prob=computeProbVec(kc)
    entropy=computeEntropy(prob)
    return entropy

def splitVec(data,query):
    subData=data[data==query]
    return subData

def split2DVec(data,col,query):
    subData=data[data[:,col]==query]
    return subData


def splitList(data,col,query):
    subData=list()
    for item in data:
        if item[col]==query:
            subData.append(item)
    return subData

def splitListByRow(data,row,query):
    subData=list()
    index=list()
    # to save indices (associative)
    for i,v in enumerate(data[row]):
        if v==query:
            index.append(i)
    # save rows by index
    for j in range(len(data)):
        subDataByRow=list()
        for i in index:
            subDataByRow.append(data[j][i])
        subData.append(subDataByRow)
    return subData

def getInfoGainVec(data):
    import numpy as np
    nFeature=data.shape[1]-1 # except the last column
    InfoGain=np.zeros([nFeature]) # by feature
    for item in range(nFeature):
        InfoGain[item]=computeInfoGain(data[:,[item,-1]])
        print "[getInfoGainVec] {0}th InfoGain={1}".format(item,InfoGain)
    return InfoGain

def computeInfoGain(data):
    di=0 # the first column = data column
    classEntropy=entropyVec(data[:,-1]) # class labels
    kc=buildKeyCountVec(data[:,di])
    nKey=kc.shape[1] #n of unique keys
    prob=computeProbVec(kc)
    rawInfoGain=0.
    for item in range(nKey):
        keyToSearch=kc[0][item]
        subData=split2DVec(data,di,keyToSearch)
        classEntropyByItem=entropyVec(subData[:,-1])
        rawInfoGain+=prob[item]*classEntropyByItem
    InfoGain=classEntropy-rawInfoGain
    return InfoGain

def createTree(keyList,n=0):
    from collections import defaultdict
    nStop=len(keyList)-1
    if n==nStop:
        cLables=keyList[-1]
        print "{0} stopping {1}".format(n,cLables)
        return cLables
    # d grows dynamically
    d=defaultdict()
    parentKeyList=set(keyList[n])
    for key in parentKeyList:
        print "{0} +++> adding {1}".format(n,key)
        subKeyList=splitListByRow(keyList,n,key)
        d.update({key:createTree(subKeyList,n+1)})
    print "{0} ===> after tree created {1}".format(n,d)
    return d

def swapColRow(data):
    dataCol=list()
    for c in range(len(data[0])):
        col=[eachRow[c] for eachRow in data]
        dataCol.append(col)
    return dataCol


def dPrint(d,depth=0):
    depth+=1
    for k, v in d.iteritems():
        print "+"*depth,
        if isinstance(v, dict):
            print "{0} {1}".format(depth,k)
            dPrint(v,depth)
        else:
            print "{0} {1}: {2}".format(depth, k, v)


def main():
    setupDirs()
    setupDt()
    lab()

if __name__=="__main__":
    main()



