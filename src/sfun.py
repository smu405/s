# functions from *.ipynb

# directory setup
import os
myhome=os.path.expanduser('~')
mywd=os.path.join(myhome,'Code/git/jsl/algo/src/pystat/')
plantdir=os.path.join(myhome,'Code/git/gh/s/lib/')
ch2wd=os.path.join(myhome,'Code/git/else/machinelearninginaction/Ch02/')
ch2dd=os.path.join(ch2wd,'trainingDigits')

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
# input: directory path
# output: list of absolute file names
# usage: trainFiles=getFileList(ch2dd)
def getFileList(d):
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
# myX=np.array([0,0])
# normDistances=getDistance("norm",myV,myX)
# euclDistances=getDistance("eucl",myV,myX)
# mathDistances=getDistance("math",myV,myX)
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


