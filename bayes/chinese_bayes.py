#coding:utf-8
import csv
import jieba
import jieba.posseg as pseg
import jieba.analyse
import sys

from numpy import *

def textParse(bigString):
    seg_list = jieba.cut(bigString,cut_all = True)
    return list(seg_list)

def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        #else:
          #  print ("the word is not in my vocabulry")
    return returnVec

def createVocabList(dataSet):
    vocabSet=set([])
    for docment in dataSet:
        vocabSet=vocabSet| set(docment) #union of tow sets
    return list(vocabSet) #convet if to list

def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        print child.decode('gbk') # for chinese gbk encode


def train(trainMat,trainGategory):
    numTrain=len(trainMat)
    numWords=len(trainMat[0])  #is vocabulry length
    pAbusive=sum(trainGategory)/float(numTrain)
    p0Num=ones(numWords);p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrain):
        if trainGategory[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom +=sum(trainMat[i])
    p1Vec=log(p1Num/p1Denom)
    p0Vec=log(p0Num/p0Denom)
    return p0Vec,p1Vec,pAbusive

def classfy(vec2classfy,p0Vec,p1Vec,pClass1):
    p1=sum(vec2classfy*p1Vec)+log(pClass1)
    p0=sum(vec2classfy*p0Vec)+log(1-pClass1)
    return [p1,p0]

def spamtest():
    #jieba.enable_parallel(4)
    fullTest = []
    docList = []
    classList = []
    print "start"
    for i in range(1, 41):  # it only 40 doc in every class
        wordList = textParse(open('normal/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = textParse(open('spam/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    print "load over"
    vocabList = createVocabList(docList)  # create vocabulry
    trainSet = range(80)
    testSet = []
    # choose 10 sample to test ,it index of trainMat
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))  # num in 0-79
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        print docIndex
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0, p1, pSpam = train(array(trainMat), array(trainClass))
    print "train finish"
    errCount = 0
    for docIndex in testSet:
        wordVec = bagOfWords2Vec(vocabList, docList[docIndex])
        s=classfy(array(wordVec), p0, p1, pSpam)
        if s[0]>s[1] :
            x=1
        else:
            x=0
        if  x!= classList[docIndex]:
            errCount += 1
            print ("classfication error"), docList[docIndex]

    print ("the error rate is "), float(errCount) / len(testSet)


reload(sys)
sys.setdefaultencoding('utf8')
spamtest()
