# -*- coding: utf-8 -*-
"""
searching zero points of a continuous function 
Created on Sun Sep  1 14:00:00 2019

@author: tm_py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os

#valley function
def valley(a,x):
    val=np.minimum(abs(x)/abs(a),1.0)
    return val

#numrical gradient of function
DELTA_GRADIENT = 1e-4
#DELTA_GRADIENT = 1e-10
def numerical_diff(func,x,h=DELTA_GRADIENT):
    d=func(x+h)-func(x-h)
    d=d/(2*h)
    return d

#function value on random points     
class FuncValByRand:
    def __init__(self,rSeed,nRand,start,ended,func):
        np.random.seed(seed=rSeed)
        r=np.random.rand(nRand)
        rr=start+(ended-start)*r
        val = func(rr)
        se = {'dom':rr,'val':val}
        self.frame = pd.DataFrame(se,columns=['dom','val'])
        
    def getval(self):
        return self.frame

#make data of function value on random points        
def MakeData(rSeed,nRand,start,ended,func,nDisp=0):  
    robj = FuncValByRand(rSeed,nRand,start,ended,func)
    d = robj.getval()
    
    if ( nDisp==1 ):
        d.plot(kind='scatter',x='dom',y='val',marker='.')
        plt.xlim([d['dom'].min(),d['dom'].max()])
        plt.ylim([d['val'].min()-1,d['val'].max()+1])
        plt.show()
    
    return d

#calculate Depth
def CalcDepth(df,dfMax,nDisp=0):
    dfDepthTmp = df
    dfDepthTmp['val'] = (-1)*np.log2(abs(dfDepthTmp['val'])/dfMax)
    dfDepthTmp['val'] = np.ceil(dfDepthTmp['val'])
    dfDepthTmp['val'][dfDepthTmp['val'] < 0] = 0
    dfDepth = dfDepthTmp[dfDepthTmp.val!=float('inf')]
    ExNearZero = dfDepthTmp[dfDepthTmp.val==float('inf')]
    
    if ( nDisp == 1 ):
        dfDepth.plot(kind='scatter',x='dom',y='val',marker='.')
        plt.xlim([dfDepth['dom'].min(),dfDepth['dom'].max()])
        plt.ylim([dfDepth['val'].min()-1,dfDepth['val'].max()+1])
        plt.show()
    
    return dfDepth,ExNearZero

#make bar graph of Depth
def MakeBar(df,nDisp=0,start_t=0,ended_t=0):
    dsort = df.sort_values(by='dom').reset_index(drop=True)
    se = {'dom':dsort.dom,'val':dsort.val}
    dfsort = pd.DataFrame(se,columns=['dom','val'])
    
    start = []
    ended = []
    barval=[]
    
    j=0
    val_tmp = dfsort.val[0]
    barval.append(val_tmp)
    start.append(dfsort.dom[0])
    
    for i in dfsort.index:
        if val_tmp != dfsort.val[i]:
            ended.append(dfsort.dom[i-1])
            j = j+1
            start.append(dfsort.dom[i-1])
            val_tmp = dfsort.val[i]
            barval.append(val_tmp)
    ended.append(dfsort.dom[i])
    
    se = {'start':start,'ended':ended,'bar':barval}
    dfbar = pd.DataFrame(se,columns=['start','ended','bar'])
    if (nDisp == 1):
        grBar_dom=[]
        grBar_val=[]
        dfBarRows = dfbar.size/3
        dfBarIndexes = np.arange(0,dfBarRows,1)
        for idx in dfBarIndexes:
            grBar_dom.append(dfbar.ix[idx]['start'])
            grBar_val.append(0.0)
            grBar_dom.append(dfbar.ix[idx]['start'])
            grBar_val.append(dfbar.ix[idx]['bar'])
            grBar_dom.append(dfbar.ix[idx]['ended'])
            grBar_val.append(dfbar.ix[idx]['bar'])
            grBar_dom.append(dfbar.ix[idx]['ended'])
            grBar_val.append(0.0)
        seBar={'BarDom':grBar_dom,'BarVal':grBar_val}
        dfGrBar=pd.DataFrame(seBar,columns=['BarDom','BarVal'])
        dfGrBar.plot(kind='line',x='BarDom',y='BarVal',marker='',title='Bar graph of Depth ')
        plt.xlim([start_t,ended_t])
        plt.ylim([0,dfGrBar['BarVal'].max()])
        plt.show()
        
    return dfbar

#distribute to Probability
def CalcProb(df):
    subint = (df.ended - df.start)*df.bar
    denom = subint.sum()
    pr = df.bar/denom
    se = {'start':df.start,'ended':df.ended,'Prob':pr}
    prob = pd.DataFrame(se,columns=['start','ended','Prob'])
    return prob

#Kullback-Leibler Divergence between depth distribution and constant
def KLD_DepthConst(df,w):
    dfpos = df.query('Prob != 0')
    epy = (dfpos.ended - dfpos.start)*dfpos.Prob*np.log2(dfpos.Prob/(1/w))
    kld = epy.sum()
    
    return kld

#select neiborhood interval of zero points
def SelNbdZero(df):
    start=[]
    middle=[]
    ended=[]

    bInc=False
    forary = np.arange(1,len(df),1)
    temp=df.Prob[0]
    for i in forary:
        if temp < df.Prob[i]:
            bInc=True
        else:
            if bInc == True:
                start.append(df.start[i-1])
                middle.append(df.ended[i-1])
                ended.append(df.ended[i])
                bInc=False
        temp=df.Prob[i]
    
    se={'start':start,'middle':middle,'ended':ended}
    dfNbdZero=pd.DataFrame(se,columns=['start','middle','ended'])
    return dfNbdZero

#search minima by bracketing    
def searchmin(f,x1,x2,x3):
    #application of bracketing method
    #f(x1)>f(x2),f(x3)>f(x2) , x1 < x2 < x3
    xo1 = x1
    xo2 = x2
    xo3 = x3
        
    r0 = np.random.random_sample()
    r=xo1+(xo3-xo1)*r0
    
    nSearchLimit=0

    #random number is in [0.0, 1.0),so r < x3.
    if ( r == x3 ) or ( (xo3 - xo1) < sys.float_info.epsilon ):
        nSearchLimit=1
        if ( f(xo2) != 0.0 ):
            w=f(xo1)+f(xo3)
            a=f(xo3)/w
            xo2 = a*xo1+(1.0-a)*xo3
        
    if (nSearchLimit != 1):
        #set points for bracketing
        if ( r < x2):
            if ( f(r) <= f(x2) ):
                xo3 = x2
                xo2 = r
            else:
                xo1=r
        elif ( x2 < r ):
            if ( f(x2) <= f(r) ):
                xo3 = r
            else:
                xo1 = x2
                xo2 = r

    rtn = np.zeros(4,dtype='float64')
    rtn[0] = xo1
    rtn[1] = xo2
    rtn[2] = xo3
    rtn[3] = nSearchLimit
    return rtn

#search minima locally  
def LocalSearch(rSeed,nIter,f,x1,x2,x3,nDisp):
    np.random.seed(seed=rSeed)
    nIterTerminate = 0

    for i in np.arange(0,nIter,1):
        func_abs=lambda x:np.abs(f(x))
        rtn = searchmin(func_abs,x1,x2,x3)
        x1 = rtn[0]
        x2 = rtn[1]
        x3 = rtn[2]
        nSearchLimit = int(rtn[3])

        if ( nDisp == 1 ):
            print("neighbor three points_"+str(i+1)+" = ",x1,x2,x3)

        if ( nSearchLimit==1 ):
            break
        
        nIterTerminate = i+1

    return [x2,nIterTerminate]

#learning number of rondom points that Kullback-Leibler Divergence is stable
#Stability is determined by correlation coefficient
def LearnRandNum( rSeed,nRand,nCheckDegree,thlCorre,nRandIterMax,start,ended,InitEpsilon,func,nDisp=0):

    func_abs = lambda x: np.abs(func(x))
    checkArry = np.ones(nCheckDegree)
    checkZero = np.zeros(nCheckDegree)
    
    nall=nRand*nRandIterMax
    allRnd=np.random.rand(nall)
    forIter = np.arange(0,nRandIterMax,1)
    forIter = nRand*forIter

    lhKLDs=[]
    lhr=[]
    lhdom=[]
    lhval=[]
    nloop=0
    rc=0.0
    for fi in forIter:
        if(nDisp==1):
            print('Number of random points : ',fi+nRand)
        lhrap=allRnd[fi:fi+nRand]
        lhr=np.append(lhr,lhrap)
        lhdomap=start+(ended-start)*lhrap
        lhdom=np.append(lhdom,lhdomap)
        lhvalap=func_abs(lhdomap)
        lhval=np.append(lhval,lhvalap)
        se = {'dom':lhdom,'val':lhval}
        data = pd.DataFrame(se,columns=['dom','val'])
        dfDepth,ExNearZero = CalcDepth(data,InitEpsilon)
        dfDepthBar = MakeBar(dfDepth,nDisp,start,ended)
        dfProb = CalcProb(dfDepthBar)
        kld = KLD_DepthConst(dfProb,float(ended-start))
        lhKLDs.append(kld)
        nloop+=1
        if (nloop >= nCheckDegree):
            checkKLD=np.array(lhKLDs[nloop-nCheckDegree:nloop])
            checkKLD=checkKLD-checkKLD.mean()
            if (checkZero.all() == checkKLD.all()):
                if(rc!=0):
                    rc = 1.0
                    break
            else:
                rc=np.dot(checkKLD,checkArry)/(np.linalg.norm(checkKLD)*np.linalg.norm(checkArry))
            if (rc >= thlCorre):
                break

    seKLDs={'RandNum':nRand+np.arange(0,nRand*nloop,nRand),'KLD':lhKLDs}
    dfKLDs = pd.DataFrame(seKLDs,columns=['RandNum','KLD'])
    if (nDisp==1):
        dfKLDs.plot(kind='line',x='RandNum',y='KLD',marker='.',title='Kullback-Leibler of Depth ')
        plt.xlim([dfKLDs['RandNum'].min(),dfKLDs['RandNum'].max()])
        plt.ylim([dfKLDs['KLD'].min(),dfKLDs['KLD'].max()])
        plt.show()
        
    return dfKLDs

#Enumerate candidate zero points   
def CandidateZeroPts( rSeed,nRand,start,ended,InitEpsilon,func,strDir,nIterMax,nSave=0,nDisp=0):
    func_abs = lambda x: np.abs(func(x))    
    data = MakeData(rSeed,nRand,start,ended,func_abs,nDisp)
    dfDepth,ExNearZero = CalcDepth(data,InitEpsilon,nDisp)
    if (nSave == 1):
        dfDepth.to_csv(strDir+"\dfDepth_"+str(start)+"_"+str(ended)+"_"+str(nRand)+".csv",index=False)
    if len(ExNearZero) > 0:
        print('extreamly near Zero points = ',ExNearZero['dom'])

    dfDepthBar = MakeBar(dfDepth,nDisp,start,ended)
    if (nSave == 1):
        dfDepthBar.to_csv(strDir+"\dfDepthBar_"+str(start)+"_"+str(ended)+"_"+str(nRand)+".csv",index=False)
    
    dfProb = CalcProb(dfDepthBar)

    dfNbdZero=SelNbdZero(dfProb)
 
    forary = np.arange(0,len(dfNbdZero),1)

    estZeros=[]
    estValues=[]
    estIters=[]
    for i in forary:
        x1=dfNbdZero.start[i]
        x2=dfNbdZero.middle[i]
        x3=dfNbdZero.ended[i]
        if ( nDisp == 1 ):
            print("---- LocalSearch at " + str(i+1)+ "-th neighborhood ----")
        xzero=LocalSearch(rSeed,nIterMax,func,x1,x2,x3,nDisp)
        estZeros.append(xzero[0])
        estValues.append(func(xzero[0]))
        estIters.append(xzero[1])
    
    for ie in ExNearZero['dom']:
        estZeros.append(ie)
        estValues.append(func(ie))
        estIters.append(0)

    seEst={'EstimateZeroPt':estZeros,'FunctionValue':estValues,'Iteration':estIters} 
    dfEst = pd.DataFrame(seEst,columns=['EstimateZeroPt','FunctionValue','Iteration'])
    if (nSave == 1):
        dfEst.to_csv(strDir+"\dfEst_"+str(start)+"_"+str(ended)+"_"+str(nRand)+".csv",index=False)

    return dfEst

#search zero points by depth method
def SearchZeroPtByDepth(func,func_name,start,ended,rSeed=0,InitRandNum=10,nRandIterMax=200,nCheckStabilityDegree=3,thlCorre=0.5,LocalIter=500,nSave=0,nDisp=0):
    strDir = os.getcwd()
    strDir = strDir + '\\..\\data\\'
    strDirMk = strDir + func_name
    if ( nSave==1 ) and ( os.path.exists(strDirMk)==False):
        os.makedirs(strDirMk)
    
    #Get the base value of depth
    np.random.seed(seed=rSeed)
    r=np.random.rand(InitRandNum)
    rr = start + (ended - start)*r
    f_abs=np.abs(func(rr))
    InitEpsilon=f_abs.max()

    #learn "good number of Random Points"
    dfKLDs = LearnRandNum(rSeed,InitRandNum,nCheckStabilityDegree,thlCorre,nRandIterMax,start,ended,InitEpsilon,func,nDisp)
    RandNum=dfKLDs['RandNum'].max()
    print('Good Number of Random Points : ',RandNum)
    
    #Candidate Zero Points by using "good number of Random Points"
    est_hist=[]
    est = CandidateZeroPts( rSeed,RandNum,start,ended,InitEpsilon,func,strDirMk,LocalIter,nSave, nDisp )
    est_hist.append(est)
       
    return est_hist,RandNum
