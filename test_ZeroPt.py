# -*- coding: utf-8 -*-
"""
test of searching zero points of a continuous function 
Created on Sun Sep  1 14:00:00 2019

@author: tm_py
"""

import numpy as np
import pandas as pd
from scipy import special
from decimal import Decimal
import matplotlib.pyplot as plt

import time
import datetime
from DepthMethod import SearchZeroPtByDepth,valley

#Printing Parameters
def PrintPara(inPara):
        for k,v in inPara.items():
            print(k,' = ', v)  

#check threshold
def CheckThreshold(estZero,threshold):
    blnExist = False
    for estfv in estZero.FunctionValue:
        if estfv < threshold:
            blnExist = True
            
    return  blnExist

def ZeroFunc(x):
    rtn = np.zeros(x.shape)
    return rtn


#definisiton of Hyper Parameters(only one set)
HyperParam = dict(seed=0,randInitNum=10,randIterMax=100,checkStabilityDegree=5,ThresholdCorre=0.8,localIterMax=500)
#HyperParam = dict(seed=0,randInitNum=10,randIterMax=200,checkStabilityDegree=3,ThresholdCorre=0.5,localIterMax=500)
#HyperParam = dict(seed=0,randInitNum=10,randIterMax=600,checkStabilityDegree=3,ThresholdCorre=0.5,localIterMax=500)
#HyperParam = dict(seed=1,randInitNum=10,randIterMax=600,checkStabilityDegree=3,ThresholdCorre=0.5,localIterMax=500)
print('\n----- Hyper Parameters -----')
PrintPara(HyperParam)

#definition of OPtions(only one set)
#Option = dict(disp=1,save=0,thld=1.0E-14,checkresultNum=10000)
Option = dict(disp=0,save=1,thld=1.0E-14,checkresultNum=10000)
print('\n----- Options -----')
PrintPara(Option)

#dictionary of functions(multiple set OK)
FuncInfo = {}
FuncInfo['(x-5.0)**2'] = dict(definition=lambda x:(x-5.0)**2,name='poly0')
FuncInfo['(x-1.0)*(x-8.0)'] = dict(definition=lambda x:(x-1.0)*(x-8.0),name='poly1')
FuncInfo['(x-1.1)*(x-8.9)'] = dict(definition=lambda x:(x-1.1)*(x-8.9),name='poly2')
FuncInfo['x*(x-1.5)*(x-8.5)'] = dict(definition=lambda x:x*(x-1.5)*(x-8.5),name='poly3')
FuncInfo['(x-1.5)**2*(x-8.5)'] = dict(definition=lambda x:(x-1.5)**2*(x-8.5),name='poly4')
FuncInfo['x**2-2'] = dict(definition=lambda x:x**2-2,name='poly5')
FuncInfo['x**2-5'] = dict(definition=lambda x:x**2-5,name='poly6')
FuncInfo['x**3'] = dict(definition=lambda x:x**3,name='poly7')
FuncInfo['(x-2.0)*x*(x+2.0)**2'] = dict(definition=lambda x:(x-2.0)*x*(x+2.0)**2,name='poly8')
FuncInfo['|(x-1.0)*(x-8.0)|'] = dict(definition=lambda x:np.abs((x-1.0)*(x-8.0)),name='abs1')
FuncInfo['|x*(x-1.5)*(x-8.5)|'] = dict(definition=lambda x:np.abs(x*(x-1.5)*(x-8.5)),name='abs2')
FuncInfo['sin(x)'] = dict(definition=lambda x:np.sin(x),name='sin_x')
FuncInfo['cos(x)'] = dict(definition=lambda x:np.cos(x),name='cos_x')
FuncInfo['tan(x)'] = dict(definition=lambda x:np.tan(x),name='tan_x')
FuncInfo['jv(0,x)'] = dict(definition=lambda x : special.jv(0,x),name='jv0_x')
FuncInfo['jv(1,x)'] = dict(definition=lambda x : special.jv(1,x),name='jv1_x')
FuncInfo['valley(2,x)'] = dict(definition=lambda x : valley(2.0,x),name='valley2')
FuncInfo['valley(1,x)'] = dict(definition=lambda x : valley(1.0,x),name='valley1')
FuncInfo['valley(0.5,x)'] = dict(definition=lambda x : valley(0.5,x),name='valley05')
FuncInfo['valley(0.1,x)'] = dict(definition=lambda x : valley(0.1,x),name='valley01')
FuncInfo['valley(0.01,x)'] = dict(definition=lambda x : valley(0.01,x),name='valley001')
FuncInfo['x**2+1.0'] = dict(definition=lambda x : x**2 + 1.0,name='NoZero')

#dictionary of domains(multiple set OK)
DomainInfo = {}
#DomainInfo['(-1,1)'] = dict(x0=-1.0,x1=1.0)
#DomainInfo['(-3,3)'] = dict(x0=-2.5,x1=2.5)
#DomainInfo['(-5,5)'] = dict(x0=-5.0,x1=5.0)
DomainInfo['(-10,10)'] = dict(x0=-10.0,x1=10.0)
#DomainInfo['(-20,20)'] = dict(x0=-20.0,x1=20.0)
#DomainInfo['(-30,30)'] = dict(x0=-30.0,x1=30.0)

#input data
In_dict = dict(Func = FuncInfo,Domain = DomainInfo)

#buffer for result
col_func_disp=[]
col_func_def=[]
col_Interval=[]
col_seed=[]
col_Rand=[]
col_Index=[]
col_ezero=[]
col_valatzero=[]
col_Iteration=[]

rSeed = HyperParam['seed']
rdNum = HyperParam['randInitNum']
rdIterMax = HyperParam['randIterMax']
nStabilityDegree = HyperParam['checkStabilityDegree']
thlCorre = HyperParam['ThresholdCorre']
nIMax = HyperParam['localIterMax']
nDisp = Option['disp']
nSave = Option['save']
threshold = Option['thld']
nCheckResult = Option['checkresultNum']

for kf,vf in In_dict['Func'].items():
    for kd,vd in In_dict['Domain'].items():
        print('\n----- Searching Zero Points -----')
        print(kf,' on ',kd)
        funcdef = vf['definition']
        funcname = vf['name']
        start = vd['x0']
        ended = vd['x1']
        
        ts = time.time()    #start time
        
        #Search Zero Points
        estZero,rdGoodNum=SearchZeroPtByDepth(funcdef,funcname,start,ended,
                                              rSeed,rdNum,rdIterMax,
                                              nStabilityDegree,thlCorre,nIMax,
                                              nSave,nDisp)
        
        td = time.time() - ts   #proc time
        print('\nproc time = ',1000*td,'[ms] \n')
        
        #check threshold
        blnExist = True
        if (threshold > 0.0):
            blnExist = CheckThreshold(estZero[0],threshold)

        #preperation of making table 
        if ( (len(estZero[0]) > 0) and (blnExist == True)):
            ezero=[]
            valatzero=[]
            for i in np.arange(0,len(estZero[0]),1):
                ez=Decimal.from_float(np.around(estZero[0].EstimateZeroPt[i],16))
                fv=Decimal.from_float(np.around(estZero[0].FunctionValue[i],16))
                it=int(estZero[0].Iteration[i])
                print(kf,'{:.16E}'.format(ez),'{:.16E}'.format(fv),it)
                col_func_disp.append(funcname)
                col_func_def.append(kf)
                col_Interval.append("("+str(start)+","+str(ended)+")")
                col_seed.append(rSeed)
                col_Rand.append(rdGoodNum)
                col_Index.append(i)
                col_ezero.append('{:.16E}'.format(ez))
                col_valatzero.append('{:.16E}'.format(fv))
                col_Iteration.append(it)
                ezero.append(ez)
                valatzero.append(fv)

            #graph of result
            step = (ended-start)/nCheckResult
            xf = np.arange(start = start,stop = ended,step = step)
            yf = funcdef(xf)
            y0 = ZeroFunc(xf)
            plt.title(' Function values and Candidate Zero points ')
            plt.plot(xf, yf, marker="", color = "blue", linestyle = "-")
            #plt.plot(xf, y0, marker="", color = "green", linestyle = "--")
            plt.plot(ezero, valatzero, marker="o",color = "red" , linestyle = "" )
            plt.show()

        else:
            print(kf,"Can't estimate zero points")
            col_func_disp.append(funcname)
            col_func_def.append(kf)
            col_Interval.append("("+str(start)+","+str(ended)+")")
            col_seed.append(rSeed)
            col_Rand.append(rdGoodNum)
            col_Index.append("Nan")
            col_ezero.append("Nan")
            col_valatzero.append("Nan")
            col_Iteration.append("Nan")
        
#DataFrame for save as html/tex
arrays = [col_func_disp,col_func_def,col_Interval,col_seed,col_Rand,col_Index]
index = pd.MultiIndex.from_arrays(arrays,names=['Function','Definition','Domain','Seed','RandNum','Index'])
dfTestZeroPt = pd.DataFrame({'Candidate Zeros':col_ezero,'Value at Candidate Zero':col_valatzero,'Local Iteration':col_Iteration},index=index)

#day time
now=datetime.datetime.now()
#nowstr=now.strftime("%Y-%m-%d-%H-%M-%S")
nowstr=now.strftime("%Y-%m-%d")
#output to html
print(dfTestZeroPt.to_html('testZeroPt'+nowstr+'.html'))

