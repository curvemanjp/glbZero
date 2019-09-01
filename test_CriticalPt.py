# -*- coding: utf-8 -*-
"""
test of searching critical points of a differential function 
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
from DepthMethod import SearchZeroPtByDepth,numerical_diff

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
#HyperParam = dict(seed=0,randInitNum=10,randIterMax=200,checkStabilityDegree=5,ThresholdCorre=0.8,localIterMax=500)
#HyperParam = dict(seed=0,randInitNum=10,randIterMax=200,checkStabilityDegree=3,ThresholdCorre=0.5,localIterMax=500)
#HyperParam = dict(seed=0,randInitNum=10,randIterMax=600,checkStabilityDegree=3,ThresholdCorre=0.5,localIterMax=500)
HyperParam = dict(seed=1,randInitNum=10,randIterMax=600,checkStabilityDegree=3,ThresholdCorre=0.5,localIterMax=500)
print('\n----- Hyper Parameters -----')
PrintPara(HyperParam)

#definition of OPtions(only one set)
#Option = dict(disp=1,save=0,thld=1.0E-14,checkresultNum=10000)
Option = dict(disp=0,save=0,thld=1.0E-14,checkresultNum=10000)
print('\n----- Options -----')
PrintPara(Option)

#definition of functions(multiple OK)
#fprimitive=lambda x:((1-np.sign(x))/2)*(-1/3*x**3) + ((1+np.sign(x))/2)*(1/2*x**2)
#fprimitive=lambda x:(x+5.0)*(x-8.0)
fprimitive=lambda x:(x-2.0)*x*(x+2.0)**2
#fprimitive=lambda x:x**3*(x+12.0)*(x-12.0)
#fprimitive=lambda x:np.sin(x)
#fprimitive=lambda x:special.jv(0,x)
#fprimitive=lambda x:special.jv(1,x)

g=lambda x:numerical_diff(fprimitive,x,1e-10)

#dictionary of functions
FuncInfo = {}
#FuncInfo['-1/3*x**3 if (x<0) else 1/2*x**2'] = dict(definition=g,name='combo')
#FuncInfo['(x+5.0)*(x-8.0)'] = dict(definition=g,name='poly1')
FuncInfo['(x-2.0)*x*(x+2.0)**2'] = dict(definition=g,name='poly2')
#FuncInfo['x**3*(x+12.0)*(x-12.0)'] = dict(definition=g,name='poly3')
#FuncInfo['sin(x)'] = dict(definition=g,name='sin')
#FuncInfo['jv(0,x)'] = dict(definition=g,name='jv(0,x)')
#FuncInfo['jv(1,x)'] = dict(definition=g,name='jv(1,x)')
#FuncInfo['jv(1,x)'] = dict(definition=g,name='jv(1,x)')
##FuncInfo['jv(1,x)'] = dict(definition=g,name='jv(1,x)')

#dictionary of domains(Multiple OK)
DomainInfo = {}
#DomainInfo['(-3,3)'] = dict(x0=-3.0,x1=3.0)
DomainInfo['(-3,2)'] = dict(x0=-2.5,x1=2.0)
DomainInfo['(-5,5)'] = dict(x0=-5.0,x1=5.0)
#DomainInfo['(-10,10)'] = dict(x0=-10.0,x1=10.0)
#DomainInfo['(-12,12)'] = dict(x0=-12.0,x1=12.0)
#DomainInfo['(-20,20)'] = dict(x0=-20.0,x1=20.0)
#DomainInfo['(-30,30)'] = dict(x0=-30.0,x1=30.0)
#DomainInfo['(-30,30)'] = dict(x0=-100.0,x1=100.0)

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
col_primitveValatZero=[]

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
            primitiveval=[]
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
                col_primitveValatZero.append(fprimitive(estZero[0].EstimateZeroPt[i]))
                ezero.append(ez)
                valatzero.append(fv)
                primitiveval.append(fprimitive(estZero[0].EstimateZeroPt[i]))

            #graph of result
            step = (ended-start)/nCheckResult
            xf = np.arange(start = start,stop = ended,step = step)
            yf = funcdef(xf)
            #y0 = ZeroFunc(xf)
            plt.title(' Differential values and Candidate Zero points ')
            plt.plot(xf, yf, marker="", color = "blue", linestyle = "-")
            plt.plot(ezero, valatzero, marker="o",color = "red" , linestyle = "" )
            plt.show()

            ezero.append(start)
            ezero.append(ended)
            primitiveval.append(fprimitive(start))
            primitiveval.append(fprimitive(ended))
            glob_min_pt = ezero[np.argmin(primitiveval)]
            glob_min_val = np.min(primitiveval)          
            glob_max_pt = ezero[np.argmax(primitiveval)]
            glob_max_val = np.max(primitiveval)
            glob_lminmax_pt = [glob_min_pt,glob_max_pt]
            glob_lminmax_val = [glob_min_val,glob_max_val]
            
            print("")
            print("global minimum = ",'{:.16E}'.format(glob_min_pt),'{:.16E}'.format(glob_min_val))
            print("global maximum = ",'{:.16E}'.format(glob_max_pt),'{:.16E}'.format(glob_max_val))
            
            plt.title(' Function values and Candidate critical points\n(including boundary points) ')
            yfprim=fprimitive(xf)
            plt.plot(xf, yfprim, marker="", color = "blue", linestyle = "-")
            plt.plot(ezero, primitiveval, marker="o",color = "green" , linestyle = "" )
            plt.plot(glob_lminmax_pt, glob_lminmax_val, marker="o",color = "red" , linestyle = "" )
            
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
dfTestZeroPt = pd.DataFrame({'critical points':col_ezero,'critical values':col_primitveValatZero,'Local Iteration':col_Iteration},index=index)

#day time
now=datetime.datetime.now()
#nowstr=now.strftime("%Y-%m-%d-%H-%M-%S")
nowstr=now.strftime("%Y-%m-%d")
#output to html
print(dfTestZeroPt.to_html('testlminmaxPt'+nowstr+'.html'))

