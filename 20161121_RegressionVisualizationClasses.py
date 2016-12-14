import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sympy import Symbol, Float
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score

def timeElapsed(startTime):
    time = round((datetime.now()-startTime).total_seconds()/60,3)
    return time

def prepString(badString):
    return ''.join(e for e in badString if e.isalnum())

class createRegModel:
    def __init__(self, shipName, equipName, yLabel, featLabels):
        self.shipName = shipName
        self.equipName = equipName
        self.yLabel = yLabel
        self.featLabels = featLabels
        self.allLabels = list([self.yLabel]) + self.featLabels
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None
        self.yPredicts = None
        self.yResids = None
        self.yCorrected = None
        self.model = None
        self.equation = None
        self.correctedEquation = None
        self.metrics = {}
    def constructXYTrainTest(self, train, test):
        timerTrainTest = datetime.now()
        trainFeat = train[self.allLabels].dropna()
        testFeat = test[self.allLabels].dropna()
        self.xTrain = trainFeat[self.featLabels]
        self.xTest = testFeat[self.featLabels]
        self.yTrain = trainFeat[self.yLabel]
        self.yTest = testFeat[self.yLabel]
        return timeElapsed(timerTrainTest)
    def trainModel(self):
        timerModel = datetime.now()
        self.model = RidgeCV(alphas = np.logspace(-3,3,num=50), normalize = True)         
        self.model.fit(self.xTrain, self.yTrain)
        return timeElapsed(timerModel)
    def makePredictions(self):
        timerPredict = datetime.now()
        self.yPredicts = self.model.predict(self.xTest)
        self.yResids = self.yTest - self.yPredicts       
        return timeElapsed(timerPredict)
    def calcCorrectedY(self):
        timerCorrect = datetime.now()
        self.yCorrected = self.yTest.copy()
        for ii, coef in enumerate(self.model.coef_[1:]):
            self.yCorrected -= coef*self.xTest[self.featLabels[ii+1]]
        return timeElapsed(timerCorrect)
    def buildEquation(self, rd):
        timerBuild = datetime.now()
        ySym = Symbol(prepString(self.yLabel))
        primSym = Symbol(prepString(self.featLabels[0]))
        secSym = []
        primEq = Float(self.model.intercept_,rd) + Float(self.model.coef_[0],rd)*primSym
        secEq = 0
        #Build the secondary factor equation
        for ii, coef in enumerate(self.model.coef_[1:]):
            newSym = Symbol(self.featLabels[ii+1].replace(' ',''))
            secSym.append(newSym)
            secEq += Float(coef,rd)*secSym[-1]
        self.equation = 'Value={}'.format(str(primEq + secEq))
        self.correctedEquation = 'Value={}'.format(str(ySym - secEq))
        return timeElapsed(timerBuild)
    def calcMetrics(self, rd, warn, alarm):
        timerMetrics = datetime.now()
        yResidsStd = self.yResids.std()
        self.metrics['rSquared'] = round(r2_score(self.yTest, self.yPredicts),rd)
        self.metrics['meanAbsEr'] = round(mean_absolute_error(self.yTest, self.yPredicts),rd)
        self.metrics['explVar'] = round(explained_variance_score(self.yTest, self.yPredicts),rd)
        self.metrics['warnLim'] = round(warn * yResidsStd,rd)
        self.metrics['alarmLim'] = round(alarm * yResidsStd,rd)
        return timeElapsed(timerMetrics)
    def filterResids(self,filtResid, medWin):
        timerFilter = datetime.now()
        if filtResid:
            self.yResids = self.yResids.rolling(window=medWin).median()
            self.yResids[0:medWin+1] = 0
        return timeElapsed(timerFilter)

class plotRegression:
    def __init__(self, parentFeat,shipName, equipName):
        #Labeling
        self.parentFeat = parentFeat
        self.yLabel = parentFeat.yLabel
        self.yTitle = prepString(self.yLabel)
        self.featLabels = parentFeat.featLabels
        self.primLabel = self.featLabels[0]                    
        #Data (references, not copies)
        self.primData = parentFeat.xTest[[self.primLabel]]
        self.yResids = parentFeat.yResids
        self.yPredicts = parentFeat.yPredicts
        self.yTest = parentFeat.yTest
        self.yCorrected = parentFeat.yCorrected
        #Metrics
        self.rSquared = parentFeat.metrics['rSquared']
        self.meanAbsEr = parentFeat.metrics['meanAbsEr']
        self.warnLim = parentFeat.metrics['warnLim']
        self.alarmLim = parentFeat.metrics['alarmLim']  
        #Summary
        self.plotTitle = 'BL_{}_{}_{}(R{:.4f})vs{}'.format(shipName, equipName, self.yTitle, 
                                                        self.rSquared, str(self.featLabels).replace(' ',''))
        self.eqTitle = '\n{}'.format(parentFeat.equation)
    def buildPlots(self, figSize):
        timerPlots = datetime.now()
        f,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2,4,figsize=figSize)        
        f.suptitle(self.plotTitle + self.eqTitle, fontsize=16)
        self.buildResiduals(ax1,range(len(self.yResids)),'Index')
        self.buildResiduals(ax2,self.primData.values,self.primLabel)
        self.buildYYhat(ax3)
        self.buildHistogram(ax4,self.yResids,'Residuals')
        self.buildScatter(ax5,self.primData,self.yTest,self.primLabel,self.yLabel,'Actual')
        self.buildScatter(ax6,self.primData,self.yCorrected,self.primLabel,self.yLabel,'Corrected')
        self.buildHistogram(ax7,self.yTest,'Actual {}'.format(self.yLabel))
        self.buildHistogram(ax8,self.yCorrected,'Corrected {}'.format(self.yLabel))
        return timeElapsed(timerPlots)
    def buildResiduals(self, curAx, x, xLab):       
        curAx.scatter(x,self.yResids)
        curAx.set_title('Residuals vs. {}'.format(xLab))
        curAx.set_ylabel('Residual Value')
        curAx.set_xlabel(xLab)
        minVal, maxVal = min(x), max(x)
        self.plotLimits(curAx,minVal,maxVal)
    def buildYYhat(self,curAx):
        absmin = min([self.yTest.min(),self.yPredicts.min()])
        absmax = max([self.yTest.max(),self.yPredicts.max()])
        curAx.scatter(self.yTest, self.yPredicts)
        self.plotTextBox(curAx,'$MAE=%.3f$\n$R2=%.3f$' % (self.meanAbsEr, self.rSquared))
        curAx.set_title('Predicted {} vs. Actual {}'.format(self.yLabel,self.yLabel))
        curAx.axis([absmin, absmax, absmin, absmax])
        curAx.plot([absmin, absmax], [absmin, absmax],c="k")
        curAx.set_ylabel('Predicted {}'.format(self.yLabel))
        curAx.set_xlabel('Actual {}'.format(self.yLabel))  
    def buildHistogram(self, curAx,values,label):
        sns.distplot(values,ax=curAx,kde=False, fit=stats.norm) 
        self.plotTextBox(curAx,'$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' %
                               (values.mean(), values.median(), values.std()))
        curAx.set_title('Histogram of {}'.format(label))
        curAx.set_ylabel('# Occurences')
        curAx.set_xlabel('{} Value'.format(label))
    def buildScatter(self, curAx,x,y,xlabel,ylabel,typelabel):
        curAx.scatter(x,y)
        curAx.set_title('{} {} vs. {}'.format(typelabel, ylabel, xlabel))
        curAx.set_ylabel('{} {}'.format(typelabel, ylabel))
        curAx.set_xlabel(xlabel)     
    def plotLimits(self, curAx, minVal, maxVal):
        curAx.plot([minVal, maxVal],[0,0],c='g')
        curAx.plot([minVal, maxVal],[self.warnLim,self.warnLim],c='y')
        curAx.plot([minVal, maxVal],[-self.warnLim,-self.warnLim],c='y')
        curAx.plot([minVal, maxVal],[self.alarmLim,self.alarmLim],c='r')
        curAx.plot([minVal, maxVal],[-self.alarmLim,-self.alarmLim],c='r')
    def plotTextBox(self, curAx,textStr):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        curAx.text(0.05, 0.95, textStr, transform=curAx.transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)