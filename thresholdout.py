# Experiments for Thresholdhout
# Fast implementation of Thresholdout specific to the experiment.
# Thresholdout with threshold = 4/sqrt(n), tolerance = 1/sqrt(n)
# Signal: 20 variables with 6/sqrt(n) bias toward the label

import numpy as np
import matplotlib
import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def createnosignaldata(n,d):
    """
    Data points are random Gaussian vectors.
    Class labels are random and uniform
    """
    X_train = np.random.normal(0,1,(n,d+1))
    X_train[:,d] = np.sign(X_train[:,d])
    X_holdout = np.random.normal(0,1,(n,d+1))
    X_holdout[:,d] = np.sign(X_holdout[:,d])
    X_test = np.random.normal(0,1,(n,d+1))
    X_test[:,d] = np.sign(X_test[:,d])
    return X_train, X_holdout, X_test


def createhighsignaldata(n,d):
    """
    Data points are random Gaussian vectors.
    Class labels are random and uniform
    First nbiased are biased with bias towards the class label
    """
    X_train = np.random.normal(0,1,(n,d+1))
    X_train[:,d] = np.sign(X_train[:,d])
    X_holdout = np.random.normal(0,1,(n,d+1))
    X_holdout[:,d] = np.sign(X_holdout[:,d])
    X_test = np.random.normal(0,1,(n,d+1))
    X_test[:,d] = np.sign(X_test[:,d])

    # Add correlation with the sign
    nbiased = 20
    bias = 6.0/np.sqrt(n)
    b = np.zeros(nbiased)
    for i in xrange(n):
        b[0:nbiased] = bias*X_holdout[i,d]
        X_holdout[i,range(0,nbiased)] = np.add(X_holdout[i,range(0,nbiased)], b)

    for i in xrange(n):
        b[0:nbiased] = bias*X_train[i,d]
        X_train[i,range(0,nbiased)] = np.add(X_train[i,range(0,nbiased)], b)

    for i in xrange(n):
        b[0:nbiased] = bias*X_test[i,d]
        X_test[i,range(0,nbiased)] = np.add(X_test[i,range(0,nbiased)], b)

    return X_train, X_holdout, X_test

def runClassifier(n,d,krange, X_train,X_holdout,X_test):
    """
    Variable selection and basic boosting on synthetic data. Variables
    with largest correlation with target are selected first.
    """
    # Compute values on the standard holdout
    tolerance = 1.0/np.sqrt(n)
    threshold = 4.0/np.sqrt(n)

    vals = []
    trainanswers = np.dot(X_train[:,xrange(0,d)].T,X_train[:,d])/n
    holdoutanswers = np.dot(X_holdout[:,xrange(0,d)].T,X_holdout[:,d])/n
    trainpos = trainanswers > 1.0/np.sqrt(n)
    holdopos = holdoutanswers > 1.0/np.sqrt(n)
    trainneg = trainanswers < -1.0/np.sqrt(n)
    holdoneg = holdoutanswers < -1.0/np.sqrt(n)
    selected = (trainpos & holdopos) | (trainneg & holdoneg)
    trainanswers[~selected] = 0
    sortanswers = np.abs(trainanswers).argsort()
    for k in krange:
        weights = np.zeros(d+1)
        topk = sortanswers[-k:]
        weights[topk] = np.sign(trainanswers[topk])
        ftrain = 1.0*np.count_nonzero(np.sign(np.dot(X_train,weights)) == X_train[:,d])/n
        fholdout = 1.0*np.count_nonzero(np.sign(np.dot(X_holdout,weights)) == X_holdout[:,d])/n
        ftest = 1.0*np.count_nonzero(np.sign(np.dot(X_test,weights)) == X_test[:,d])/n
        if k == 0:
            vals.append([0.5,0.5,0.5])
        else:
            vals.append([ftrain,fholdout,ftest])

    # Compute values using Thresholdout
    noisy_vals = []
    trainanswers = np.dot(X_train[:,xrange(0,d)].T,X_train[:,d])/n
    holdoutanswers = np.dot(X_holdout[:,xrange(0,d)].T,X_holdout[:,d])/n
    diffs = np.abs(trainanswers - holdoutanswers)
    noise = np.random.normal(0,tolerance,d)
    abovethr = diffs > threshold + noise
    holdoutanswers[~abovethr] = trainanswers[~abovethr]
    holdoutanswers[abovethr] = (holdoutanswers+np.random.normal(0,tolerance,d))[abovethr]
    trainpos = trainanswers > 1.0/np.sqrt(n)
    holdopos = holdoutanswers > 1.0/np.sqrt(n)
    trainneg = trainanswers < -1.0/np.sqrt(n)
    holdoneg = holdoutanswers < -1.0/np.sqrt(n)
    selected = (trainpos & holdopos) | (trainneg & holdoneg)
    trainanswers[~selected] = 0
    sortanswers = np.abs(trainanswers).argsort()
    for k in krange:
        weights = np.zeros(d+1)
        topk = sortanswers[-k:]
        weights[topk] = np.sign(trainanswers[topk])
        ftrain = 1.0*np.count_nonzero(np.sign(np.dot(X_train,weights)) == X_train[:,d])/n
        fholdout = 1.0*np.count_nonzero(np.sign(np.dot(X_holdout,weights)) == X_holdout[:,d])/n
        if abs(ftrain-fholdout) < threshold + np.random.normal(0,tolerance):
            fholdout = ftrain
        else:
            fholdout += np.random.normal(0,tolerance)
        ftest = 1.0*np.count_nonzero(np.sign(np.dot(X_test,weights)) == X_test[:,d])/n
        if k == 0:
            noisy_vals.append([0.5,0.5,0.5])
        else:
            noisy_vals.append([ftrain,fholdout,ftest])

    return vals, noisy_vals

def plot1(x,mean,std,plotname,plottitle,legend_pos=2):
    fig = plt.figure(1,figsize=(6.5,4))
    plt.title(plottitle,fontsize='14')
    plt.ylabel('accuracy',fontsize='14')
    plt.xlabel('number of variables',fontsize='14')
    plt.axis([x[0], x[-1], 0.45, 0.75])
    plt.plot(x,mean[:,0],'b^-',label='training')
    plt.fill_between(x,mean[:,0]-std[:,0],mean[:,0]+std[:,0],alpha=0.5,edgecolor='#B2B2F5',facecolor='#B2B2F5',linestyle='dashdot')
    plt.plot(x,mean[:,1],'x-',label='holdout',color='#006600')
    plt.fill_between(x,mean[:,1]-std[:,1],mean[:,1]+std[:,1],alpha=0.5,edgecolor='#33CC33',facecolor='#CCFFCC',linestyle='dashdot')
    plt.plot(x,mean[:,2],'r|-',label='fresh')
    plt.fill_between(x,mean[:,2]-std[:,2],mean[:,2]+std[:,2],alpha=0.5,edgecolor='#CC4F1B',facecolor='#FF9848',linestyle='dashdot')
    plt.legend(loc=legend_pos,prop={'size':12})
    plt.tight_layout()
    plt.savefig(plotname+".pdf")
    plt.close()

def avgout(vallist):
    """ entry-wise average of a list of matrices """
    r = len(vallist)
    A = 0
    if r > 0:
        for B in vallist:
            A += (1.0/r) * B
    return A


def stddev(vallist):
    """ entry-wise standard deviation of a list of matrices """
    r = len(vallist)
    mean = avgout(vallist)
    A = 0
    if r > 0:
        for B in vallist:
            A += (1.0/r) * (B - mean)**2
    return np.sqrt(A)


def repeatexp(n,d,krange,reps,datafn):
    """ Repeat experiment specified by fn for reps steps """
    vallist = []
    vallist2 = []
    for r in range(0,reps):
        print "Repetition:", r
        X_train,X_holdout,X_test = datafn(n,d)
        vals,vals2 = runClassifier(n,d,krange,X_train,X_holdout,X_test)
        vallist.append(np.array(vals))
        vallist2.append(np.array(vals2))
    return vallist, vallist2


def runandplotsummary(n,d,krange,reps,datafn,plotname):
    vallist,vallist2 = repeatexp(n,d,krange,reps,datafn)
    mean = avgout(vallist)
    std = stddev(vallist)
    mean2 = avgout(vallist2)
    std2 = stddev(vallist2)
    f = open(plotname+".txt",'w')
    f.write(str(mean))
    f.write("\n")
    f.write(str(std))
    f.write("\n")
    f.write(str(mean2))
    f.write("\n")
    f.write(str(std2))
    f.close()
    plot1(krange,mean,std,plotname+"-std","Standard holdout")
    plot1(krange,mean2,std2,plotname+"-thr","Thresholdout")


reps = 100
n, d = 10000, 10000
krange = [0,10,20,30,45,70,100,150,200,250,300,400,500]

today = datetime.datetime.now()
timestamp = str(today.month)+str(today.day)+"."+str(today.hour)+str(today.minute)


# Experiment 1:
# No correlations
plotname = "plot-"+timestamp+"-"+str(n)+"-"+str(d)+"-"+str(reps)+"-nosignal"
runandplotsummary(n,d,krange,reps,createnosignaldata,plotname)



# Experiment 2:
# Some variables are correlated
plotname = "plot-"+timestamp+"-"+str(n)+"-"+str(d)+"-"+str(reps)+"-highsignal"
runandplotsummary(n,d,krange,reps,createhighsignaldata,plotname)
