import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import datetime
import re
import glob
import matplotlib.dates as dates
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def BuoysFrame(path):
    '''
    Create DataFrame from txt file with multiindex columns: buoyID and time.
    
    path - string containing a path specification with all the downloaded NDBC files, 
    where name of files have to contain 'stdmet_*', * being buoy's ID.
    Example of file name: NDBC_historical_stdmet_41044_2020.txt (ID 41044).
    '''
    buoys=[]
    for file in glob.glob(path):
        f=pd.read_table(file,delim_whitespace=True,dtype='unicode')
        #search buoyID in name
        b = re.search(r'stdmet_(\d+)',file)
        if b is not None:
            b=b.group(1)
        else:
            print("%s is not a valid file -- not included"%(file))
            continue            
        f['buoy']=b
        f=f[~f['#YY'].str.contains("#",na=False)]
        f['time']=f.apply(lambda x:datetime.datetime.strptime
                          ("{0}-{1}-{2} {3}:{4}".format(x['#YY'],x['MM'], x['DD'], x['hh'], x['mm']), "%Y-%m-%d %H:%M"),axis=1)
        f=f.drop(f.columns[0:5],1)
        buoys.append(f)
    buoys=pd.concat(buoys)
    buoys=buoys.set_index(['buoy','time']).astype(float)    
    return(buoys)
  
def ERA5frame(path):
    '''
    Create DataFrame from txt file with multiindex columns: buoyID and time.
    
    path - string containing a path specification with all the downloaded ERA5 files, 
    where name of files have to contain 'ERA5_*.txt', * being buoy's ID.
    '''
    reanalysis=[]
    for file in glob.glob(path):
        f=pd.read_table(file,delim_whitespace=True,header=1) 
        #search buoyID in name
        b=re.search(r'ERA5_(\d+).txt',file)
        if b is not None:
            b=b.group(1)
        else:
            print("%s is not a valid file -- not included"%(file))
            continue  
        f['buoy']=b
        f=f.set_index(['buoy'])
        reanalysis.append(f)
    reanalysis=pd.concat(reanalysis)
    #the columns' names were shifted by 1 because of string %
    columns=list(reanalysis.columns)
    #----shift back----
    columns.pop(0)
    reanalysis=reanalysis.drop(reanalysis.columns[-1],axis=1)
    reanalysis.columns=columns
    #--------------------
    reanalysis['time']=reanalysis.apply(lambda x:datetime.datetime.strptime("{0}-{1}-{2} {3}".format
                    (x['YEAR'].astype(int),x['MONTH'].astype(int), x['DAY'].astype(int), x['HOUR'].astype(int)), "%Y-%m-%d %H"),axis=1)
    reanalysis=reanalysis.drop(reanalysis.columns[0:4],1)
    reanalysis=reanalysis.set_index(['time'],append=True)
    
    #add julian days
    years=np.unique(reanalysis.index.get_level_values('time').year)
    aux=[0]*len(reanalysis)
    for i in range(len(reanalysis)):
        for j in years:
            if (reanalysis.index.get_level_values('time')[i].year == j):   
                #subtract each time with january 1st at 00h of the same year
                aux[i]=reanalysis.index.get_level_values('time')[i]-datetime.datetime(j,1,1,0,0)
                break

    aux=np.array([x.total_seconds() for x in aux],dtype=np.float32)
    reanalysis['cost']=np.cos(aux*np.pi*2/(np.max(aux)-np.min(aux)))
    reanalysis['sint']=np.sin(aux*np.pi*2/(np.max(aux)-np.min(aux)))
    reanalysis=reanalysis.astype(float)

    return(reanalysis)
  
def target(X,Buoy,ex_vars,eb_vars,fctime=0):
    '''
    Create the output variables -- error of predictions (model-real) -- for one or more locations (buoys).
    
    X (numerical model) with multiindex names: 
        - buoy 
        - time 

    buoys data with multiindex:
        - buoy 
        - time
    
    ex_vars - list of the variables' names in X wanted for the error prediction.
    eb_vars - list of the variables' names in Buoy wanted for the error prediction, ordered as ex_vars.
    fctime - int forecast time.
    
    '''
    if len(eb_vars)!= len(ex_vars):
        return ('List of variables do not match in size.')    
    #list of error arrays
    elist=[[]]*len(ex_vars) 
    buoy_mean=[[]]*len(ex_vars)
    T=[]
    B=[]
    b=np.unique(X.index.get_level_values('buoy'))
    for i in b:        
        model=X.iloc[np.where(X.index.get_level_values('buoy') == i) ]
        real=Buoy.iloc[np.where(Buoy.index.get_level_values('buoy') == i) ]
        l=len(model.index)
        for j in range(l):    
            
            time=pd.Timestamp(model.index.get_level_values('time')[j])+timedelta(seconds=fctime)
            ind=np.where(np.logical_and(real.index.get_level_values('time') >= (time - timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S'), real.index.get_level_values('time') <= (time + timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S')))

            if len(ind[0])!=0: #if one or more positions were found ...
                for w in range(len(ex_vars)):
                    
                    m=np.mean(real[eb_vars[w]].values[ind])
                    
                    buoy_mean[w] = np.append(buoy_mean[w],m)
                    elist[w] = np.append(elist[w], model[ex_vars[w]].values[j] - m) 

            else: #if no position was found ...
                for w in range(len(ex_vars)):
                    buoy_mean[w] = np.append(buoy_mean[w],np.nan)
                    elist[w] = np.append(elist[w],np.nan)
           
        B.extend(model.index.get_level_values('buoy'))
        T.extend(model.index.get_level_values('time'))
        
    y=pd.DataFrame(elist).transpose()
    buoys_corresp=pd.DataFrame(buoy_mean).transpose()
    y.columns=['e_%s' % i for i in ex_vars]
    buoys_corresp.columns=['%s' % i for i in ex_vars]
    y['time']=T
    buoys_corresp['time']=T
    y['buoy']=B
    buoys_corresp['buoy']=B
    y.set_index(['buoy','time'],inplace=True)
    buoys_corresp.set_index(['buoy','time'],inplace=True)
    
    return y,buoys_corresp 

def series_to_supervised(train,test,outvars,scaler=MinMaxScaler(),n_in=1,n_out=1,dropnan=True):
    '''
    Data reframe for LSTM.
    
    Train/Test - DataFrames (train and test sets) of input variables merged with output variables (if existing) 
    scaler - a scaler function or None
    n_in - number of previous time steps in input
    n_out - number of time steps to predict
    outvars - list of variables' names to predict
    
    '''
    if set(train.columns) != set(test.columns):
        return('Train and Test sets must have same col names')
    if scaler is not None:
        scaler.fit(train)
        train=pd.DataFrame(scaler.transform(train),columns=train.columns)
        test=pd.DataFrame(scaler.transform(test),columns=train.columns)

    #predicting with prior n_in time steps will leave the last n_in steps of training out, which will be moved to test set to predict first outputs
    move=train.iloc[-n_in:,:]    
    names,colsT,colst = list(), list(), list()
    
    #input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        colsT.append(train.shift(i))
        colst.append(test.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in train.columns]
        
    #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        colsT.append(train[outvars].shift(-i))
        colst.append(test[outvars].shift(-i))
        if i == 0:
            names += [('%s(t)' % (j)) for j in outvars]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in outvars]
    #put it all together
    aggT = pd.concat(colsT, axis=1)
    aggT.columns = names
    aggt=pd.concat(colst,axis=1)
    aggt.columns=names
    
    #adding the data moved from training to test
    for i in range(0,n_in):
        for j in range(0,n_in-i):
            aggt.iloc[i,len(train.columns)*j:len(train.columns)*(j+1)]=move.iloc[(j+i),:].values
            
    #drop rows with NaN values after reframe
    if dropnan:
        aggT.dropna(inplace=True)
        aggt.dropna(inplace=True)
        
    if scaler is None:
        return aggT,aggt
    else:
        return aggT,aggt, scaler
