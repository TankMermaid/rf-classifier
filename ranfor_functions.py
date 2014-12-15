import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Function to run random forest classifier
def run_ranfor(X,Y):
    from sklearn.ensemble import RandomForestClassifier
    num_trees=100
    clf =RandomForestClassifier(n_estimators=num_trees, max_features=0.5, oob_score=True)
    clf.fit(X,Y)
    print "Out of bag score:", clf.oob_score_
    return clf


#Function that gets data from a specific day in the treatment

def get_data(df_norm, taxonomy, treatment, day, ylabel):
    #df_norm = Data structure
    #taxonomy = Data structure
    #treatment = string, e.g. 'HSD'
    #day = string, e.g. '+01' or '+14' 
    #ylabel = int, e.g. 0, 1, 2, ...
 
    data = df_norm.filter(regex="%s_group\d_day\%s_animal\d+_\w+"% (treatment,day)) #Use given day
    
    if data.empty:
        print "Dataset " + treatment + day +  " doesn't exist"
    else:
        df = pandas.concat([data, taxonomy], join="inner", axis=1)     
        grp = df.groupby(df.Taxonomy)
        X = grp.sum() / np.sum(grp.sum())
        X = X.T   
        Y = [ylabel]*np.shape(X)[0]
    
    return X,Y  #return X,Y from specific treatment and day
    
def get_data_seq(df_norm, treatment, day, ylabel):
    #df_norm = Data structure
    #taxonomy = Data structure
    #treatment = string, e.g. 'HSD'
    #day = string, e.g. '+01' or '+14' 
    #ylabel = int, e.g. 0, 1, 2, ...
 
    data = df_norm.filter(regex="%s_group\d_day\%s_animal\d+_\w+"% (treatment,day)) #Use given day
    
    if data.empty:
        print "Dataset " + treatment + day +  " doesn't exist"
    else:
        X = data.T   
        Y = [ylabel]*np.shape(X)[0]
    
    return X,Y  #return X,Y from specific treatment and day


def make_pairwise(X1, Y1, X2, Y2):
    X = pandas.concat([X1, X2], join="inner", axis=0)
    Y = Y1 + Y2
    return X,Y


def logit(X):
    X = np.log(X/(1-X))
    return X
