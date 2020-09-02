# this function take dataframe as input it is to print description shape info of dataframe and returns list of contineous and discrete attribute name
def brief_info(df):
    discols=[]
    conticols=[]
    print(df.info())
    print(df.shape)
    print('Discrete columns are:')
    for cols in df.columns:
        if (df[cols].dtypes)=='int64':
            discols=discols+[cols]
    print(discols)       
    print('\n__________________________')
    print('Continous Columns are:')
    for cols in df.columns:
        if (df[cols].dtypes)!='int64':
            conticols=conticols+[cols]
    print(conticols)       
    return df.describe()

# this is to value_counts() of all the discrete columns
def valuecount(df,columnlist):
  for i in columnlist:
    print("----------",i,"-----------")
    print(df[i].value_counts())

# this to count plot of all discrete column in the dataframe
def showcount(df,dislist,target):
    l1=dislist
    print("------ Bivariate ------")
    for i in l1:
        sns.countplot(x=i,hue=target,data=df)
        plt.show()
    print("------ Univariate -----")  
    for i in l1:
        sns.countplot(data[i])
        plt.show()

# it shows the distribution of contineous columns
def showdist(dislist,df,target):
    l1=dislist
    print("---------- Bivariate ----------")
    for i in l1:
        fig=sns.FacetGrid(data=df,hue=target)
        fig.map(sns.kdeplot,i)
        fig.add_legend()
        plt.show()
    print("---------- Univariate ----------") 
    for i in l1:
        sns.distplot(df[i])
        plt.show()

# the function is to plot boxplot of all numerical attribute wrt categorical attribute
def showboxplot(conti,disc,data,target):
  for i in conti:
    print("------------ wrt ",i,"-------------")
    for j in disc: 
      sns.boxplot(x=j,y=i,hue=target,data=data)
      plt.show()

# this is to plot scatterplot all attribute wrt target attri      
def showscatterplot(dcol,ccol,data,target):
  print("---------- wrt ",target,"-----------")
  for i in ccol+dcol:  
    fig,ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=i, y=target,hue=target, data=data)
    plt.show()
  print("------ among contineous columns ------")  
  for i in ccol:
    for j in ccol:
      fig,ax = plt.subplots(figsize=(7,5))
      sns.scatterplot(x=i, y=j,hue=target, data=data)
      plt.show()

# this function is transform the skewed attribute to non skewed normalized attribute      
def normalize(col,data):
  trnsdata = power_transform(data[col], method='yeo-johnson',standardize=True)
  trnsdata = pd.DataFrame(data=trnsdata,columns=col)
  data = data.drop(col,axis=1)
  data = pd.concat([data,trnsdata],axis=1)
  return data

# this function is to plot the roc_auc curve 
def Auc_curve(model,Xtest,ytest,ypredicted):
    #import sklearn.metrics as metrics
    probs = model.predict_proba(Xtest)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(ytest, ypredicted)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    #import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# this function is to test different model on the given data and view different scores for classification problem
def modeltest(Xtrain,Xtest,ytrain,ytest,model):
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        print("recall :",metrics.recall_score(ytest,ypred))
        print("acc :",metrics.accuracy_score(ytest,ypred))
        print("roc :",metrics.roc_auc_score(ytest,ypred))
        print("f1 :",metrics.f1_score(ytest,ypred))
        print("precision:",metrics.precision_score(ytest,ypred))
        Auc_curve(model,Xtest,ytest,ypred)
        matrix=confusion_matrix(ytest, ypred)
        print(matrix)

# the function Most_imp_feature() is for feature selection it implements 1) mutual_info_classif() 2) Randomforestclassifier() 3) REF()
# and output a list of names of column which are important common columns in all  
def Intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))
def Most_imp_feature(Xtrain,ytrain):
    mi=feature_selection.mutual_info_classif(Xtrain,ytrain)
    miser=pd.Series(mi)
    miser.index=Xtrain.columns.values
    m=miser.sort_values(ascending=False)
    robj=ensemble.RandomForestClassifier(n_estimators=200)
    robj.fit(Xtrain,ytrain)
    ser1=pd.Series(robj.feature_importances_)
    ser1.index=Xtrain.columns.values
    r=ser1.sort_values(ascending=False)
    rfeobj=feature_selection.RFE(estimator=linear_model.LogisticRegression(C=100,penalty="l2"),n_features_to_select=10)
    a=rfeobj.fit_transform(Xtrain,ytrain)
    f=list(Xtrain.columns[rfeobj.get_support()])
    lst1=[]
    for i in range(0,10):
        lst1=lst1+[m.index[i]]
    lst2=[]
    for i in range(0,10):
        lst2=lst2+[r.index[i]]
    
    intsec=Intersection(lst1, lst2)
    
    impfeature=Intersection(intsec, f)
    
    return impfeature

# this function is also for feature selection which implements chi2 test and outputs a list of features
def Chi2(Xtraindis,ytrain,dis):
    chiarr,parr=feature_selection.chi2(Xtraindis,ytrain)      # chi2 test for discrete values
    cols=dis
    ser=pd.Series(chiarr)
    ser.index=cols
    ch=ser.sort_values(ascending=False)
    lst1=[]
    for i in range(int(len(cols)/2)):
            lst1=lst1+[ch.index[i]]
    return lst1

# this function tests the given data on 4 different model 1) logistic regression 2)Randomforest 3)KNN 4) Naive bayes and outputs the performance score  of the
# models so that we can compare the results across the models.
def modelstats(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","Randomforest","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(ensemble.RandomForestClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.f1_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.f1_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","f1","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames),pd.crosstab(ytest,testprediction)

# this function does the stratified splitting of the data 
def splitdata(df,target,ts):
  split = StratifiedShuffleSplit(n_splits=1, test_size=ts, random_state=42)
  for train_index, valid_index in split.split(df, df[target]):
          train = df.loc[train_index]
          valid = df.loc[valid_index]
  train=train.reset_index(drop=True) 
  valid=valid.reset_index(drop=True)       
  return train,valid
