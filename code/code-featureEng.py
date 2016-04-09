# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import re
from sklearn import preprocessing
from sklearn.ensemble import BaggingRegressor

data_train=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/train_11.csv")
data_test=pd.read_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/test.csv")
data_train
data_train.info()
fig=plt.figure()
fig.set(alpha=0.2)
data_train.survived.value_counts().plot(kind='bar')
plt.title(u"survived（1 is survived）")
plt.ylabel(u"population")
plt.xlabel(u"Pclass")
plt.show()
plt.scatter(data_train.Survived,data_train.Age)
plt.show()
############ deal with the null value of cabin#############
combine["Cabin_2"]=combine["Cabin_1"][combine["Cabin_1"]!=0]
combine["Cabin_2"][combine["PassengerId"].isin(combine["PassengerId"][combine["Pclass"]==1][combine["Cabin_2"].isnull()])]=combine["Cabin_1"][combine["Pclass"]==1][combine["Cabin_1"]!=0].mean()
combine["Cabin_2"][combine["PassengerId"].isin(combine["PassengerId"][combine["Pclass"]==2][combine["Cabin_2"].isnull()])]=combine["Cabin_1"][combine["Pclass"]==2][combine["Cabin_1"]!=0].mean()
combine["Cabin_2"][combine["PassengerId"].isin(combine["PassengerId"][combine["Pclass"]==3][combine["Cabin_2"].isnull()])]=combine["Cabin_1"][combine["Pclass"]==3][combine["Cabin_1"]!=0].mean()

##############
fig=plt.figure()
fig.set(alpha=0.2)                               
Survived_0=data_train.Survived[data_train.Cabin_1==0].Value_counts()
Survived_1=data_train.Survived[data_train.Cabin_1==1].value_counts()
Survived_2=data_train.Survived[data_train.Cabin_1==2].value_counts()
Survived_3=data_train.Survived[data_train.Cabin_1==3].value_counts()
Survived_4=data_train.Survived[data_train.Cabin_1==4].value_counts()
Survived_5=data_train.Survived[data_train.Cabin_1==5].value_counts()
Survived_6=data_train.Survived[data_train.Cabin_1==6].value_counts()
Survived_7=data_train.Survived[data_train.Cabin_1==7].value_counts()
df=pd.DataFrame({u'NA':Survived_0,u'A':Survived_1,u'b':Survived_2,u'c':Survived_3,u'd':Survived_4,u'e':Survived_5,u'f':Survived_6,u'g':Survived_7})
df.plot(kind='bar')
plt.show()
###################
# merge train and test
combine=pd.concat([data_train,data_test])
# rebuild index
combine.reset_index(inplace=True)
# delete index column
combine.drop('index', axis=1, inplace=True)
print combine.shape[1], "columns:", combine.columns.values
print "Row count:", combine.shape[0]
###################“Fare”，“Embarked” using median value to fill where the NA value is
combine['Fare'][np.isnan(combine['Fare'])]=combine['Fare'].median()#14.4542
#common value：
combine.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values#S
###############decrease the varience of value in particular variables
combine['Fare_bin']=pd.qcut(combine['Fare'],4)
combine = pd.concat([combine, pd.get_dummies(combine['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)
######translate the non-numeric value into numeric value
combine['Embarked'][combine['Embarked']=='S']=1
combine['Embarked'][combine['Embarked']=='C']=2
combine['Embarked'][combine['Embarked']=='Q']=3

###########store pro1.csv###########
#                                       #
combine.to_csv('/Users/jessicazhao/Documents/homework/DataAnalytics_final/pro1.csv',index=False)
########################################

#########deal with Age->Child################
child=combine['Age']
Child=pd.DataFrame(data=child)
Child['Age'][Child['Age']<18]=1
Child['Age'][Child['Age']>=18]=0
combine['Child']=Child

########## missing data :Age ##################
#####average +/- stantard deviation #####--good
average_age   = combine["Age"].mean()
std_age       = combine["Age"].std()
count_nan_age = combine["Age"].isnull().sum()
rand = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)
combine['Age'][df.Age.isnull()] = rand
Child['Age'][Child['Age']<18]=1
Child['Age'][Child['Age']>=18]=0
#################################

###########store pro2.csv###########
#                                       #
combine.to_csv('/Users/jessicazhao/Documents/homework/DataAnalytics_final/pro2.csv',index=False)
########################################

combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Miss"])]=combine["Age"][combine["Title"]=="Miss"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Mrs"])]=combine["Age"][combine["Title"]=="Mrs"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Dr"])]=combine["Age"][combine["Title"]=="Dr"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Lady"])]=combine["Age"][combine["Title"]=="Lady"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Master"])]=combine["Age"][combine["Title"]=="Master"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Mr"])]=combine["Age"][combine["Title"]=="Mr"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Rev"])]=combine["Age"][combine["Title"]=="Rev"].mean()
combine["Age"][combine["PassengerId"].isin(combine["PassengerId"][combine["Age"].isnull()][combine["Title"]=="Sir"])]=combine["Age"][combine["Title"]=="Sir"].mean()

combine.to_csv("/Users/jessicazhao/Documents/homework/DataAnalytics_final/pro4_age.csv")
######  2.random forests（pro4.csv）#####
from sklearn.ensemble import RandomForestRegressor

### using RandomForestClassifier predict the missing data
def set_missing_ages(df):

    
    age_df = df[['Age','Age_sc','Fare_sc', 'Parch', 'SibSp', 'Pclass','Title_id']]

    # saperate passengers into two groups:passengers with age/without age
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # predicton
    y = known_age[:, 0]

    # X:a set of feature
    X = known_age[:, 1:]

    # fiting RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # prediction process
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # fill the blank with the prediction
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

combine, rfr = set_missing_ages(combine)
combine = set_Cabin_type(combine)

###########restore pro5.csv###########
#                                       #
combine.to_csv('/Users/jessicazhao/Documents/homework/DataAnalytics_final/pro5.csv',index=False)
########################################

########### Name->Title############
combine['Title'] = combine['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
combine['Title'][combine.Title == 'Jonkheer'] = 'Master'
combine['Title'][combine.Title.isin(['Ms','Mlle'])] = 'Miss'
combine['Title'][combine.Title == 'Mme'] = 'Mrs'
combine['Title'][combine.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
combine['Title'][combine.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
###########  Family   ############
#withfamily->1/0
combine['Family']=combine['Parch']+combine['SibSp']
combine['Family'][combine['Family']>1]=1
combine['Family'][combine['Family']==0]=0
#picture1 
Survived_1 = combine.Survived[combine.Family == 1].value_counts()
Survived_0 = combine.Survived[combine.Family == 0].value_counts()
df=pd.DataFrame({u'Family=0':Survived_0,u'Family=1':Survived_1})
df.plot(kind='bar')
plt.xlabel(u'Survived')
plt.ylabel(u'counts')
plt.title(u'Survived&Family')
plt.show()
#picture2
Family_0 = combine.Family[combine.Survived == 0].value_counts()
Family_1=combine.Family[combine.Survived==1].value_counts()
df_0=pd.DataFrame({u'Survived=1':Family_1,u'Survived=0':Family_0})
df_0.plot(kind='bar')
plt.xlabel(u'Family(1->withFamily)')
plt.ylabel(u'counts')
plt.title(u'Survived&Family')
plt.show()
#familysize:<3->small;>=3->big
#familyname
combine['Surname']=combine['Name'].map(lambda x: re.compile("(Mr|Mrs|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Lady|Sir|Mlle|Col|Capt|the Countess|Jonkheer|Dona)\.\s(\w*)").findall(x)[0][1])
combine['Surname']=pd.factorize(df['Surname'])[0]
#title
combine['Title_id']=pd.factorize(combine['Title'])[0]+1
#familysize
combine['FamilySize']=combine['Parch']+combine['SibSp']+1
combine['FamilySize'].loc[combine['FamilySize']<3]='small'
combine['FamilySize'].loc[combine['FamilySize']!='small']='big'
combine['FamilySize'][combine['FamilySize']=='small']=0
combine['FamilySize'][combine['FamilySize']=='big']=1
combine['FamilySize']=combine['FamilySize'].astype(int)
###########store pro3.csv###########
#                                       #
combine.to_csv('/Users/jessicazhao/Documents/homework/DataAnalytics_final/pro3.csv',index=False)
########################################

##scalling Age&Fare
scaler=preprocessing.StandardScaler()
combine['Age_sc']=scaler.fit_transform(combine['Age'])
combine['Fare_sc']=scaler.fit_transform(combine['Fare'])
###########store pro4.csv(Age is still missing)pro4_1.csv(conpleted data)###########--
#                                       #
combine.to_csv('/Users/jessicazhao/Documents/homework/DataAnalytics_final/pro4.csv',index=False)
########################################

