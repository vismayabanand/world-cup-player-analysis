#!/usr/bin/env python
# coding: utf-8

# In[1067]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Batsman_Data.csv")   # reading all the rows and columns
#Since a player did not bat in a particular match I'm using df.replace()

df=df.replace("DNB",-1)
df=df.replace("TDNB",-1)
df=df.replace("-",0)
#TO replace v India and so on to India 


a2=list(set(df['Opposition']))
a3= ["India","Australia","England","Bangladesh","West Indies","Afghanistan","Pakistan","South Africa","New Zealand","Sri Lanka"]
a4=[]
for i in a2:
    a4.append(i[2:])
a5=list(set(a4)-set(a3))

a6=[]
for j in a5:
    a6.append("v "+j)




for i in range(0,len(a6),1):
        
        df=df.replace(a6[i],np.nan)
    
df=df.dropna()

for i in df:
    df.loc[df['Opposition']=='v India', 'Opposition']='India'
    df.loc[df['Opposition']=='v Australia','Opposition']='Australia'
    df.loc[df['Opposition']=='v Sri Lanka','Opposition']='Sri Lanka'
    df.loc[df['Opposition']=='v New Zealand','Opposition']='New Zealand'
    df.loc[df['Opposition']=='v South Africa','Opposition']='South Africa'
    df.loc[df['Opposition']=='v Afghanistan','Opposition']='Afghanistan'
    df.loc[df['Opposition']=='v Pakistan','Opposition']='Pakistan'
    df.loc[df['Opposition']=='v West Indies','Opposition']='West Indies'
    df.loc[df['Opposition']=='v England','Opposition']='England'
    df.loc[df['Opposition']=='v Bangladesh','Opposition']='Bangladesh'


df=df.drop(['Player_ID','Match_ID','Start Date'],axis=1)
df=df.drop(df.columns[0],axis=1)

df


# In[1068]:


p1=list(set(df['Runs']))
for i in range(0,len(p1),1):
    df=df.replace(p1[i],int(p1[i]))
p2=list(set(df['Batsman']))


g=df.groupby(['Batsman','Opposition'])
g1=[]
g2=[]
for i,j in g:
    g1.append(list(i))
    g2.append(str(j.Runs.mean()))


# In[1069]:


df1=pd.read_csv("Bowler_data.csv")
   #dropping first column 
#Replacing average to 0 when the bowler doesn't take any wickets in that match
df1=df1.replace("-",0)


        
a2=list(set(df1['Opposition']))
a3= ["India","Australia","England","Bangladesh","West Indies","Afghanistan","Pakistan","South Africa","New Zealand","Sri Lanka"]
a4=[]
for i in a2:
    a4.append(i[2:])
a5=list(set(a4)-set(a3))
a6=[]
for j in a5:
    a6.append("v "+j)




for i in range(0,len(a6),1):
        
        df1=df1.replace(a6[i],np.nan)
    
df1=df1.dropna()
#df1=df1.drop(df.columns[0],axis=1)
for i in df1:
    df1.loc[df1['Opposition']=='v India', 'Opposition']='India'
    df1.loc[df1['Opposition']=='v Australia','Opposition']='Australia'
    df1.loc[df1['Opposition']=='v Sri Lanka','Opposition']='Sri Lanka'
    df1.loc[df1['Opposition']=='v New Zealand','Opposition']='New Zealand'
    df1.loc[df1['Opposition']=='v South Africa','Opposition']='South Africa'
    df1.loc[df1['Opposition']=='v Afghanistan','Opposition']='Afghanistan'
    df1.loc[df1['Opposition']=='v Pakistan','Opposition']='Pakistan'
    df1.loc[df1['Opposition']=='v West Indies','Opposition']='West Indies'
    df1.loc[df1['Opposition']=='v England','Opposition']='England'
    df1.loc[df1['Opposition']=='v Bangladesh','Opposition']='Bangladesh'

df1=df1.drop(['Player_ID','Match_ID','Start Date','SR'],axis=1)
df1=df1.drop(df1.columns[0],axis=1)


# In[1070]:


df1


# In[1071]:


p1=list(set(df1['Wkts']))
for i in range(0,len(p1),1):
    df1=df1.replace(p1[i],int(p1[i]))
p2=list(set(df1['Bowler']))


g=df1.groupby(['Bowler','Opposition'])
a=[]
b=[]
for i,j in g:
    a.append(list(i))
    b.append(j.Wkts.mean())
f=[]
for i in a:
    f.append(i[0])

        


# In[1072]:


df2=pd.read_csv("Ground_Averages.csv")
df2=df2.drop(['Span','NR'],axis=1)


# In[1073]:


df2


# In[1074]:


df3=pd.read_csv("ODI_Match_Totals.csv")


df3.head()

df3=df3.drop([df3.columns[0],df3.columns[9],df3.columns[10],df3.columns[12]],axis='columns')#remove columns

df3.head()

columns=list(df3)

for i in columns: #replace respective values by desired value
    df3.loc[df3['Result']=='won', 'Result']=1
    df3.loc[df3['Result']=='lost','Result']=0
    df3.loc[df3['Result']=='n/r','Result']=-1
    df3.loc[df3['Result']=='aban','Result']=-1
    df3.loc[df3['Result']=='tied','Result']=-1


df3=df3[~df3.Opposition.str.contains("v Ireland")] #remove rows with these countries
df3=df3[~df3.Opposition.str.contains("v U.S.A.")]
df3=df3[~df3.Opposition.str.contains("v U.A.E")]
df3=df3[~df3.Opposition.str.contains("v Netherlands")]
df3=df3[~df3.Opposition.str.contains("v Kenya")]
df3=df3[~df3.Opposition.str.contains("v Zimbabwe")]
df3=df3[~df3.Opposition.str.contains("v P.N.G")]
df3=df3[~df3.Opposition.str.contains("v Hong Kong")]

for i in columns:  #correct name 
    df3.loc[df3['Country']=='Newzealad', 'Country']='New Zealand'


for i in columns:  #removing 'v' 
    df3.loc[df3['Opposition']=='v Australia', 'Opposition']='Australia'
    df3.loc[df3['Opposition']=='v New Zealand', 'Opposition']='New Zealand'
    df3.loc[df3['Opposition']=='v England', 'Opposition']='England'
    df3.loc[df3['Opposition']=='v Sri Lanka', 'Opposition']='Sri Lanka'
    df3.loc[df3['Opposition']=='v South Africa', 'Opposition']='South Africa'
    df3.loc[df3['Opposition']=='v West Indies', 'Opposition']='West Indies'
    df3.loc[df3['Opposition']=='v Pakistan', 'Opposition']='Pakistan'
    df3.loc[df3['Opposition']=='v Afghanistan', 'Opposition']='Afghanistan'
    df3.loc[df3['Opposition']=='v Bangladesh', 'Opposition']='Bangladesh'
    df3.loc[df3['Opposition']=='v India', 'Opposition']='India'


df3['Target']=df3['Target'].fillna("YTB") #replace nan values by YTB in 'Target' column

df3.head()

for i in columns: #replacing the RPO in unheld matches by 0
    df3.loc[df3['RPO']=='-', 'RPO']=0
    df3.loc[df3['Result']=='-','Result']=-1


# In[1075]:


df3


# In[ ]:





# In[1076]:


df



# In[1077]:


p1=list(set(df['BF']))
for i in range(0,len(p1),1):
    df=df.replace(p1[i],int(p1[i]))
    
p1=list(set(df['4s']))
for i in range(0,len(p1),1):
    df=df.replace(p1[i],int(p1[i]))
    
p1=list(set(df['6s']))
for i in range(0,len(p1),1):
    df=df.replace(p1[i],int(p1[i]))
    
    
p=list(set(df['SR']))
#p1=[float(x) for x in p]
for i in p:
    df=df.replace(i,float(i))


gbat=df.groupby(['Batsman']).transform(lambda x: (x - x.mean()) / x.std())

    
     


# In[1078]:


df1


# In[1079]:


p1=list(set(df1['Wkts']))
for i in range(0,len(p1),1):
    df1=df1.replace(p1[i],int(p1[i]))
    
p1=list(set(df1['Runs']))
for i in range(0,len(p1),1):
    df1=df1.replace(p1[i],int(p1[i]))
    
p1=list(set(df1['Ave']))
for i in range(0,len(p1),1):
    df1=df1.replace(p1[i],float(p1[i]))
    
    
p=list(set(df1['Econ']))
#p1=[float(x) for x in p]
for i in p:
    df1=df1.replace(i,float(i))



gbowl=df1.groupby(['Bowler']).transform(lambda x: (x - x.mean()) / x.std())


# In[1080]:


wk=[]
econ=[]
ave=[]
f5=list(df1['Opposition'])
f1=list(df1['Bowler'])
f2=list(df1['Wkts'])
f3=list(df1['Econ'])
f4=list(df1['Ave'])
for j in range(0,len(f1),1):
    if(f1[j] =='Adam Zampa'):
        wk.append(int(f2[j]))
        econ.append(float(f3[j]))
        ave.append(float(f4[j]))


# In[1081]:


df


# In[1082]:



df.to_csv("Batsmannorm.csv")


# In[1083]:


df1.to_csv("Bowlernorm.csv")


# In[1084]:


print(gbat)


# In[1085]:


print(gbowl)


# In[1086]:


gbat.to_csv("Batno.csv")


# In[1087]:


gbowl.to_csv("Bowlno.csv")


# In[1088]:


sns.pairplot(df)


# In[1089]:


import matplotlib.pyplot as plt
plt.plot(f,b, '-p', color='gray',
         markersize=0.2, linewidth=0.2,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)


# In[1090]:


index = np.arange(len(a))
plt.bar(index, b)
plt.xlabel('Players', fontsize=1)
plt.ylabel('Runs', fontsize=1)
plt.xticks(index, b, fontsize=1, rotation=30)
plt.title('LOL')
plt.show()


# In[1091]:


g1


# In[1098]:


#g2
e=[]
e2=[]
for i in range(0,len(g1),1):
     g1[i].append(g2[i])
for i in g1:
    if(i[0]=="Aaron Finch "):
        e.append(i[1])
        e2.append(i[3])
e


# In[1099]:


df2


# In[1093]:


index = np.arange(len(df2['Ground']))
plt.figure(figsize=(30, 15))
plt.bar(index, df2['Runs'],color='green',align='edge',width=0.4)
plt.xlabel('Grounds', fontsize=10)
plt.ylabel('Runs', fontsize=10)
plt.xticks(index, df2['Ground'], fontsize=14,rotation=90)
plt.title('Best batting pitch')
plt.show()


# In[1094]:


df2['Ground']


# In[1101]:


index = np.arange(len(df2['Ground']))
plt.figure(figsize=(30, 15))
plt.bar(index, df2['Wkts'],color='red',align='edge', width=0.4)
plt.xlabel('Grounds', fontsize=30)
plt.ylabel('Wkts', fontsize=30)
plt.xticks(index, df2['Ground'], fontsize=10,rotation=90)
plt.title('Best Bowling Pitch',fontsize=30)
plt.show()


# In[1102]:


#gbat.dropna()
#gbowl.dropna()

e3=[float(i) for i in e2]
e3


# In[1103]:


index = np.arange(len(e))
plt.figure(figsize=(30, 7))
plt.bar(e, e3,align='center', color='green',width=0.3)
plt.xlabel('Countries', fontsize=30)
plt.ylabel('Average Runs scored', fontsize=30)
plt.xticks(index,e, fontsize=20,rotation=90)
plt.title('Aaron Finch Batting performance ',fontsize=30)
plt.show()


# In[1104]:


e12=[]
e21=[]
k1=df1.groupby(['Bowler','Opposition'])
k2=[]
k3=[]
for i,j in k1:
    k2.append(list(i))
    k3.append(str(j.Wkts.mean()))

for i in range(0,len(k2),1):
     k2[i].append(k3[i])
for i in k2:
    if(i[0]=='Bhuvneshwar Kumar'):
        e12.append(i[1])
        e21.append(float(i[2]))
index = np.arange(len(e12))
plt.figure(figsize=(30, 7))
plt.bar(index, e21,align='center', color='blue',width=0.3)
plt.xlabel('Countries', fontsize=30)
plt.ylabel('Average Wickets taken', fontsize=30)
plt.xticks(index,e12, fontsize=20,rotation=90)
plt.title('Bhuvneshwar Kumar\'s Bowling performance ',fontsize=30)
plt.show()


# In[1139]:


g3=df1.groupby(['Bowler','Opposition'])
g4=[]
g5=[]
g6=[]
g7=[]
for i,j in g3:
    g4.append(list(i))
    g5.append(str(j.Wkts.mean()))
    g6.append(str(j.Econ.mean()))
    g7.append(str(j.Ave.mean()))

wk=[]
econ=[]
ave=[]
c=[]
for i in range(0,len(g4),1):
        g4[i].append(g5[i])
        g4[i].append(g6[i])
        g4[i].append(g7[i])
for i in g4:
    if(i[0]=='Adam Zampa'):
        
        wk.append(float(i[2]))
        econ.append(float(i[3]))
        ave.append(float(i[4]))
        c.append(i[1])
g10=[]
for i in g4:
    if(i[0]=='Adam Zampa'):
        g10.append([i[1],i[2],i[3],i[4]])
g10


# In[1150]:


index=np.arange(len(g10))
bars = np.add(wk, econ).tolist()
plt.bar(x=index,height=wk,color='red',width=0.35)
plt.bar(x=index,height=econ,color='orange',width=0.35,bottom=wk)
plt.bar(x=index,height=ave,color='blue',bottom=bars,width=0.35)

plt.xlabel('Countries', fontsize=30)
plt.ylabel('Bowling stats', fontsize=30)
plt.xticks(index,c, fontsize=20,rotation=90)
plt.title('Adam Zampa performance ',fontsize=30)

import matplotlib.patches as mpa
red=mpa.Patch(color='red',label='Wickets')
orange=mpa.Patch(color='orange',label='Economy')
blue=mpa.Patch(color='blue',label='Average')
plt.legend(handles=[red,orange,blue],loc='upper left')
plt.show()


# In[1107]:


plt.bar(x=index,height=wk,color='red',width=0.35)
plt.xlabel('Countries', fontsize=20)
plt.ylabel('Wickets against each team', fontsize=10)
plt.xticks(index,c, fontsize=15,rotation=90)
plt.title('Adam Zampa wickets ',fontsize=30)
plt.show()


# In[1108]:


#Regression Problem 
q=df.groupby(['Batsman','Runs','BF'])
runsscored=[]
ballsfaced=[]
player=[]

for i,j in q:
    player.append(list(i))
for i in player:
    if(i[0]=='Virat Kohli '):
        runsscored.append(float(i[1]))
        ballsfaced.append(float(i[2]))
len(runsscored)==len(ballsfaced)

plt.scatter(ballsfaced,runsscored,color='red')
plt.xlabel("Balls Faced",fontsize=20)
plt.ylabel("Runs scored",fontsize=20)
plt.title("Virat Kohli's runs prediction")
plt.show()


# In[1109]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
bf2=[[x] for x in ballsfaced]
X_train,X_test,Y_train,Y_test=train_test_split(bf2,runsscored,test_size=0.3,random_state=3)
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_predict=reg.predict(X_test)

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train))
plt.xlabel("Balls Faced",fontsize=20)
plt.ylabel("Runs scored",fontsize=20)
plt.title("Virat Kohli's runs prediction")
plt.show()


# In[1110]:


reg.predict([[25.0],[140.0]])


# In[1111]:


#Hypothesis1 
#Indian fast bowler : Bhuvneshwar Kumar 
#Indian slow bowler : Yuzvendra Chahal
abc=df1.groupby(['Bowler','Opposition'])
bowler=[]
econstr=[]
fastbowlers=[]
slowbowlers=[]

econvalbhuv=[]
econvalchahal=[]
for i,j in abc:
    bowler.append((list(i)))
    econstr.append(str(j.Econ.mean()))
for i in range(0,len(bowler),1):
    if(bowler[i][0]=='Bhuvneshwar Kumar'):
        econvalbhuv.append(float(econstr[i]))
    if(bowler[i][0]=='Yuzvendra Chahal'):
        econvalchahal.append(float(econstr[i]))

fastbowlers.append(econvalbhuv)
slowbowlers.append(econvalchahal)

#England fast bowler : Ben Stokes
#England slow bowler : Adil Rashid
econvalben=[]
econvalrash=[]
for i in range(0,len(bowler),1):
    if(bowler[i][0]=='Ben Stokes'):
        econvalben.append(float(econstr[i]))
    elif(bowler[i][0]=='Adil Rashid'):
        econvalrash.append(float(econstr[i]))
fastbowlers.append(econvalben)
slowbowlers.append(econvalrash)

#Australian fast bowler : Mitchell Starc
#Australian slow bowler : Adam Zampa
econvalsta=[]
econvalz=[]
for i in range(0,len(bowler),1):
    if(bowler[i][0]=='Mitchell Starc'):
        econvalsta.append(float(econstr[i]))
    elif(bowler[i][0]=='Adam Zampa'):
        econvalz.append(float(econstr[i]))
fastbowlers.append(econvalsta)
slowbowlers.append(econvalz)


# In[1112]:


from scipy.stats import ttest_ind
ttest_ind(fastbowlers[0][0:9],slowbowlers[0][0:9],equal_var=True)


# In[1113]:


ttest_ind(fastbowlers[1],slowbowlers[1],equal_var=True)


# In[1114]:


ttest_ind(fastbowlers[2][0:9],slowbowlers[2][0:9],equal_var=True)


# In[1115]:


s1=[]
for i in fastbowlers:
    s1=i+i

s1=fastbowlers[0]+fastbowlers[1]+fastbowlers[2]
s2=slowbowlers[0]+slowbowlers[1]+slowbowlers[2]
ttest_ind(s1,s2,equal_var=True)


# In[1116]:


import statistics as st

d={
     'FastBowlers':pd.Series(s1[0:24]),
     'SlowBowlers':pd.Series(s2)
}
dd=pd.DataFrame(d)
dd.describe()


# In[1117]:


from scipy.stats import variation
p1= variation(s1[0:24])
p2=variation(s2)
p1,p2


# In[1118]:


from scipy.stats import skew
p1= skew(s1)
p2=skew(s2)
p1,p2


# In[1119]:


fp=pd.read_csv("Batno.csv")


# In[1120]:


fp.drop(fp.columns[0],axis=1)


# In[1121]:


k1=df.groupby(['Batsman','Opposition'])
k2=[]
k3=[]
k4=[]
k5=[]
k6=[]
for i,j in k1:
    k2.append(list(i))
    k3.append(str(j.Runs.mean()))
    k4.append(str(j.SR.mean()))
    #k5.append(str(j.k1["4s"].mean()))
    #k6.append(str(j.k1["6s"].mean()))

bat=fp['Runs']
sr=fp['BF']
runs1=[]
bfs=[]
for i in range(0,len(k2),1):
    k2[i].append(float(bat[i]))
    k2[i].append(float(sr[i]))
for i in k2:
    if(i[0]=='David Warner'):
        runs1.append(float(i[2]))
        bfs.append(float(i[3]))


# 

# In[1122]:


runs1


# In[1123]:


bfs


# In[1124]:



plt.plot(df['BF'],df['Runs'],color='red')
plt.xlabel("Balls Faced",fontsize=20)
plt.ylabel("Runs scored",fontsize=20)
plt.title("Batsmen normal plot")
plt.show()


# In[1125]:


from sklearn import preprocessing
scaler = preprocessing.Normalizer()
scaled_df = scaler.fit_transform([df['Runs']])
scale=scaler.fit_transform([df['BF']])

s1=scaler.fit_transform([df['SR']])
s2=scaler.fit_transform([df['4s']])
s3=scaler.fit_transform([df['6s']])
plt.plot(scale[0],scaled_df[0])
plt.show()
a=sns.distplot(scaled_df[0])
b=sns.distplot(scale[0])
c=sns.distplot(s1[0])
d=sns.distplot(s2[0])
e=sns.distplot(s3[0])


# In[1126]:


a1=sns.distplot([df['Runs']])
b1=sns.distplot([df['BF']])
c1=sns.distplot([df['SR']])
d1=sns.distplot([df['4s']])
e1=sns.distplot([df['6s']])


# In[1127]:


import statistics as st
meanofruns=st.mean(scaled_df[0])


# In[1128]:


meanofruns


# In[1129]:


st.variance(scaled_df[0])


# In[1130]:


a=df['Runs'].transform(lambda x: (x - x.mean())/x.std())
b=df['BF'].transform(lambda x: (x - x.mean())/x.std())
c=df['SR'].transform(lambda x: (x - x.mean())/x.std())
d=df['4s'].transform(lambda x: (x - x.mean())/x.std())
e=df['6s'].transform(lambda x: (x - x.mean())/x.std())


# In[1131]:


a


# In[1132]:


a.mean()


# In[1133]:


sns.distplot(a)


# In[1134]:


sns.distplot([df['Runs']])


# In[ ]:





# In[ ]:




