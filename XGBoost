from xgboost import XGBClassifier

import numpy as np
import pandas as pd 
import os
imgs = (os.listdir("new"))

train = pd.read_csv("new/"+imgs[0])
del(train['Unnamed: 0'])
for i in range(1,len(imgs)):
    a = pd.read_csv("new/"+imgs[i])
    del(a['Unnamed: 0'])
    train = train.append(a)
    
r_train = train[:300000]
ans_train = r_train['answer']
del(r_train['answer'])
r_test = train[300000:]
train = train.sample(frac=1).reset_index(drop=True)
r_test.reset_index(drop=True,inplace=True)
ans_test = r_test['answer']
del(r_test['answer'])

del(train)
mod = XGBClassifier(n_jobs=2,solver="newton-cg",n_estimators=100)

mod.fit(r_train,ans_train)

pred = mod.predict(r_test)
true = 0
for i in range(0,len(pred)):
    if(ans_test[i]==pred[i]):
        true+=1
print(1.0*true/len(pred))
