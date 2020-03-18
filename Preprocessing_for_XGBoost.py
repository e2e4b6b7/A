import numpy as np
import pandas as pd 
import os
imgs = os.listdir("../Downloads/Новая папка")
import math

number = 1000
pos = 0
for ll in range(0,345):
    tab = pd.DataFrame()
    for i in range(0,number):
        tab.loc[i,"answer"] = ll
    a = np.load("../Downloads/Новая папка/"+imgs[ll])
    np.random.shuffle(a)
    for l in range(0,number):
        N = 0
        aXY = 0.0
        aX = 0.0
        aY = 0.0
        aX2 = 0.0
        for i in range(0,28):
            for j in range(0,28):
                aXY += i*j*a[l][i*28+j]
                aX += i*a[l][i*28+j]
                aY += j*a[l][i*28+j]
                aX2 += (i**2)*a[l][i*28+j]
                N += a[l][i*28+j]
        aXY /= N
        aY /= N
        aX /= N
        aX2 /= N
        k = (aXY-aX*aY)/(aX2 - aX**2)
        b = aY-k*aX
        err = 0.0
        for i in range(0,28):
            for j in range(0,28):
                err += (a[l][i*28+j]*(j-(k*i+b))**2)/N
        k2 = -1.0/k
        b2 = aY-k*aX
        err2 = 0.0
        for i in range(0,28):
            for j in range(0,28):
                err2 += (a[l][i*28+j]*(j-(k2*i+b2))**2)/N
        tab.loc[l,"k"] = k
        tab.loc[l,"b"] = b
        tab.loc[l,"err"] = err
        tab.loc[l,"err2"] = err2

        avI = [0 for i in range(0,28)]
        avJ = [0 for i in range(0,28)]
        for i in range (0,28):
            for j in range(0,28):
                avI[i] += a[l][i*28+j]/28.0
                avJ[j] += a[l][i*28+j]/28.0
        for i in range(0,28):
            tab.loc[pos+l,"avI"+str(i)] = avI[i]
            tab.loc[pos+l,"avJ"+str(i)] = avJ[i]

        num1 = [0 for i in range(0,24)]
        num0 = [0 for i in range(0,24)]

        def dfs(i, j, color, used):
            used[i][j] = True
            if(i > 0 and not used[i-1][j] and a[l][(i-1)*28+j] >= color):
                dfs(i-1,j,color,used)
            if(j > 0 and not used[i][j-1] and a[l][i*28+j-1] >= color):
                dfs(i,j-1,color,used)
            if(i < 27 and not used[i+1][j] and a[l][(i+1)*28+j] >= color):
                dfs(i+1,j,color,used)
            if(j < 27 and not used[i][j+1] and a[l][i*28+j+1] >= color):
                dfs(i,j+1,color,used)

        def dfs2(i, j, color, used):
            used[i][j] = True
            if(i > 0 and not used[i-1][j] and a[l][(i-1)*28+j] < color):
                dfs2(i-1,j,color,used)
            if(j > 0 and not used[i][j-1] and a[l][i*28+j-1] < color):
                dfs2(i,j-1,color,used)
            if(i < 27 and not used[i+1][j] and a[l][(i+1)*28+j] < color):
                dfs2(i+1,j,color,used)
            if(j < 27 and not used[i][j+1] and a[l][i*28+j+1] < color):
                dfs2(i,j+1,color,used)

        for m in range (1,25):
            colorNow = m*10
            used = [[False for j in range(0,28)]for j in range(0,28)]
            for i in range(0,28):
                for j in range(0,28):
                    if(a[l][i*28+j]>=colorNow and not used[i][j]):
                        dfs(i,j,colorNow,used)
                        num1[m-1]+=1
                    if(a[l][i*28+j]<colorNow and not used[i][j]):
                        dfs2(i,j,colorNow,used)
                        num0[m-1]+=1
        for i in range(0,24):
            tab.loc[l,'numComponent1Color'+str((i+1)*10)] = num1[i]
            tab.loc[l,'numComponent0Color'+str((i+1)*10)] = num0[i]
    tab.replace(np.nan,0,inplace = True)
    tab.to_csv("new/"+imgs[ll]+".csv")
