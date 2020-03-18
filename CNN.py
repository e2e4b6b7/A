import numpy as np
import keras
from keras.layers import Convolution2D, MaxPooling2D,Dense, Dropout, Activation, Flatten
import os
imgs = (os.listdir("../input"))
from keras.optimizers import Adam

lens = np.array([ 58252,  75811,  61699,  74002,  74868,  68923,  62306,  63115,
        72361,  60475,  84051,  62061,  62900, 153968,  73807,  75569,
        61904,  67687,  59229,  66896,  59057,  87168,  62469,  67381,
        82601,  56931,  60445,  95559,  64347,  63263,  62095,  66786,
        72491,  64076,  63939,  59682,  71341,  76603,  65141,  59708,
        71516,  60285,  66505,  66413,  58463,  62032,  93704,  83104,
        60260,  58999,  65838,  62452,  64187, 160990,  60699,  64386,
        86355,  66697,  70772,  70697,  61883,  91382,  66229,  61267,
        61601,  57706,  60565,  74862, 111353,  83751,  82112,  61438,
        63107,  60268,  60132,  91716,  63804,  61942,  65676, 135722,
        59831,  61541,  63465,  64976,  63966,  67044,  61705,  65360,
        65793,  84879, 145119,  76079,  60806,  70375,  60115,  62181,
        61697,  68393,  68649,  67740,  78987,  61448,  63126,  63484,
        67431,  59169,  62944, 112881,  80833,  68079,  59955,  64713,
        83978,  68621,  77785, 110347,  67075,  62284, 119881,  60759,
        83177,  72409,  75983, 101543,  63038,  79523,  61912,  60921,
        79263,  63591,  95001,  97421,  77652,  61535,  60225,  64836,
        59506, 145886, 142701, 111305,  59453,  60263,  79969,  60949,
        71217, 101650,  65055,  89143,  83724,  63175,  90999,  60139,
        67978,  61498,  67710,  68122,  61566, 107062,  60065,  87235,
        80482,  93883, 133770,  86328,  62694,  74956, 130750,  62785,
        58402,  60439,  60630,  80451,  75780,  71774,  60474,  63811,
        70087,  64424,  65026,  60314, 159568,  71984,  68667,  90152,
        60285,  65266,  63816,  60830,  61514,  84965,  64270,  89413,
        89962,  67067,  76459,  71083,  79296,  60290,  98786,  65746,
        79737,  75076,  66148, 103455,  84816,  61723,  93501,  60979,
        56806,  72132,  63564,  63659,  92765,  75132,  63313,  58452,
        80828,  61000, 126895,  58435,  65370,  61185,  93385,  59376,
        62535,  65185,  86274,  65012,  60810,  66719,  63353,  62853,
       164602,  84731,  61660,  77644,  59794,  67864,  67340,  63422,
        77319,  59822,  94242,  86222,  66635,  71785,  59886,  68253,
        65866,  60628,  59053,  61020,  61695,  82844,  58156,  59938,
        65968,  63025,  63060,  60115,  62485,  58597, 104205,  64366,
        63087,  91854,  59845,  62193,  66878,  61136,  77266,  58342,
       170014,  62674, 102857,  94290, 104723,  62514,  85100,  62572,
        59220,  78441,  64490,  68809,  61021,  61222,  76897,  62596,
        59907,  58267,  61150,  61640,  59541,  62181,  63221,  66890,
        76044,  60092,  59678,  61901,  67911,  62616,  64010,  63402,
        89784,  63942,  61568, 115575,  65763,  67400,  96507,  60691,
        60533,  61217,  76826,  64944,  62532,  62414,  65518,  71635,
        58338,  62660,  63974,  72360,  61585,  92379,  65677,  84773,
        62042,  62274,  82954,  63237, 108630,  60425,  66469,  92682,
        58251,  68329,  60322,  63186,  66151,  81322, 140221,  72304,
        60036])
        
count = 400
first = True
for i in range(0,23):
    b = np.load("../input/"+str(i)+".npy")
    pos = 0
    for j in range(0,15):
        c = b[pos:pos+lens[i*15+j]]
        pos+=lens[i*15+j]
        np.random.shuffle(c)
        if(first):
            trai = c[:int(count*0.9)]
            first = False
        else:
            trai = np.vstack((trai,c[:int(count*0.9)]))

for i in range(0,23):
    b = np.load("../input/"+str(i)+".npy")
    pos = 0
    for j in range(0,15):
        c = b[pos:pos+lens[i*15+j]]
        pos+=lens[i*15+j]
        np.random.shuffle(c)
        if(first):
            trai = c[:int(count*0.1)]
            first = False
        else:
            trai = np.vstack((trai,c[:int(count*0.1)]))
            
ans_train = [0 for i in range(0,count*345)]
position = 0
for i in range(0,345):
    for j in range(0,int(count*0.9)):
        ans_train[position] = i
        position+=1;
for i in range(0,345):
    for j in range(0,int(count*0.1)):
        ans_train[position] = i
        position+=1;
        
trai = trai.astype("float32")
trai/=255
ans_train = keras.utils.np_utils.to_categorical(ans_train,345)

model = keras.Sequential()

model.add(Convolution2D(70, (4, 4), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(60, (3, 3), activation='relu', data_format='channels_first'))
model.add(Convolution2D(50, (3, 3), activation='relu', data_format='channels_first'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = (2,2), data_format='channels_first'))
model.add(Flatten())
model.add(Dense(800, activation='relu',kernel_regularizer=keras.regularizers.l2(0.000015)))
model.add(Dropout(0.4))
model.add(Dense(700, activation='relu',kernel_regularizer=keras.regularizers.l2(0.000015)))
model.add(Dropout(0.35))
model.add(Dense(345, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(trai, ans_train, batch_size=64, epochs=15, verbose=1, validation_split=0.1)
