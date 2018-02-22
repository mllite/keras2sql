
import os, numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


sample = True
small = True

batch_size = 128
num_classes = 10
epochs = 12

P = 1
# input image dimensions
img_rows, img_cols = 28, 28
if(small):
    P = 4
    img_rows, img_cols = img_rows // P, img_cols // P

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_cols_ind = [P*col for col in range(img_cols)]
img_rows_ind = [P*row for row in range(img_rows)]
ixgrid_train = np.ix_(range(x_train.shape[0]), img_rows_ind, img_cols_ind)
x_train = x_train[ixgrid_train]
ixgrid_test = np.ix_(range(x_test.shape[0]), img_rows_ind, img_cols_ind)
x_test = x_test[ixgrid_test]


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


if(sample):
    indices = np.random.choice(x_train.shape[0], x_train.shape[0] // 100, replace=False)
    x_train = x_train[indices, : , :, :]
    y_train = y_train[indices]
    indices = np.random.choice(x_test.shape[0], x_test.shape[0] // 100, replace=False)
    x_test = x_test[indices, : , :, :]
    y_test = y_test[indices]


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



def create_model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    # model.add(Conv2D(4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


from keras.wrappers.scikit_learn import KerasClassifier

clf = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

clf.fit(x_train, y_train ,
        batch_size=batch_size,
        epochs=12,
        verbose=1,
        validation_data=(x_test, y_test))

print(x_test.shape)
preds = clf.predict(x_test[0,:].reshape(1, img_rows , img_cols, 1))
print(preds)


import json, requests, base64, dill as pickle, sys



sys.setrecursionlimit(200000)
pickle.settings['recurse'] = False

# no luck for the web service... pickling feature of tensorflow and/or keras objects seems not to be a priority.
# there is a lot of github issues in the two projects when I search for pickle keyword!!!.

def test_ws_sql_gen(pickle_data):
    WS_URL="http://localhost:1888/model"
    b64_data = base64.b64encode(pickle_data).decode('utf-8')
    data={"Name":"model1", "PickleData":b64_data , "SQLDialect":"postgresql"}
    r = requests.post(WS_URL, json=data)
    print(r.__dict__)
    content = r.json()
    # print(content)
    lSQL = content["model"]["SQLGenrationResult"][0]["SQL"]
    return lSQL;



def test_sql_gen(keras_regressor , metadata):
    import sklearn2sql.PyCodeGenerator as codegen
    cg1 = codegen.cAbstractCodeGenerator();
    lSQL = cg1.generateCodeWithMetadata(clf, metadata, dsn = None, dialect = "postgresql");
    return lSQL[0]


# In[37]:


# commented .. see above
# pickle_data = pickle.dumps(clf)
# lSQL = test_ws_sql_gen(pickle_data)
# print(lSQL[0:2000])


# In[38]:


lMetaData = {}
NC = x_test.shape[1] *  x_test.shape[2] *  x_test.shape[3]
lMetaData['features'] = ["X_" + str(x+1)  for x in range(0 , NC)]

lMetaData["targets"] = ['TGT']
lMetaData['primary_key'] = 'KEY'
lMetaData['table'] = 'mnist'

    
lSQL = test_sql_gen(clf , lMetaData)


# In[39]:


print("SQL_START")
print(lSQL)
print("SQL_END")


# # Execute the SQL Code

# In[40]:


# save the dataset in a database table


import sqlalchemy as sa

# engine = sa.create_engine('sqlite://' , echo=False)
engine = sa.create_engine("postgresql://db:db@localhost/db?port=5432", echo=False)
conn = engine.connect()
NR = x_test.shape[0]
lTable = pd.DataFrame(x_test.reshape(NR , NC));
lTable.columns = lMetaData['features']
# lTable['TGT'] = None
lTable['KEY'] = range(NR)
lTable.to_sql(lMetaData['table'] , conn,   if_exists='replace', index=False)


# In[41]:


sql_output = pd.read_sql(lSQL , conn);
sql_output = sql_output.sort_values(by='KEY').reset_index(drop=True)
conn.close()

# In[ ]:


print(sql_output.sample(12, random_state=1960))


# # Keras Prediction

# In[ ]:


keras_output = pd.DataFrame()
keras_output_key = pd.DataFrame(list(range(x_test.shape[0])), columns=['KEY']);
keras_output_score = pd.DataFrame(columns=['Score_' + str(x) for x in range(num_classes)]);
keras_output_proba = pd.DataFrame(clf.predict_proba(x_test), columns=['Proba_' + str(x) for x in range(num_classes)])
keras_output = pd.concat([keras_output_key, keras_output_score, keras_output_proba] , axis=1)
for class_label in range(num_classes):
    keras_output['LogProba_' + str(class_label)] = np.log(keras_output_proba['Proba_' + str(class_label)])
keras_output['Decision'] = clf.predict(x_test)
print(keras_output.sample(12, random_state=1960))


# # Comparing the SQL and Keras Predictions

# In[ ]:


sql_keras_join = keras_output.join(sql_output , how='left', on='KEY', lsuffix='_keras', rsuffix='_sql')


# In[ ]:


sql_keras_join.head(12)


# In[ ]:


condition = (sql_keras_join.Decision_sql != sql_keras_join.Decision_keras)
print("CONDITION_START")
lResult_df = sql_keras_join[condition]
print(lResult_df.head())
print(lResult_df[['KEY_keras', 'KEY_sql', 'Decision_keras' , 'Decision_sql']].head())

print("CONDITION_END")
