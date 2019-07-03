
import dill as pickle, json, requests, base64

from sklearn import datasets
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical

keras.backend.set_floatx('float64')


def create_dataset():
    iris = datasets.load_iris()
    X = iris.data  
    y = iris.target
    y = to_categorical(y, num_classes=None)
    # y[y > 0] = 12
    # y[y == 0] = -12
    print(X[0:2,:])
    print(y[0:2,:])
    return (X , y)

def create_model():
    model = Sequential()
    model.add(Dense(5, input_shape=(4,) , activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def fit_model():
    (X, y) = create_dataset()
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=1960)
    
    clf = KerasClassifier(build_fn=create_model, epochs=12, batch_size=1, verbose=0)

    clf.fit(train_X, train_y, verbose=0, batch_size=1)
    # print(test_X.shape)
    # preds = clf.predict(test_X[0,:].reshape(1,4))
    # print(preds)
    return clf

def test_ws_sql_gen(pickle_data):
    WS_URL="https://sklearn2sql.herokuapp.com/model"
    # WS_URL="http://localhost:1888/model"
    b64_data = base64.b64encode(pickle_data).decode('utf-8')
    data={"Name":"keras_iris_dense_model", "PickleData":b64_data , "SQLDialect":"postgresql"}
    r = requests.post(WS_URL, json=data)
    # r.raise_for_status()
    content = r.json()
    # print(content.keys())
    # print(content)
    lSQL = content["model"]["SQLGenrationResult"][0]["SQL"]
    return lSQL;

clf = fit_model()
pickle_data = pickle.dumps(clf)
lSQL = test_ws_sql_gen(pickle_data)
print(lSQL)

