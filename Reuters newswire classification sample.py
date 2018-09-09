import numpy as np
import keras
#import keras's Reuters Neswire Classification Dataset
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
'''
Loading the datas set and dividing it into train and test.And limiting the max words
to 1000 to get a control of the dataset and easier for execution
'''
max_words=2000
print('Loading data')
print()
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,seed=45,test_split=0.3)
print('x_train values',x_train[0:10])
print('x_test values',x_test[0:10])
print('y_train values',y_train[0:10])
print('y_test values',y_test[0:10])
print('length of x_train sequences',len(x_train))
print('length of x_test sequences',len(x_test))
'''
Now we are calculating the number of classes because it is required to categorize
We are adding one here because because the indexing starts from zer
'''
num_classes = np.max(y_train) + 1
print()
print(num_classes, 'classes')
print()
'''
We are using tozeniser api here and sequence function.
We initialize the model by using the sequential() function and then keep adding layers to this mode.
'''
print()
print('Sequence that has to be vectorized')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train)
x_test = tokenizer.sequences_to_matrix(x_test)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print("************")
print()

#Now x being vectorised we have to vectorize y in a 1D matrix

print('*******************')
print('Converting the class vector to binary class matrix')      
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('*******************')
print()
'''
Now we have the fully connected layer with 700 hidden layer units
'''
print()
print('Building model')
model = Sequential()
model.add(Dense(700, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
#Now compiling the model
print('Compiling model')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Fitting the data to the model')
batch_size = 20
epochs = 5
#We are giving the epoch as 5, as we dont want the model to learn too much which might cause a deviation from the accuracy
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1)
print('Evaluating the test data on the model')
score = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])