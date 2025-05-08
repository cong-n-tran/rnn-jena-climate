# Khanh Nguyen Cong Tran
# 1002046419


from tensorflow import keras
from keras import layers
import numpy as np
import math

def data_normalization(raw_data, train_start, train_end): 

    # get the size of the data anad its dimensions
    (number, dimensions) = raw_data.shape 

    # array filled with zeroes
    normalized_data = np.zeros((number, dimensions)) 

    for d in range(dimensions): 

        # get the feature value in each dimension
        feature_values = raw_data[train_start:train_end, d] 

        # find the mean and standard deviation
        m = np.mean(feature_values) 
        s = np.std(feature_values)

        # calculate the normalized data for that specific dimension
        normalized_data[:, d] = (raw_data[:, d] - m) / s # find normalized values
    
    return normalized_data

def make_inputs_and_targets(data, months, size, sampling): 
    # data -> time series of training, validiation, or test
    # month -> correct month[i] for data[i] 
    # size of (input, target)
    # sampling rate -> how many time series we are skipping

    # size of a 5 days of input time series
    # time_step_length = 720
    time_step_length = 2016 # 144 10-minute intervals per day * 14 days = 2016

    # size of 1 day of target time series
    target_steps = 24*6

    # get time series length and dimensions
    (ts_length, dimensions) = data.shape

    #get the length of input based on how many time series we are skipping
    input_length = math.ceil(time_step_length / sampling) 

    # array of zeroes for inputs and target
    inputs = np.zeros((size, input_length, dimensions))
    targets = np.zeros((size))

    

    for i in range(size):

        # gets random time series in data
        inp, target, _ = get_random_input(data, time_step_length, target_steps, months, sampling)

        # assign them to the respected input and target array
        inputs[i] = inp
        targets[i] = target
    
    return (inputs, targets)



def build_and_train_dense(train_inputs, train_targets,
                                val_inputs, val_targets, filename):
    epochs = 10
    input_shape = train_inputs[0].shape
    num_classes = 12
    model = keras.Sequential([keras.Input(shape=input_shape),
                              keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="tanh"), 
                                # keras.layers.BatchNormalization(),
                                keras.layers.Dropout(0.5),  
                                keras.layers.Dense(512, activation="tanh"),    
                                # keras.layers.BatchNormalization(), 
                                keras.layers.Dropout(0.5),         
                                keras.layers.Dense(num_classes, activation='softmax')
                             ])
    
    # saves the model under filename with the best accuracy
    callbacks = [keras.callbacks.ModelCheckpoint(filename,
                                             save_best_only=True)]
    
    #compile with loss: scc, optimzer: adam and everything else default
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
             optimizer="adam", 
             metrics=['accuracy'])

    # train model and it saves with the callback function
    history_dense = model.fit(train_inputs, train_targets, epochs=epochs, 
                              validation_data=(val_inputs, val_targets), callbacks=callbacks)
    
    return history_dense


def build_and_train_rnn(train_inputs, train_targets, 
                              val_inputs, val_targets, filename): 
    epochs = 10
    input_shape = train_inputs[0].shape
    num_classes = 12
    model = keras.Sequential([keras.Input(shape=input_shape),
                            keras.layers.Bidirectional(keras.layers.LSTM(32)),
                            keras.layers.Dense(64, activation='tanh'), 
                            keras.layers.Dropout(0.3),                  
                            keras.layers.Dense(num_classes, activation='softmax')
                         ])
    

    # saves the model under filename with the best accuracy
    callbacks = [keras.callbacks.ModelCheckpoint(filename,
                                             save_best_only=True)]
    
    #compile with loss: mse, optimzer: adam and everything else default
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
             optimizer="adam", 
             metrics=['accuracy'])

    # train model and it saves with the callback function
    history_dense = model.fit(train_inputs, train_targets, epochs=epochs, 
                              validation_data=(val_inputs, val_targets), callbacks=callbacks)
    
    return history_dense
    

def test_model(filename, test_inputs, test_targets): 
    #load the model
    model = keras.models.load_model(filename)

    # compile the model and print summary 
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
             optimizer="adam", 
             metrics=['accuracy'])
    # model.summary()

    # evaulate
    test_loss, test_acc = model.evaluate(test_inputs, test_targets, verbose=0)

    # return test accurary
    return test_acc 

def confusion_matrix(filename, test_inputs, test_targets): 
    
    #load the model
    model = keras.models.load_model(filename)
    
    # get the predictions
    predictions = model.predict(test_inputs)

    # convert to class labels
    predicted_classes = np.argmax(predictions, axis=1)


    true_classes = test_targets

    # make a 12 x 12 empty matrix 
    conf_matrix = np.zeros((12, 12))

    # fill up the confusion matrix
    for i in range(len(true_classes)): 
        true_class = int(true_classes[i])
        pred_class = int(predicted_classes[i])
        conf_matrix[true_class, pred_class] +=1

    return conf_matrix



# get random input for input and target for our time series
def get_random_input(timeseries, time_step_length, target_steps, target_data, sampling_rate = 1): 
    (ts_length, dimensions) = timeseries.shape

    # find the max start within our time series we can choose (given the 5 input + 1 target rule)
    # max_start = ts_length - time_step_length - target_steps
    max_start = ts_length - time_step_length

    # find a random start value between the 0 and the max start value we can choose
    start = np.random.randint(0, max_start)

    # end will always be + 5 days after the start (on the 6th day)
    end = start + time_step_length

    # input will be day 0 to day 5
    result_input = timeseries[start:end:sampling_rate, :]

    # target will be day 5 to day 6
    midpoint = start + (time_step_length // 2)
    target = target_data[midpoint]

    return (result_input, target, start)
