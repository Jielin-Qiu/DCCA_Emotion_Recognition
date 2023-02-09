try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import random

from keras.callbacks import ModelCheckpoint
from utils import load_data, svm_classify_1,svm_classify_2,svm_classify_3
from linear_cca import linear_cca
from models import create_model
import pickle


def train_model(model, data1, data2, epoch_num, batch_size):
    """
    trains the model
    # Arguments
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively. data should be packed
        like ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        epoch_num: number of epochs to train the model
        batch_size: the size of batches
    # Returns
        the trained model
    """

    # Unpacking the data
    train_set_x1, train_set_y1 = data1[0]
    valid_set_x1, valid_set_y1 = data1[1]
    test_set_x1, test_set_y1 = data1[2]

    train_set_x2, train_set_y2 = data2[0]
    valid_set_x2, valid_set_y2 = data2[1]
    test_set_x2, test_set_y2 = data2[2]

    # best weights are saved in "temp_weights.hdf5" during training
    # it is done to return the best model based on the validation loss
    checkpointer = ModelCheckpoint(filepath="temp_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)

    # used dummy Y because labels are not used in the loss function
    model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
              batch_size=batch_size, epochs=epoch_num, shuffle=True,
              validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))),
              callbacks=[checkpointer])

    model.load_weights("temp_weights.h5")

    results = model.evaluate([test_set_x1, test_set_x2], np.zeros(len(test_set_x1)), batch_size=batch_size, verbose=1)
    print('loss on test data: ', results)

    results = model.evaluate([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1)), batch_size=batch_size, verbose=1)
    print('loss on validation data: ', results)
    return model


def test_model(model, data1, data2, outdim_size, apply_linear_cca):
    """produce the new features by using the trained model
    # Arguments
        model: the trained model
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively.
            Data should be packed like
            ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        outdim_size: dimension of new features
        apply_linear_cca: if to apply linear CCA on the new features
    # Returns
        new features packed like
            ((new X for train - view 1, new X for train - view 2, Y for train),
            (new X for validation - view 1, new X for validation - view 2, Y for validation),
            (new X for test - view 1, new X for test - view 2, Y for test))
    """

    # producing the new features
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k][0], data2[k][0]])
        r = int(pred_out.shape[1] / 2)
        new_data.append([pred_out[:, :r], pred_out[:,  r:], data1[k][1]])
    output = open('new_data.pkl', 'wb')
    pickle.dump(new_data, output)
    output.close()


    if apply_linear_cca:
        w = [None, None]
        m = [None, None]
        print("CCA calculating")
        w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], outdim_size)
        

        for k in range(3):
            data_num = len(new_data[k][0])
            for v in range(2):
                new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
                new_data[k][v] = np.dot(new_data[k][v], w[v])

    return new_data


if __name__ == '__main__':

    save_to = './new_features.gz'

    # size of the input for view 1 and view 2
    input_shape1 = 310
    input_shape2 = 33

    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 10
    batch_size = 100

    reg_par = 1e-7
    use_all_singular_values = False

    apply_linear_cca = True


    j=1
    #load data
    data1 = load_data('data1.pkl')
    data2 = load_data('data2.pkl')

    outdim_size = 20

    layer_sizes1 = [389,218,115,outdim_size ]
    layer_sizes2 = [389,218,115,outdim_size ]
        
    # Building, training, and producing the new features by DCCA
    model = create_model(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                            learning_rate, reg_par, outdim_size, use_all_singular_values)
    model.summary()
    model = train_model(model, data1, data2, epoch_num, batch_size)
    new_data = test_model(model, data1, data2, outdim_size, apply_linear_cca)


    [test_acc, valid_acc] = svm_classify_1(new_data, C=100)
    print("Accuracy on test data eeg is:", (test_acc))
    
    [test_acc, valid_acc] = svm_classify_2(new_data, C=100)
    print("Accuracy on test data eye is:", (test_acc))

    [test_acc, valid_acc] = svm_classify_3(new_data, C=100)
    print("Accuracy on test data average is:", (test_acc))

    f1 = gzip.open(save_to, 'wb')
    thepickle.dump(new_data, f1)
    f1.close()
