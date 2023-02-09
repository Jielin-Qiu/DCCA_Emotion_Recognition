import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import theano
from keras.utils.data_utils import get_file
import pickle as thepickle
from sklearn.svm import NuSVC

def load_data(data_file):
    """loads the data """
    print('loading data ...')
    fid = open(data_file, 'rb')
    train_set, valid_set, test_set = thepickle.load(fid)
    fid.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify_1(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]


    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())
    result=clf.predict(test_data)

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    print(p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]

def svm_classify_2(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    _, train_data, train_label = data[0]
    _, valid_data, valid_label = data[1]
    _, test_data, test_label = data[2]

    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())
    result=clf.predict(test_data)

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    print(p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]

def svm_classify_3(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data_1, train_data_2, train_label = data[0]
    train_data=(train_data_1+train_data_2)/2
    valid_data_1, valid_data_2, valid_label = data[1]
    valid_data=(valid_data_1+valid_data_2)/2
    test_data_1, test_data_2, test_label = data[2]
    test_data=(test_data_1+test_data_2)/2

    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())
    result=clf.predict(test_data)

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    print(p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]