# -*- coding: utf-8 -*-
"""
base.py
~~~~~~~
.. topic:: Contents

    The base module includes the basic functions such as
    to load data, annotations, to normalize matrices
    and generate nonnegative random matrices
    
    Copyright 2014-2016 Romain Serizel

    This software is distributed under the terms of the GNU Public License 
    version 3 (http://www.gnu.org/licenses/gpl.txt)"""

from sklearn import preprocessing
import h5py
import numpy as np
import itertools
import more_itertools
import theano.tensor as T
from theano.ifelse import ifelse
import theano


def load_data(f_name, dataset, scale=True, rnd=False):

    """Get data from from a specific set stored H5FS file.

    Parameters
    ----------
    f_name : String
        file name
    dataset : String
        name of the set to load (e.g., train, dev, test)
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data_dic : Dictionnary
        dictionary containing the data

        :data: numpy array

            data matrix """
    data_file = h5py.File(f_name, 'r')
    data = data_file[('x_{0}').format(dataset)][:]
    data_file.close()
    if scale:
        print "scaling..."
        data = preprocessing.scale(data, with_mean=False)
    print "Total dataset size:"
    print "n samples: %d" % data.shape[0]
    print "n features: %d" % data.shape[1]

    if rnd:
        print "Radomizing..."
        np.random.shuffle(data)
    data_dic = dict(
        x=data,
    )
    return data_dic


def load_all_data(f_name, scale=True, rnd=False):
    """Get data from from all sets stored H5FS file.

    Parameters
    ----------
    f_name : String
        file name
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data_dic : Dictionnary
        dictionary containing the data

        :x_train: numpy array

            train data matrix
        :x_test: numpy array

            test data matrix
        :x_dev: numpy array

            dev data matrix  """
    data_file = h5py.File(f_name, 'r')
    x_test = data_file['x_test'][:]
    x_dev = data_file['x_dev'][:]
    x_train = data_file['x_train'][:]
    data_file.close()
    if scale:
        print "scaling..."
        x_test = preprocessing.scale(x_test, with_mean=False)
        x_dev = preprocessing.scale(x_dev, with_mean=False)
        x_train = preprocessing.scale(x_train, with_mean=False)
    print "Total dataset size:"
    print "n train samples: %d" % x_train.shape[0]
    print "n test samples: %d" % x_test.shape[0]
    print "n dev samples: %d" % x_dev.shape[0]
    print "n features: %d" % x_test.shape[1]
    if rnd:
        print "Radomizing training set..."
        np.random.shuffle(x_train)

    data_dict = dict(
        x_train=x_train,
        x_test=x_test,
        x_dev=x_dev,
    )
    return data_dict


def load_labels(f_name, dataset):
    """Get labels for a specific set.

    Parameters
    ----------
    f_name : String
        file name
    dataset : String
        name of the set to load (e.g., train, dev, test)


    Returns
    -------
    lbl_dic : Dictionnary
        dictionary containing the labels

        :labels: numpy array

            labels vector """
    data_file = h5py.File(f_name, 'r')
    labels = data_file[('y_{0}').format(dataset)][:]
    data_file.close()
    print "Total dataset size:"
    print "n samples: %d" % labels.shape[0]
    lbl_dict = dict(
        y=labels,
    )
    return lbl_dict


def load_fids(f_name, dataset):
    """Get file ids for a specific set.

    Parameters
    ----------
    f_name : String
        file name
    dataset : String
        name of the set to load (e.g., train, dev, test)


    Returns
    -------
    fids_dic : Dictionnary
        dictionary containing the files ids

        :file_ids: numpy array

            file ids vector """
    data_file = h5py.File(f_name, 'r')
    file_ids = data_file[('file {0}').format(dataset)][:]
    data_file.close()
    print "Total dataset size:"
    print "n samples: %d" % file_ids.shape[0]
    fids_dic = dict(
        f=file_ids,
    )
    return fids_dic


def load_all_labels(f_name):
    """Get labels for all sets.

    Parameters
    ----------
    f_name : String
        file name


    Returns
    -------
    lbl_dic : Dictionnary
        dictionary containing the data

        :y_train: numpy array

            train labels vector
        :y_test: numpy array

            test labels vector
        :y_dev: numpy array

            dev labels vector  """
    data_file = h5py.File(f_name, 'r')
    y_test = data_file['y_test'][:]
    y_dev = data_file['y_dev'][:]
    y_train = data_file['y_train'][:]
    data_file.close()
    print "Total dataset size:"
    print "n train samples: %d" % y_train.shape[0]
    print "n test samples: %d" % y_test.shape[0]
    print "n dev samples: %d" % y_dev.shape[0]

    lbl_dic = dict(
        y_train=y_train,
        y_test=y_test,
        y_dev=y_dev,
    )
    return lbl_dic


def load_all_fids(f_name):
    """Get file ids for all sets.

    Parameters
    ----------
    f_name : String
        file name


    Returns
    -------
    fids_dic : Dictionnary
        dictionary containing the data

        :f_train: numpy array

            train file ids vector
        :f_test: numpy array

            test file ids vector
        :f_dev: numpy array

            dev file ids vector"""
    data_file = h5py.File(f_name, 'r')
    f_test = data_file['file test'][:]
    f_dev = data_file['file dev'][:]
    f_train = data_file['file train'][:]
    data_file.close()
    print "Total dataset size:"
    print "n train samples: %d" % f_train.shape[0]
    print "n test samples: %d" % f_test.shape[0]
    print "n dev samples: %d" % f_dev.shape[0]

    fids_dic = dict(
        f_train=f_train,
        f_test=f_test,
        f_dev=f_dev,
    )
    return fids_dic


def load_data_labels(f_name, dataset, scale=True, rnd=False):
    """Get data with labels, for a particular set.

    Parameters
    ----------
    f_name : String
        file name
    dataset : String
        name of the set to load (e.g., train, dev, test)
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data_dic : Dictionnary
        dictionary containing the data

        :x: numpy array

            data matrix
        :y: numpy array

            labels vector"""
    data = load_data(f_name, dataset, scale)
    labels = load_labels(f_name, dataset)
    if rnd:
        print "Radomizing training set..."
        ind = np.arange(labels['y'].shape[0])
        np.random.shuffle(ind)
        data['x'] = data['x'][ind, ]
        labels['y'] = labels['y'][ind, ]

    data_dic = dict(
        x=data['x'],
        y=labels['y'],
    )
    return data_dic


def load_all_data_labels(f_name, scale=True, rnd=False):
    """Get data with labels, for all sets.

    Parameters
    ----------
    f_name : String
        file name
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data_dic : Dictionnary
        dictionary containing the data

        :x_train: numpy array

            train data matrix
        :x_test: numpy array

            test data matrix
        :x_dev: numpy array

            dev data matrix
        :y_train: numpy array

            train labels vector
        :y_test: numpy array

            test labels vector
        :y_dev: numpy array

            dev labels vector"""
    data = load_all_data(f_name, scale)
    labels = load_all_labels(f_name)
    if rnd:
        print "Radomizing training set..."
        ind = np.arange(labels['y_train'].shape[0])
        np.random.shuffle(ind)
        data['x_train'] = data['x_train'][ind, ]
        labels['y_train'] = labels['y_train'][ind, ]

    data_dic = dict(
        x_train=data['x_train'],
        x_test=data['x_test'],
        x_dev=data['x_dev'],
        y_train=labels['y_train'],
        y_test=labels['y_test'],
        y_dev=labels['y_dev'],
    )
    return data_dic


def load_data_labels_fids(f_name, dataset, scale=True, rnd=False):
    """Get data with labels and file ids for a specific set.

    Parameters
    ----------
    f_name : String
        file name
    dataset : String
        name of the set to load (e.g., train, dev, test)
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data_dic : Dictionnary
        dictionary containing the data

        :x: numpy array

            data matrix
        :y: numpy array

            labels vector
        :f: numpy array

            file ids vector"""
    data = load_data(f_name, dataset, scale)
    labels = load_labels(f_name, dataset)
    fids = load_fids(f_name, dataset)
    if rnd:
        print "Radomizing training set..."
        ind = np.arange(labels['y'].shape[0])
        np.random.shuffle(ind)
        data['x'] = data['x'][ind, ]
        labels['y'] = labels['y'][ind, ]
        fids['f'] = fids['f'][ind, ]

    data_dic = dict(
        x=data['x'],
        y=labels['y'],
        f=fids['f']
    )
    return data_dic


def load_all_data_labels_fids(f_name, scale=True, rnd=False):
    """Get data with labels and file ids for all sets.

    Parameters
    ----------
    f_name : String
        file name
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data_dic : Dictionnary
        dictionary containing the data

        :x_train: numpy array

            train data matrix
        :x_test: numpy array

            test data matrix
        :x_dev: numpy array

            dev data matrix
        :y_train: numpy array

            train labels vector
        :y_test: numpy array

            test labels vector
        :y_dev: numpy array

            dev labels vector
        :f_train: numpy array

            train file ids vector
        :f_test: numpy array

            test file ids vector
        :f_dev: numpy array

            dev file ids vector"""
    data = load_all_data(f_name, scale)
    labels = load_all_labels(f_name)
    fids = load_all_fids(f_name)
    if rnd:
        print "Radomizing training set..."
        ind = np.arange(labels['y_train'].shape[0])
        np.random.shuffle(ind)
        data['x_train'] = data['x_train'][ind, ]
        labels['y_train'] = labels['y_train'][ind, ]
        fids['f'] = fids['f'][ind, ]

    data_dic = dict(
        x_train=data['x_train'],
        x_test=data['x_test'],
        x_dev=data['x_dev'],
        y_train=labels['y_train'],
        y_test=labels['y_test'],
        y_dev=labels['y_dev'],
        f_train=fids['f_train'],
        f_test=fids['f_test'],
        f_dev=fids['f_dev'],
    )
    return data_dic


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape


    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


def reorder_cls_ses(data, cls, ses, with_index=False):
    """reorder the data such that there is only
    one continuous bloc for each pair class/session

    Parameters
    ----------
    data : array
        the data
    cls : array
        the class labels for the data
    ses : array
        the session label for the data
    with_index : Boolean (default False)
        if True, the function returns the reordered indexes together
        with data and labels

    Returns
    -------
    data : array with the same shape as data
        reordered data
    cls : array with the same shape as cls
        reordered class labels
    ses : array with the same shape as ses
        reordered session labels
    ind : array with the same shape as data.shape[1]
        reordered indexes (only if with_index==True)
    """

    data_ordered = np.zeros((data.shape))
    cls_ordered = np.zeros((cls.shape))
    ses_ordered = np.zeros((ses.shape))
    if with_index:
        index = np.arange((data.shape[1],))
        index_ordered = np.zeros((index.shape))
    data_fill = 0
    for i in more_itertools.unique_everseen(itertools.izip(cls, ses)):
        ind = np.where((cls == i[0]) & (ses == i[1]))[0]
        bloc_length = data[(cls == i[0]) & (ses == i[1]), :].shape[0]
        data_ordered[data_fill:data_fill+bloc_length, ] = data[ind, :]
        cls_ordered[data_fill:data_fill+bloc_length] = cls[ind]
        ses_ordered[data_fill:data_fill+bloc_length] = ses[ind]
        if with_index:
            index_ordered[data_fill:data_fill+bloc_length] = index[ind]
        data_fill += bloc_length
    if with_index:
        return {
            'data': data_ordered,
            'cls': cls_ordered,
            'ses': ses_ordered,
            'ind': index_ordered}
    else:
        return {
            'data': data_ordered,
            'cls': cls_ordered,
            'ses': ses_ordered}


def truncate(data, cls_label, ses_label, ind):
    """Truncate data and labels to the legnth specified in ind

    Parameters
    ----------
    data : array
    cls_label : array
        the class labels for the data
    ses_label : array
        the session label for the data
    ind: array
        start and stop indices for the truncation

    Returns
    -------
    data_trunc : array, same shape as (ind[1]-ind[0], data.shape.[1])
        truncateded data
    cls_ordered : array, same shape as (ind[1]-ind[0], cls_label.shape.[1])
        truncated class labels
    ses_ordered : array, same shape as (ind[1]-ind[0], ses_label.shape.[1])
        truncated session labels
    ind : array
        truncation indices"""
    newlen = np.sum(ind[:, 1]-ind[:, 0])
    data_trunc = np.zeros((newlen, data.shape[1]))
    cls_trunc = np.zeros((newlen,))
    ses_trunc = np.zeros((newlen,))
    current_ind = 0
    for i in range(ind.shape[0]):
        bloc_len = ind[i, 1]-ind[i, 0]
        data_trunc[current_ind: current_ind+bloc_len, ] = data[ind[i, 0]:
                                                               ind[i, 1], ]
        cls_trunc[current_ind: current_ind+bloc_len, ] = cls_label[ind[i, 0]:
                                                                   ind[i, 1], ]
        ses_trunc[current_ind: current_ind+bloc_len, ] = ses_label[ind[i, 0]:
                                                                   ind[i, 1], ]
        ind[i, 0] = current_ind
        ind[i, 1] = current_ind+bloc_len
        current_ind += bloc_len
    return {
        'data': data_trunc,
        'cls': cls_trunc,
        'ses': ses_trunc,
        'ind': ind}


def norm_col(w, h):
    """normalize the column vector w (Theano function).
    Apply the invert normalization on h such that w.h does not change

    Parameters
    ----------
    w: Theano vector
        vector to be normalised
    h: Ttheano vector
        vector to be normalised by the invert normalistation

    Returns
    -------
    w : Theano vector with the same shape as w
        normalised vector (w/norm)
    h : Theano vector with the same shape as h
        h*norm
    """
    norm = w.norm(2, 0)
    eps = 1e-12
    size_norm = (T.ones_like(w)).norm(2, 0)
    w = ifelse(T.gt(norm, eps),
               w/norm,
               (w+eps)/(eps*size_norm).astype(theano.config.floatX))
    h = ifelse(T.gt(norm, eps),
               h*norm,
               (h*eps*size_norm).astype(theano.config.floatX))
    return w, h


def get_norm_col(w):
    """returns the norm of a column vector

     Parameters
    ----------
    w: 1-dimensionnal array
        vector to be normalised

    Returns
    -------
    norm: scalar
        norm-2 of w
    """
    norm = w.norm(2, 0)
    return norm[0]
