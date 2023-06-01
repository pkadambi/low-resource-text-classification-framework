from sklearnex import patch_sklearn
patch_sklearn()
import torch
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError, CalibrationError
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, accuracy_score
from sklearn.preprocessing import label_binarize
from torchmetrics.classification import CalibrationError
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from lrtc_lib.metrics.nn_calib_metrics import confECE, calc_MCE

def bin_class_probas_and_preds(y_true, y_prob, nclasses, n_bins, strategy='quantile'):
    '''
    y_true - vector list of class labels by numerical class
    y_probas - shape [n_samples x 1]
    '''
    labels = np.unique(y_true)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )
    binids = np.digitize(y_prob, bins) - 1

    correct_prediction = ((y_prob > 1 / nclasses) == y_true).astype('float')
    bin_pred_prob = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_empirc_prob = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_empirc_prob[nonzero] / bin_total[nonzero]
    prob_pred = bin_pred_prob[nonzero] / bin_total[nonzero]
    bin_fracs = bin_total[nonzero] / sum(bin_total)

    return prob_true, prob_pred, bin_fracs

def bin_class_conf_and_acc(y_true, y_prob, nclasses, n_bins, strategy='quantile'):
    acc_y=None
    conf_x=None
    bin_fracs=None
    return acc_y, conf_x, bin_fracs

def binary_cw_ece(y_true, y_probas, nclasses, n_bins=10, strategy='quantile'):
    '''
    y_true - vector list of labels by numerical class
    y_probas - vector of probabilities of class 1
    nclasses - this argument exists so that this function can be used for 1-vs-rest ece
    '''
    y_prob = np.copy(y_probas)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. Provided labels %s." % labels
        )

    acc_y, conf_x, bin_fracs = bin_class_probas_and_preds(y_true, y_probas, nclasses=nclasses,
                                                                     n_bins=n_bins, strategy=strategy)

    return mean_absolute_error(conf_x, acc_y, sample_weight=bin_fracs)

def calc_confid_ece(y_true, y_prob, nclasses, n_bins=10,strategy='quantile'):
    '''
    y_true - vector of class labels by numerical class. ex [3 4 2 5 9 0 9 2 4 ...], size [n_samples]
    y_prob - [n_samples x n_classes], sum along n_classes dimension should be 1
    '''
    labels = np.unique(y_true)
    ytrue_oh = to_one_hot(y_true)
    if len(labels) == 1:
        raise Exception('Error only one class in y_true labels.')
    elif len(labels) == 2:
        return binary_cw_ece(ytrue_oh[:, 1], y_prob[:, 1], n_bins=n_bins,
                             nclasses=nclasses, strategy=strategy)
    else:
        classwise_ece = np.zeros_like(labels).astype('float')
        for ii in range(len(labels)):
            class_ytrue = label_binarize(ytrue_oh[:, ii], classes=labels)[:, ii]
            class_posterior = y_prob[:, ii]
            # prob_true, prob_pred, accuracy, bin_fracs = bin_class_probas_and_preds(nclasses=nclasses, n_bins=n_bins,
            #                                                                        strategy=strategy)
            classwise_ece[ii] = binary_cw_ece(class_ytrue, class_posterior, n_bins=n_bins,
                                              nclasses=nclasses, strategy=strategy)
        return np.mean(classwise_ece)

def binary_conf_ece(y_true, y_probas, nclasses, n_bins=10, strategy='quantile'):
    # TODO: implement confidence ece function
    '''
    y_true - vector list of labels by numerical class
    y_probas - vector of probabilities of class 1
    '''
    y_prob = np.copy(y_probas)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. Provided labels %s." % labels
        )

    prob_true, prob_pred, bin_fracs = bin_class_conf_and_acc(y_true, y_probas, nclasses=nclasses,
                                                                     n_bins=n_bins, strategy=strategy)

    return mean_absolute_error(prob_true, prob_pred, sample_weight=bin_fracs)


def calc_classwise_ece(y_true, y_prob, nclasses, n_bins=10, strategy='quantile'):
    '''
    y_true - vector of class labels by numerical class. ex [3 4 2 5 9 0 9 2 4 ...], size [n_samples]
    y_prob - [n_samples x n_classes], sum along n_classes dimension should be 1
    '''
    labels = np.unique(y_true)
    ytrue_oh = to_one_hot(y_true)
    if len(labels) == 1:
        raise Exception('Error only one class in y_true labels.')
    elif len(labels) == 2:
        return binary_cw_ece(ytrue_oh[:, 1], y_prob[:, 1], nclasses=nclasses,
                               n_bins=n_bins, strategy=strategy)
    else:
        classwise_ece = np.zeros_like(labels).astype('float')
        for ii in range(len(labels)):
            class_ytrue = label_binarize(ytrue_oh[:, ii], classes=labels)[:, ii]
            class_posterior = y_prob[:, ii]
            # prob_true, prob_pred, accuracy, bin_fracs = bin_class_probas_and_preds(nclasses=nclasses, n_bins=n_bins,
            #                                                                        strategy=strategy)
            classwise_ece[ii] = binary_cw_ece(class_ytrue, class_posterior, nclasses=nclasses,
                                                n_bins=n_bins, strategy=strategy)
        return np.mean(classwise_ece)

def to_one_hot(labels):
    '''

    :param labels: should have astype 'int'
    :return:
    '''
    nclasses = np.max(labels) + 1
    return np.eye(nclasses)[labels].astype('int')


def calc_brier_score(y_true, y_prob):
    '''

    y_true - class membership. ex. [0 1 1 2 0 0 3 4 9 2]
    y_pred - predicted probas. shape [n_samples, n_classes]

    '''
    if y_prob.shape[1] == 2:
        _ytrue = y_true
        _yprob = y_prob[:, 1]
    else:
        _ytrue = to_one_hot(y_true.astype('int'))
        _yprob = y_prob

    return 2 * mean_squared_error(_ytrue, _yprob)
    # if len(y_true.shape)>1:
    # return sum(((qwer - asdf)**2).sum(axis=1))
    # else:
    # mean_squared_error(y_true, y_prob)


def calc_log_loss(y_true, y_pred):
    '''
    y_true - class membership. ex. [0 1 1 2 0 0 3 4 9 2]
    y_pred - predicted probas. shape [n_samples, n_classes]
    '''
    return log_loss(y_true, y_pred)

def calc_ece(y_true, y_prob, nclasses, n_bins=10):
    '''
    y_true - list of labels
    y_prob - n_samples x n_classes
    '''
    ypreds = torch.Tensor(y_prob)
    targets = torch.Tensor(y_true)
    assert y_prob.shape[1]==nclasses, f'Mismatch error. y_prob contains {y_prob.shape[1]} classes, but nclasses {nclasses}'

    if nclasses == 1:
        raise ValueError('More than one unique class should be in y_true labels.')
    elif nclasses == 2:
        ece_metric = CalibrationError(task='binary', num_classes=2, n_bins=n_bins)
        ypreds = ypreds[:, 1]
    elif nclasses>2:
        ece_metric = CalibrationError(task='multiclass', n_bins=n_bins)

    return float(ece_metric(ypreds, targets).cpu().numpy())

def return_all_calibration_metrics(y_true, y_pred, n_bins, nclasses, ytrue_post = None, binning_method='quantile'):
    '''
    Outputs a dictionary with:
    - log loss
    - brier
    - epe
    - cwepe
    '''
    y_pred = np.array(y_pred)
    if type(y_true[0])==str:
        y_true = [tr.lower() for tr in y_true]
        yt = np.array(y_true)
        yt[yt=='false'] = 0
        yt[yt=='true'] = 1
        y_true = np.array(yt.astype('float'))
    calib_metrics = {}
    try:
        calib_metrics['BrierScore'] = calc_brier_score(y_true, y_pred)
    except:
        # print()
        calib_metrics['BrierScore'] = np.nan
    try:
        calib_metrics['LogLoss'] = calc_log_loss(y_true, y_pred)
    except:
        calib_metrics['LogLoss'] = np.nan

    try:
        # calib_metrics['confECE'] = calc_confid_ece(y_true, y_pred, n_bins=n_bins,
        #                                            nclasses=nclasses, strategy=binning_method)
        calib_metrics['confECE'] = confECE(y_pred, y_true, nbins=n_bins)
    except:
        calib_metrics['confECE'] = np.nan
        # print('nanECE')

    try:
        calib_metrics['cwECE'] = calc_classwise_ece(y_true=y_true.astype('int'), y_prob=y_pred, nclasses=nclasses, n_bins=n_bins)
    except:
        calib_metrics['cwECE'] = np.nan
    #TODO: explore if MCE should be used as a calibration metric
    # try:
    #     calib_metrics['MCE'] = calc_MCE(y_pred, y_true, nbins=n_bins)
    # except:
    #     calib_metrics['MCE'] = np.nan

    #TODO: implement for multi-class
    try:
        # calib_metrics['Acc'] = sum(y_true==np.argmax(y_pred, axis=1))/len(y_true)
        calib_metrics['Acc'] = accuracy_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1)) * 100
        print(calib_metrics['Acc'])
    except:
        calib_metrics['Acc'] = np.nan

    if ytrue_post is not None:
        try:
            calib_metrics['MSE'] = sum((ytrue_post - y_pred[:, 1])**2)/len(y_pred)
        except:
            calib_metrics['MSE'] = np.nan
    else:
        calib_metrics['MSE'] = np.nan

    return calib_metrics

