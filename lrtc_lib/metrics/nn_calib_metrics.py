'''
Author: Markus Kängsepp (markus93)
'''
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics


def compute_acc_incidence_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels

    Returns:
        (accuracy, avg_conf, len_bin, py_bin): accuracy of bin, confidence, probability of y=1 of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct) / len_bin  # accuracy of BIN
        py_bin = sum([x[1]==1 for x in filtered_tuples])/ len_bin
        return accuracy, avg_conf, len_bin, py_bin

def calc_confidence(probs, normalize=True):
    if normalize:
        confs = np.max(probs, axis=1) / np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence
    return confs

def compute_bin_confs_accs_incidence(conf, pred, true, bin_size=.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    n = len(conf)

    accs = []
    confs = []
    py_bins = []
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin, py_bin = compute_acc_incidence_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        accs.append(acc)
        confs.append(avg_conf)
        py_bins.append(py_bin)

    return accs, confs, py_bins

def ECE(conf, pred, true, bin_size=0.1):
    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin, py_bin = compute_acc_incidence_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE

    return ece

def MCE(conf, pred, true, bin_size=0.1):
    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _, __ = compute_acc_incidence_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc - avg_conf))

    return max(cal_errors)


def confECE(probs, y_true, nbins=10, normalize=True):

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    confs = calc_confidence(probs, normalize=normalize)
    ece = ECE(confs, preds, y_true, bin_size=1 / nbins)
    return ece

def calc_MCE(probs, y_true, nbins=10, normalize=True):
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    confs = calc_confidence(probs, normalize=normalize)
    mce = MCE(confs, preds, y_true, bin_size=1 / nbins)
    return mce



def evaluate(probs, y_true, verbose=False, normalize=False, bins=15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    confs = calc_confidence(probs, normalize=normalize)

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size=1 / bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size=1 / bins)

    loss = log_loss(y_true=y_true, y_pred=probs)

    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    try:
        brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)
    except:
        brier = np.nan

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)

    return (error, ece, mce, loss, brier)


# reliability diagram plotting for subplot case.
# def rel_diagram_sub(accs, confs, ax, M = 10, name = "Reliability Diagram", xname = "", yname=""):
def conf_accuracy_diagram(pred_probas, ylabel, ax, M = 10, title = "Conf-Acc Diagram", xname = "", yname="",
                    normalize=True):
    bin_size=1/M
    confs = calc_confidence(pred_probas, normalize=normalize)
    preds = np.argmax(pred_probas, axis=1)
    accs_binned, confs_binned, __ = compute_bin_confs_accs_incidence(confs, preds, ylabel, bin_size=bin_size)

    acc_conf = np.column_stack([accs_binned,confs_binned])
    acc_conf.sort(axis=1)
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]

    bin_size = 1/M
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

    # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2)

    # Next add error lines
    #for i in range(M):
        #plt.plot([i/M,1], [0, (M-i)/M], color = "red", alpha=0.5, zorder=1)

    #Bars with outputs
    output_plt = ax.bar(positions, outputs, width = bin_size, edgecolor = "black", color = "blue", label="Outputs", zorder = 3)

    # Line plot with center line.
    ax.set_aspect('equal')
    ax.plot([0,1], [0,1], color='k',linestyle = "--")
    ax.legend(handles = [gap_plt, output_plt])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xname, fontsize=14, color = "black")
    ax.set_ylabel(yname, fontsize=14, color = "black")



def conf_pred_diagram(pred_probas, ylabel, ax, M = 10, title = "Conf-Incidence Diagram", xname = "Confidence", yname="P_y True",
                    normalize=True):
    bin_size=1/M
    confs = calc_confidence(pred_probas, normalize=normalize)
    preds = np.argmax(pred_probas, axis=1)
    predprobs = pred_probas[:, 1]
    accs_binned, probs_binned, py_binned = compute_bin_confs_accs_incidence(predprobs, preds, ylabel, bin_size=bin_size)
    print('Probs binned:\t', probs_binned)
    print('Py binned:\t', py_binned)
    # acc_conf = np.column_stack([accs_binned,confs_binned])
    acc_py = np.column_stack([py_binned, probs_binned])
    acc_py.sort(axis=1)
    outputs = acc_py[:, 0]
    gap = acc_py[:, 1]

    bin_size = 1/M
    positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

    # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, gap, width = bin_size, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2)

    # Next add error lines
    #for i in range(M):
        #plt.plot([i/M,1], [0, (M-i)/M], color = "red", alpha=0.5, zorder=1)

    #Bars with outputs
    output_plt = ax.bar(positions, outputs, width = bin_size, edgecolor = "black", color = "blue", label="Outputs", zorder = 3)

    # Line plot with center line.
    ax.set_aspect('equal')
    ax.plot([0,1], [0,1], color='k',linestyle = "--")
    ax.legend(handles = [gap_plt, output_plt])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xname, fontsize=14, color = "black")
    ax.set_ylabel(yname, fontsize=14, color = "black")