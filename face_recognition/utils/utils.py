import os
import numpy as np
import time
import torch


def check_dir_format(dir):
    if dir.endswith(os.path.sep):
        return dir
    else:
        return dir+os.path.sep


def str2list(string, sperator=','):
    li = list(map(int, string.split(sperator)))
    return li


def getAccuracy(scores, flags, threshold, method):
    if method == 'l2_distance':
        p = np.sum(scores[flags == 1] < threshold)
        n = np.sum(scores[flags == -1] > threshold)
    elif method == 'cos_distance':
        p = np.sum(scores[flags == 1] > threshold)
        n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum, method):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 3.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i], method)
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


def evaluation_10_fold(featureLs, featureRs, fold, flags, method='l2_distance'):
    ### Evaluate the accuracy ###
    ACCs = np.zeros(10)
    threshold = np.zeros(10)
    fold = fold.reshape(1, -1)
    flags = flags.reshape(1, -1)

    for i in range(10):

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        if method == 'l2_distance':
            scores = np.sum(np.power((featureLs - featureRs), 2), 1)  # L2 distance
        elif method == 'cos_distance':
            scores = np.sum(np.multiply(featureLs, featureRs), 1)  # cos distance

        threshold[i] = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000, method)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold[i], method)

    return np.mean(ACCs) * 100, np.mean(threshold)


def append_path_by_date(model_ckpt_dir):
    os.environ['TZ'] = 'Asia/Hong_Kong'
    time.tzset()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    return check_dir_format(model_ckpt_dir) + timestr + os.path.sep


def model_ckpt_path(model_ckpt_dir, format='/model-{epoch:02d}-{val_loss:.2f}'):
    os.environ['TZ'] = 'Asia/Hong_Kong'
    time.tzset()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    return model_ckpt_dir + timestr + format


if __name__ == "__main__":
    pass