import pickle

import torchvision.transforms as transforms
from skimage.feature import hog
from sklearn import svm

from style_classification.hinge_utils import *


def get_hog(set):
    # get histogram of oriented gradients for a data split set
    set = [hog(image, orientations=8, pixels_per_cell=(4, 4),
               cells_per_block=(1, 1)) for image in set]
    return set


# class to binarize image (so that 0 is background and 1 is a character)
class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x < self.thr).to(x.dtype)  # do not change the data type


'''
Fits the SVM to the training data, calculates accuracy on testing data and saves the trained SVM
'''


def get_acc_SVM(dataset_train, dataset_test):
    print("Running SVM for Style")
    name2idxStyle = {'archaic': 0, 'hasmonean': 1, 'herodian': 2}

    bin_transform = transforms.Compose([
        transforms.ToTensor(),
        ThresholdTransform(thr_255=200)
    ])

    train_x, train_y = [], []
    # calculate all codebook vectors and store them in dict
    for stylename, styledataset in dataset_train.items():
        for label, characterset in styledataset.items():
            for char in characterset:
                char, _ = noise_removal(char)
                char = bin_transform(char).numpy()
                train_x.append(np.asarray(char[0]))
                train_y.append(name2idxStyle[stylename])

    train_x = np.asarray(get_hog(train_x))

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_x, train_y)

    test_x, test_y = [], []
    for stylename, styledataset in dataset_test.items():
        for label, characterset in styledataset.items():
            for char in characterset:
                char = bin_transform(char).numpy()
                test_x.append(np.asarray(char[0]))
                test_y.append(name2idxStyle[stylename])

    test_x = np.asarray(get_hog(test_x))

    predictions = clf.predict(test_x)
    wro, cor = 0, 0
    for j in range(len(test_y)):
        if predictions[j] == test_y[j]:
            cor += 1
        else:
            wro += 1

    print('Accuracy: ', cor / (cor + wro))
    # save model to disk for later use
    filename = '../SVM_for_Style.sav'
    pickle.dump(clf, open(filename, 'wb'))
