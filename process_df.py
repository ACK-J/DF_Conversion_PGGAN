import os
import re
import numpy as np
import tqdm as tqdm
import pickle
import statistics
from keras.models import load_model

"""
Description: This file provides functions to convert the Deep Fingerprinting
             dataset into different, potentially more efficient version.
Usage:      load_data('/data/website-fingerprinting/datasets/undefended', max_length=1000, max_instances=1000)
"""


def process_trace(trace, style='direction', **kwargs):
    """

    :param trace:
    :param style:
    :param kwargs:
    :return:
    """

    def process_by_direction(trace):
        rep = [0, 0]
        burst_count = 0
        for i in range(len(trace[2])):
            direction = trace[2][i]
            if direction > 0:
                if len(rep) // 2 == burst_count:
                    rep.extend([0, 0])
                rep[-2] += 1
            else:
                if len(rep) // 2 != burst_count:
                    burst_count += 1
                rep[-1] += 1
        return rep

    def process_by_iat(trace, threshold=0.003):
        burst_seq = [0, 0]
        direction = trace[2][0]
        if direction > 0:
            burst_seq[0] += 1
        else:
            burst_seq[1] += 1
        for i in range(1, len(trace[0])):
            iat = trace[0][i] - trace[0][i - 1]
            if iat > threshold:
                burst_seq.extend([0, 0])
            if trace[2][i] > 0:
                burst_seq[len(burst_seq) - 2] += 1
            else:
                burst_seq[len(burst_seq) - 1] += 1
        return burst_seq

    def process_by_timeslice(trace, interval=0.05):
        burst_seq = [0, 0]
        s = 0
        for i in range(len(trace[0])):
            timestamp = trace[0][i]
            if timestamp // interval > s:
                s = timestamp // interval
                burst_seq.extend([0, 0])
            direction = trace[2][i]
            if direction > 0:
                burst_seq[len(burst_seq) - 2] += 1
            else:
                burst_seq[len(burst_seq) - 1] += 1
        return burst_seq

    # process into burst sequences using IATs
    if style == 'iat':
        rep = process_by_iat(trace)
        return rep

    # process into bursts by timeslice
    if style == 'time':
        rep = process_by_timeslice(trace)
        return rep

    # process into bursts by direction
    if style == 'direction':
        rep = process_by_direction(trace)
        return rep

def load_trace(fi, separator="\t", filter_by_size=False):
    """
    loads data to be used for predictions
    """
    sequence = [[], [], []]
    for line in fi:
        pieces = line.strip("\n").split(separator)
        if int(pieces[1]) == 0:
            break
        timestamp = float(pieces[0])
        length = abs(int(pieces[1]))
        direction = int(pieces[1]) // length
        if filter_by_size:
            if length > 512:
                sequence[0].append(timestamp)
                sequence[1].append(length)
                sequence[2].append(direction)
        else:
            sequence[0].append(timestamp)
            sequence[1].append(length)
            sequence[2].append(direction)
    return sequence


def load_data(directory, max_length=None, separator='\t', fname_pattern=r"(\d+)[-](\d+)", max_classes=99999,
              min_instances=0, max_instances=500, **t_kwargs):
    """
    Load feature files from a directory.
    Parameters
    ----------
    directory : str
        System file path to a directory containing feature files.
    max_length : int
        The maximum size of the new vector generated
    separator : str
        Character string used to split features in the feature files.
    fname_pattern : raw str
        Character string used to split feature file names.
        First substring identifies the class, while the second substring identifies the instance number.
        Instance number is ignored.
    max_classes : int
        Maximum number of classes to load.
    min_instances : int
        Minimum number of instances acceptable per class.
        If a class has less than this number of instances, the all instances of the class are discarded.
    max_instances : int
        Maximum number of instances to load per class.
    Returns
    -------
    ndarray
        Numpy array of Nxf containing site visit feature instances.
    ndarray
        Numpy array of Nx1 containing the labels for site visits.
    """
    # load pickle file if it exist
    X = []  # feature instances
    Y = []  # site labels
    for root, dirs, files in os.walk(directory):

        # filter for trace files
        files = [fname for fname in files if re.match(fname_pattern, fname)]

        # read each feature file as CSV
        class_counter = dict()  # track number of instances per class
        # Error check just incase no files are found
        if files != []:
            for fname in tqdm.tqdm(files, total=len(files)):

                # get trace class
                cls = int(re.match(fname_pattern, fname).group(1))

                # skip if maximum number of instances reached
                if class_counter.get(cls, 0) >= max_instances:
                    continue

                # skip if maximum number of classes reached
                if int(cls) >= max_classes:
                    continue

                with open(os.path.join(root, fname), "r") as fi:

                    # load the trace file
                    trace = load_trace(fi, separator, filter_by_size=False)

                    # process trace file to sequence representation
                    rep = np.array(process_trace(trace, **t_kwargs))

                    # pad trace length with zeros
                    if max_length is not None:
                        if len(rep) < max_length:
                            rep = np.hstack((rep, np.zeros((max_length - len(rep),))))
                        rep.resize((max_length, 1))

                    X.append(rep)
                    Y.append(cls)
                    class_counter[int(cls)] = class_counter.get(cls, 0) + 1
    # trim data to minimum instance count
    counts = {y: Y.count(y) for y in set(Y)}
    new_X, new_Y = [], []
    for x, y in zip(X, Y):
        if counts[y] >= min_instances:
            new_Y.append(y)
            new_X.append(x)
    X, Y = new_X, new_Y

    # adjust labels such that they are assigned a number from 0..N
    # (required when labels are non-numerical or does not start at 0)
    # try to keep the class numbers the same if numerical
    labels = list(set(Y))
    labels.sort()
    d = dict()
    for i in range(len(labels)):
        d[labels[i]] = i
    Y = list(map(lambda x: d[x], Y))

    X, Y = np.array(X), np.array(Y)

    print(X)
    # normalize data to between -1 and 1
    #xmax = np.amax(np.abs(X.ravel()))
    #X /= xmax
    X = np.clip(X, 0,255)
    # return X and Y as numpy arrays
    #return X, Y, xmax
    return X, Y

def conv_to_32_x_32(x, num):
    """
    Converting a supplied numpy array into a 32x32 image and saving it
    """
    from PIL import Image
    arr = x.astype('uint8')
    arr = arr.reshape(32, 32)
    print(arr)
    print(arr.shape)
    im = Image.fromarray(arr)
    im.save("save/test" + str(num) + ".png")


def reverse_gan_image(directory):
    """
    Directory: The directory with all the png files generated by the gan
    """
    from PIL import Image
    #  Create a list of all .png files within the specifed dir
    files = [f for f in os.listdir(directory) if ".png" in f]
    df_data = []

    # Convert each image back to np array form
    for picture in files:
        vector = np.asarray(Image.open(directory + picture))
        vector = vector.reshape(1024, 1)
        vector = vector[:1000]
        df_data.append(vector)
    return np.stack(df_data, axis=0)

def predict_image(array, path_to_df):
    """
    Takes in the path to a pre-trained df model and the array
    of data you want to run predictions on
    """
    # Load in the model
    model = load_model(path_to_df)

    # Run the data through a forward pass and get the predictions
    softmax_out = model.predict(array)
    # Find the highest prediction
    return np.argmax(softmax_out, axis=1)

def compute_distance(s1, s2, metric='cosine'):
    """
    Parameters
    ----------
    s1 : numpy array
        The numpy array representing the first sample for which to compute distance.
    s2 : numpy array
        The numpy array representing the second sample for which to compute distance.
    metric : string
        The type of distance to calculate. By default, use the cosine distance.
    Returns
    -------
    float
        A singular float value representing the computed distance between the
        provided samples.
    """
    from sklearn.metrics import pairwise_distances
    s1, s2 = s1.flatten(), s2.flatten()
    s1 = np.reshape(s1, (1, *s1.shape))
    s2 = np.reshape(s2, (1, *s2.shape))
    res = pairwise_distances(s1, s2, metric=metric)
    return res.flatten()[0]


if __name__ == '__main__':
    ######################
    # Loading in Samples #
    ######################
    # out = open("X", 'wb')
    # out2 = open("y", 'wb')
    # X = pickle.load(out)
    # Y = pickle.load(out2)
    # X, Y = load_data(directory=r'/home/jack/undefended', max_length=1024, max_instances=1000)
    # print(X)
    # print(X.shape)
    # pickle.dump(X, out)
    # pickle.dump(Y, out2)
    # out.close()
    # out2.close()


    ###########################################
    # Reading Samples and Converting to 32x32 #
    ###########################################
    # out = open("X", 'rb')
    # out2 = open("y", 'rb')
    # X = pickle.load(out) # (95000, 1024, 1)
    # Y = pickle.load(out2) # (95000, )
    # out.close()
    # out2.close()
    # for num, row in enumerate(X):
    #     conv_to_32_x_32(row, num)


    ####################
    # Test Fake Images #
    ####################
    # The directory that holds the fake images
    directory = "../progressive_growing_of_gans/jacksdataset/001-fake-images-0/"
    # Reverse the 32x32 images into (num_imgs, 1000, 1)
    fake_data = reverse_gan_image(directory)
    print("\n\n", fake_data)
    print(fake_data.shape)
    #  Use the model to predict on the fake_data
    predictions = predict_image(fake_data, "DF.h5")
    print(predictions)

    ######################################
    # Calculate distance between samples #
    ######################################
    '''
    notation:
    D( ) - distance of
    R - real samples
    F - fake samples
    R[x] - sample with label of x
    D( F[a], R[a] )  - calc distances between fake and real samples of same class
    D( R[a], R[a] ) - calc distances between real samples of sample class
    D( F[a], F[b] ) - calc distances between fake samples of different classes
    D( R[a], R[b] ) - calc distances between real samples of different class
    '''

    # Load in the real data
    out = open("X", 'rb')
    out2 = open("y", 'rb')
    X = pickle.load(out) # (95000, 1024, 1)
    # Trim the data to match our generated data
    X = X[:, :1000, :]  # (95000, 1000, 1)
    Y = pickle.load(out2) # (95000, )
    print(Y[:10])

    # D( F[a], R[a] )  - calc distances between fake and real samples of same class
    print("D( F[a], R[a] )")
    metrics = []
    max = fake_data.shape[0]
    #  Setting the max to prevent indexing errors
    if X.shape[0] < fake_data.shape[0]:
        max = fake_data.shape[0]
    #  Iterate through the samples
    for i in range(max):
        n = 0
        while predictions[i] != Y[n] and n < len(Y):
            n += 1
        if predictions[i] == Y[n]:
            dist = compute_distance(fake_data[i], X[n])
            print("Distance between FAKE sample " + str(i) + " and REAL sample " + str(n) + ": ", dist)
            metrics.append(dist)
    print("Num: ", len(metrics))
    print("Stdev: ", statistics.stdev(metrics))
    print("Mean: ", statistics.mean(metrics))
    print("Median: ", statistics.median(metrics))
    print("Variance: ", statistics.variance(metrics))

    print()

    # D( R[a], R[a] ) - calc distances between real samples of sample class
    print("D( R[a], R[a] )")
    metrics = []
    #  Iterate through every sample in the real dataset
    #for i in range(X.shape[0]):
    for i in range(10):
        n = 0
        #  While the current label != another label in the set -> increment
        while n == i or Y[i] != Y[n] and n < len(Y):
            n += 1 # This will find another sample with the same label
        # Make sure its the same class
        if Y[i] == Y[n]:
            dist = compute_distance(X[i], X[n])
            print("Distance between sample " + str(i) + " and " + str(n) + ": ", dist)
            metrics.append(dist)
    print("Num: ", len(metrics))
    print("Stdev: ", statistics.stdev(metrics))
    print("Mean: ", statistics.mean(metrics))
    print("Median: ", statistics.median(metrics))
    print("Variance: ", statistics.variance(metrics))

    print()

    # D( F[a], F[b] ) - calc distances between fake samples of different classes
    print("D( F[a], F[b] )")
    metrics = []
    for i in range(fake_data.shape[0]):
        if i == fake_data.shape[0] - 1:
            dist = compute_distance(fake_data[i], fake_data[0])
            print("Distance between sample " + str(i) + " and 0: ", dist)
            metrics.append(dist)
        else:
            dist = compute_distance(fake_data[i], fake_data[i+1])
            print("Distance between sample " + str(i) + " and " + str(i+1) + ": ", dist)
            metrics.append(dist)
    print("Num: ", len(metrics))
    print("Stdev: ", statistics.stdev(metrics))
    print("Mean: ", statistics.mean(metrics))
    print("Median: ", statistics.median(metrics))
    print("Variance: ", statistics.variance(metrics))

    print()

    # D( R[a], R[b] ) - calc distances between real samples of different class
    print("D( R[a], R[b] )")
    metrics = []
    #for i in range(X.shape[0]):
    for i in range(10):
        if i == X.shape[0] - 1:
            dist = compute_distance(X[i], X[0])
            print("Distance between sample " + str(i) + " and 0: ", dist)
            metrics.append(dist)
        else:
            dist = compute_distance(X[i], X[i+1])
            print("Distance between sample " + str(i) + " and " + str(i+1) + ": ", dist)
            metrics.append(dist)
    print("Num: ", len(metrics))
    print("Stdev: ", statistics.stdev(metrics))
    print("Mean: ", statistics.mean(metrics))
    print("Median: ", statistics.median(metrics))
    print("Variance: ", statistics.variance(metrics))
