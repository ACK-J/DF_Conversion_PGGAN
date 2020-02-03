import os
import re
import numpy as np
import tqdm as tqdm


def process_trace(trace, style='direction', **kwargs):

    def process_by_direction(trace):
        rep = [0, 0]
        burst_count = 0
        for i in range(len(trace[2])):
            direction = trace[2][i]
            if direction > 0:
                if len(rep)//2 == burst_count:
                    rep.extend([0, 0])
                rep[-2] += 1
            else:
                if len(rep)//2 != burst_count:
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
            iat = trace[0][i] - trace[0][i-1]
            if iat > threshold:
                burst_seq.extend([0, 0])
            if trace[2][i] > 0:
                burst_seq[len(burst_seq)-2] += 1
            else:
                burst_seq[len(burst_seq)-1] += 1
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
                burst_seq[len(burst_seq)-2] += 1
            else:
                burst_seq[len(burst_seq)-1] += 1
        return burst_seq


    # process into burst sequences using IATs
    if style=='iat':
        rep = process_by_iat(trace)

    # process into bursts by timeslice
    if style=='time':
        rep = process_by_timeslice(trace)

    # process into bursts by direction
    if style=='direction':
        rep = process_by_direction(trace)



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
    separator : str
        Character string used to split features in the feature files.
    split_at : str
        Character string used to split feature file names.
        First substring identifies the class, while the second substring identifies the instance number.
        Instance number is ignored.
    max_classes : int
        Maximum number of classes to load.
    max_instances : int
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
        for fname in tqdm(files, total=len(files)):

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

    # normalize data to between -1 and 1
    xmax = np.amax(np.abs(X.ravel()))
    X /= xmax

    # return X and Y as numpy arrays
    return X, Y, xmax