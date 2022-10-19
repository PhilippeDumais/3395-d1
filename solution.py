import numpy as np
banknote = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')

######## DO NOT MODIFY THIS FUNCTION ########


def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, banknote):
        without_label = np.delete(banknote, 4, axis=1)
        return np.mean(without_label, axis=0)

    def covariance_matrix(self, banknote):
        without_label = np.delete(banknote, 4, axis=1)
        return np.cov(without_label, rowvar=False)

    def feature_means_class_1(self, banknote):
        arr = np.delete(banknote, np.where(banknote == 0)[0], axis=0)
        without_label = np.delete(arr, 4, axis=1)
        return np.mean(without_label, axis=0)

    def covariance_matrix_class_1(self, banknote):
        arr = np.delete(banknote, np.where(banknote == 0)[0], axis=0)
        without_label = np.delete(arr, 4, axis=1)
        return np.cov(without_label, rowvar=False)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def compute_predictions(self, test_data):
        test_data_len = test_data.shape[0]
        counts = np.ones((test_data_len, len(self.label_list)))
        majority_class = np.zeros(test_data_len)
        d = self.train_inputs.shape[1]
        r = self.h
        for (i, ex) in enumerate(test_data):
            distances = np.sqrt(
                np.sum((ex[:d] - self.train_inputs) ** 2, axis=1))
            indices_in_h = []
            indices_in_h = np.array(
                [k for k in range(len(distances)) if distances[k] <= r])
            if len(indices_in_h) == 0:
                rand = draw_rand_label(ex, self.label_list)
                majority_class[i] = rand
            else:
                for j in indices_in_h:
                    counts[i, int(self.train_labels[j])] += 1
                majority_class[i] = np.argmax(counts[i, :])
        # print(counts)
        return majority_class


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.train_labels = train_labels
        self.train_inputs = train_inputs
        self.label_list = np.unique(train_labels)

    def compute_predictions(self, test_data):
        test_data_len = test_data.shape[0]
        counts = np.zeros((test_data_len, len(self.label_list)))
        majority_class = np.zeros(test_data_len)
        s = self.sigma
        d = self.train_inputs.shape[1]
        base = 1/(s*np.sqrt(2*np.pi))
        for (i, ex) in enumerate(test_data):
            distances = np.sqrt(
                np.sum((ex[:d] - self.train_inputs) ** 2, axis=1))
            exponent = np.square(distances)/(2*np.square(s))
            e = np.e**-exponent
            weights = base*e
            for k in range(self.train_inputs.shape[0]):
                counts[i, int(self.train_labels[k])] += weights[k]
            majority_class[i] = np.argmax(counts[i, :])
        # print(counts)
        return majority_class




def split_dataset(banknote):
    arr = np.arange(banknote.shape[0])
    train_indexes = arr[(arr % 5 == 0) | (arr % 5 == 1) | (arr % 5 == 2)]
    val_indexes = arr[arr % 5 == 3]
    test_indexes = arr[arr % 5 == 4]
    train = []
    val = []
    test = []
    for i, row in enumerate(banknote):
        if i in train_indexes:
            train.append(row)
        if i in val_indexes:
            val.append(row)
        if i in test_indexes:
            test.append(row)

    train_set = np.stack(train, axis=0)
    val_set = np.stack(val, axis=0)
    test_set = np.stack(test, axis=0)
    return (train_set, val_set, test_set)


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hard = HardParzen(h)
        hard.train(self.x_train, self.y_train)
        predicted = hard.compute_predictions(self.x_val)
        error_counter = 0
        for i in range(len(predicted)):
            if predicted[i] != self.y_val[i]:
                error_counter += 1
        error_percent = error_counter/len(predicted)
        return error_percent


    def soft_parzen(self, sigma):
        soft = SoftRBFParzen(sigma)
        soft.train(self.x_train, self.y_train)
        predicted = soft.compute_predictions(self.x_val)
        error_counter = 0
        for i in range(len(predicted)):
            if predicted[i] != self.y_val[i]:
                error_counter += 1
        error_percent = error_counter/len(predicted)
        return error_percent


def get_test_errors(banknote):
    sets = split_dataset(banknote)
    train_inputs = sets[0][:, :-1]
    train_labels = sets[0][:, -1]
    val_inputs = sets[1][:, :-1]
    val_labels = sets[1][:, -1]
    test_inputs = sets[2][:, :-1]
    test_labels = sets[2][:, -1]
    err = ErrorRate(train_inputs, train_labels, val_inputs, val_labels)
    hard_errors = []
    soft_errors = []
    list_of_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    for value in list_of_values:
        hard_errors.append(err.hard_parzen(value))
        soft_errors.append(err.soft_parzen(value))
    hstar = list_of_values[np.argmin(hard_errors)]
    sstar = list_of_values[np.argmin(soft_errors)]
    test_err = ErrorRate(train_inputs, train_labels, test_inputs, test_labels)
    return [test_err.hard_parzen(hstar), test_err.soft_parzen(sstar)]


def random_projections(X, A):
    pass


def test_predictions():
    sets = split_dataset(banknote)
    train = sets[0]
    val = sets[1]
    val_inputs = val[:, :-1]
    val_labels = val[:, -1].astype('int32')
    test = sets[2]
    inputs = train[:, :-1]
    labels = train[:, -1]
    hard = HardParzen(10)
    hard.train(inputs, labels)
    # hard.compute_predictions(test)
    # print(hard.compute_predictions(test))
    soft = SoftRBFParzen(10)
    soft.train(inputs, labels)
    # soft.compute_predictions(test)
    # print(soft.compute_predictions(test))
    err = ErrorRate(inputs, labels, val_inputs, val_labels)
    # print(err.hard_parzen(1))
    # print(err.soft_parzen(10))
    print(get_test_errors(banknote))


test_predictions()
