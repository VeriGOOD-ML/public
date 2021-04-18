# import tensorflow as tf
import numpy as np
from collections import namedtuple
NpType = namedtuple('NpType', ['max', 'min', 'dtype', 'bits'])
from time import time
PREC_TO_DTYPE = {
    'int16': np.int16,
    'int32': np.int32,
    'float16': np.float16,
    'float32': np.float32,
}


class NpSvmRef(object):
    def __init__(self, n_features=139, n_locations=325, weight_init=np.random.randn,
                 epochs=1000, lr=0.001, delta=1, model_file=None, train_size=7703):
        self.delta = delta
        self.n_features = n_features
        self.n_locations = n_locations
        self.lr = lr
        self.train_size = train_size
        if model_file:
            self.weights = self.load_model(model_file)
        else:
            self.weights = weight_init(n_features, n_locations)
        self.epochs = epochs
        self.x_train = None
        self.x_test = None
        self.x_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None

    def is_data_loaded(self):
        return self.x_train is not None

    def load_validation_data(self):
        from benchmarks.svm_darpa_benchmarks.data_loader import get_validation_data
        self.x_val, self.y_val = get_validation_data()

    def load_training_data(self, use_dummy_data):
        if use_dummy_data:
            from benchmarks.svm_darpa_benchmarks.data_loader import generate_data

            x_train, x_test, y_train, y_test = generate_data()
            x_train = x_train.to_numpy()
            x_test = x_test.to_numpy()
            y_train = y_train.to_numpy().astype(int)
            y_test = y_test.to_numpy().astype(int)
        else:
            x_train = np.random.rand(self.train_size, self.n_features)
            y_train = np.random.randint(3, size=self.train_size)

            x_test = np.random.rand(self.train_size, self.n_features)
            y_test = np.random.randint(3, size=self.train_size)


        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            weights = np.load(f)
        return weights

    def store_model(self, precision):
        if precision != np.float32:
            name = "svm_np_int16.npy"
        else:
            name = 'svm_np.npy'
        with open(name, 'wb') as f:
            np.save(f, self.weights)

    def predict(self, x):
        return np.dot(x, self.weights)

    def train_batch(self, x_train_batch, y_train_batch):
        start = time()
        num_train = x_train_batch.shape[0]
        scores = np.dot(x_train_batch, self.weights)
        # scores = self.predict(x_train_batch)

        correct_class_score = scores[np.arange(num_train), y_train_batch].reshape(num_train, 1)
        margin = np.maximum(0, scores - correct_class_score + self.delta)
        margin[np.arange(num_train), y_train_batch] = 0
        margin[margin > 1] = 1
        valid_margin_count = margin.sum(axis=1)
        margin[np.arange(num_train), y_train_batch] -= valid_margin_count
        dW = (x_train_batch.T).dot(margin)
        elapsed = time() - start
        print(f"Elapsed time: {elapsed*1000} ms")
        return dW

    def get_quantized(self, value, precision):
        x_min = np.min(value)
        x_max = np.max(value)
        scale = (x_max-x_min) / (precision.max - precision.min)
        # x_zp = precision.max - x_max / (scale)
        # x_quant = value/scale + x_zp
        x_quant = value/(scale)
        return x_quant, scale

    def lower_precision_prediction(self, x, input_prec, inter_prec):
        dtype_str = f"{inter_prec.dtype}{inter_prec.bits}"
        if dtype_str in PREC_TO_DTYPE:
            dtype = PREC_TO_DTYPE[dtype_str]
        else:
            dtype = np.float64 if input_prec.dtype == "float" else np.int64

        if input_prec.dtype == 'float':
            low_prec_weights, _ = self.get_quantized(self.weights, input_prec)
            low_prec_weights = self.weights.astype(dtype)
            low_prec_x, _ = self.get_quantized(x, input_prec)
            low_prec_x = x.astype(dtype)
        else:
            assert input_prec.dtype == 'int'
            low_prec_weights, _ = self.get_quantized(self.weights, input_prec)
            low_prec_weights = low_prec_weights.astype(dtype)
            low_prec_x, _ = self.get_quantized(x, input_prec)
            low_prec_x = low_prec_x.astype(dtype)



        out = np.dot(low_prec_x, low_prec_weights)
        return self.clip_precision(out, inter_prec)

    def clip_precision(self, val, precision):
        return np.clip(val, precision.min, precision.max)

    def lower_precision_train_batch(self, x_train_batch, y_train_batch, precision):

        num_train = x_train_batch.shape[0]
        scores = self.lower_precision_prediction(x_train_batch, precision)

        correct_class_score = scores[np.arange(num_train), y_train_batch].reshape(num_train, 1)
        margin = self.clip_precision(np.maximum(0, scores - correct_class_score + self.delta), precision)
        margin[np.arange(num_train), y_train_batch] = 0
        margin[margin > 1] = 1
        valid_margin_count = self.clip_precision(margin.sum(axis=1), precision)
        margin[np.arange(num_train), y_train_batch] -= valid_margin_count
        dW = (x_train_batch.T).dot(self.clip_precision(margin, precision))
        return self.clip_precision(dW, precision)



    def get_accuracy(self, dataset="test", input_prec=None, inter_prec=None):
        if dataset == "test" and not self.is_data_loaded():
            self.load_training_data()
        elif dataset != "test" and not self.x_val:
            self.load_validation_data()
        if input_prec != None:
            output = self.lower_precision_prediction(self.x_test, input_prec, inter_prec)
        else:
            output = self.predict(self.x_test)
        output = np.argmax(output, axis=1)
        return np.mean(self.y_test == output)

    def train(self, get_final_accuracy=True, store_model=True, precision=None, use_dummy_data=False):
        if not self.is_data_loaded():
            self.load_training_data(use_dummy_data)
        print(f"Initial accuracy is: {self.get_accuracy() * 100}")

        if precision != None:
            for _ in range(self.epochs):
                dW = self.lower_precision_train_batch(self.x_train, self.y_train, precision)
                self.weights -= self.clip_precision(self.lr * dW, precision)
                self.weights = self.clip_precision(self.weights, precision)
        else:
            for _ in range(self.epochs):
                dW = self.train_batch(self.x_train, self.y_train)
                self.weights -= self.lr * dW

        if get_final_accuracy:
            print(f"FInal accuracy is: {self.get_accuracy()*100}")

        if store_model:
            self.store_model(precision)



if __name__ == "__main__":
    model = NpSvmRef(epochs=1)
    model.train(get_final_accuracy=True, store_model=False)

    print(f"Regular accuracy: {100*model.get_accuracy()}")
