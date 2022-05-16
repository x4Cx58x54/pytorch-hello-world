import os
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm import tqdm


class MNIST():
    # read MNIST dataset according to the specifications in
    # http://yann.lecun.com/exdb/mnist/

    _pickle_dir = 'pickle'
    _pickle_name = 'mnist.pkl'

    _out_dir = 'tmp'

    def __init__(
        self,
        data_path='./data/mnist',
        train_data_path='train-images-idx3-ubyte.gz',
        train_label_path='train-labels-idx1-ubyte.gz',
        test_data_path='t10k-images-idx3-ubyte.gz',
        test_label_path='t10k-labels-idx1-ubyte.gz',
        from_pickle=True,
    ):

        self.train_size = 60000
        self.test_size = 10000

        if from_pickle:
            with open(os.path.join(MNIST._pickle_dir, MNIST._pickle_name), 'rb') as pickle_file:
                state = pickle.load(pickle_file)
            for key in state:
                setattr(self, key, state[key])

        else:
            self.train_data_path = os.path.join(data_path, train_data_path)
            self.train_label_path = os.path.join(data_path, train_label_path)
            self.test_data_path = os.path.join(data_path, test_data_path)
            self.test_label_path = os.path.join(data_path, test_label_path)
            with gzip.open(self.train_data_path, 'rb') as train_data_file:
                self.train_image = self._read_idx3_ubyte(train_data_file.read())
            with gzip.open(self.train_label_path, 'rb') as train_label_file:
                self.train_label = self._read_idx1_ubyte(train_label_file.read())
            with gzip.open(self.test_data_path, 'rb') as test_data_file:
                self.test_image = self._read_idx3_ubyte(test_data_file.read())
            with gzip.open(self.test_label_path, 'rb') as test_label_file:
                self.test_label = self._read_idx1_ubyte(test_label_file.read())

        def flatten(x):
            return np.reshape(x, (len(x), np.size(x[0])))

        self.train_image_flattened = flatten(self.train_image)
        self.test_image_flattened = flatten(self.test_image)

        if not os.path.isdir(MNIST._out_dir):
            os.makedirs(MNIST._out_dir)


    def data(self, flattened=False):
        ''' return X_train, y_train, X_test, y_test.'''
        if flattened:
            return self.train_image_flattened, self.train_label, self.test_image_flattened, self.test_label
        else:
            return self.train_image, self.train_label, self.test_image, self.test_label

    def pickle(self):
        if not os.path.isdir(MNIST._pickle_dir):
            os.makedirs(MNIST._pickle_dir)
        with open(os.path.join(MNIST._pickle_dir, MNIST._pickle_name), 'wb') as pickle_file:
            pickle.dump(self.__dict__, pickle_file)

    def _read_ith_32bit_to_int(self, i, bytes_data):
        return int.from_bytes(bytes_data[i:i+4], byteorder='big')

    _magic_number_idx3_ubyte = 2051
    _magic_number_idx1_ubyte = 2049

    def _read_idx3_ubyte(self, bytes_data):
        ptr = 0
        magic_number = self._read_ith_32bit_to_int(ptr, bytes_data)
        assert magic_number == MNIST._magic_number_idx3_ubyte
        ptr += 4
        num_items = self._read_ith_32bit_to_int(ptr, bytes_data)
        ptr += 4
        num_rows = self._read_ith_32bit_to_int(ptr, bytes_data)
        ptr += 4
        num_cols = self._read_ith_32bit_to_int(ptr, bytes_data)
        ptr += 4

        res = np.zeros((num_items,num_rows,num_cols), dtype=np.uint8)
        # print((num_items,num_rows,num_cols))
        print(res.shape)

        for item in tqdm(range(num_items)):
            for row in range(num_rows):
                for col in range(num_cols):
                    res[item][row][col] = bytes_data[ptr]
                    ptr += 1

        return res

    def _read_idx1_ubyte(self, bytes_data):
        ptr = 0
        magic_number = self._read_ith_32bit_to_int(ptr, bytes_data)
        assert magic_number == MNIST._magic_number_idx1_ubyte
        ptr += 4
        num_items = self._read_ith_32bit_to_int(ptr, bytes_data)
        ptr += 4

        res = np.zeros((num_items,), dtype=np.uint8)
        print(res.shape)

        for item in tqdm(range(num_items)):
            res[item] = bytes_data[ptr]
            ptr += 1

        return res


    def view_samples(self, train_or_test, size=1, start_id=None):
        assert train_or_test in ('train', 'test')
        if train_or_test == 'train':
            if start_id is None:
                start_id = np.random.randint(self.train_size - size**2 - 1)
            images = self.train_image
            labels = self.train_label
        elif train_or_test == 'test':
            if start_id is None:
                start_id = np.random.randint(self.test_size - size**2 - 1)
            images = self.test_image
            labels = self.test_label

        for i in range(size):
            for j in range(size):
                n = i*size + j
                plt.subplot(size, size, n+1)
                plt.imshow(255-images[start_id+n], cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.title(f'{labels[start_id+n]}', fontsize='small', loc='left')
        plt.subplots_adjust(hspace=0.5)
        pickle_name = 'mnist_view.png'
        plt.savefig(os.path.join(MNIST._out_dir, pickle_name))
        print(f'MNIST view generated to {MNIST._out_dir}/{pickle_name}.')
