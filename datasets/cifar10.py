import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


class CIFAR10():
    # read CIFAR-10 dataset python version according to the specifications in
    # https://www.cs.toronto.edu/~kriz/cifar.html

    _pickle_dir = 'pickle'
    _pickle_name = 'cifar10.pkl'

    _out_dir = 'tmp'

    def __init__(
        self,
        data_dir='./data/cifar-10-batches-py',
        from_pickle=True
    ):

        self.data_dir = data_dir

        if from_pickle:
            with open(os.path.join(CIFAR10._pickle_dir, CIFAR10._pickle_name), 'rb') as pickle_file:
                state = pickle.load(pickle_file)
            for key in state:
                setattr(self, key, state[key])

        else:
            meta_filename = os.path.join(self.data_dir, 'batches.meta')
            with open(meta_filename, 'rb') as f:
                self.name_of_label = pickle.load(f, encoding='bytes')[b'label_names']

            self.train_image = []
            self.train_label = []
            for i in range(1, 5+1):
                batch_filename = os.path.join(self.data_dir, f'data_batch_{i}')
                with open(batch_filename, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                    self.train_label.extend(data[b'labels'])
                    for image in data[b'data']:
                        r = image[1024*0:1024*1].reshape(32, 32)
                        g = image[1024*1:1024*2].reshape(32, 32)
                        b = image[1024*2:1024*3].reshape(32, 32)
                        self.train_image.append(np.stack((r, g, b), axis=2))

            self.test_image = []
            self.test_label = []
            with open(os.path.join(self.data_dir, 'test_batch'), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                self.test_label.extend(data[b'labels'])
                for image in data[b'data']:
                    r = image[1024*0:1024*1].reshape(32, 32)
                    g = image[1024*1:1024*2].reshape(32, 32)
                    b = image[1024*2:1024*3].reshape(32, 32)
                    self.test_image.append(np.stack((r, g, b), axis=2))


        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)

    def label_to_name(self, label):
        return self.name_of_label[label].decode('utf-8')

    def data(self):
        ''' return X_train, y_train, X_test, y_test.'''
        return self.train_image, self.train_label, self.test_image, self.test_label

    def pickle(self):
        if not os.path.isdir(CIFAR10._pickle_dir):
            os.makedirs(CIFAR10._pickle_dir)
        with open(os.path.join(CIFAR10._pickle_dir, CIFAR10._pickle_name), 'wb') as pickle_file:
            pickle.dump(self.__dict__, pickle_file)


    def view_samples(self, train_or_test, size=None, start_id=None):
        pickle_name = 'cifar10-view.png'
        save_path = os.path.join(CIFAR10._out_dir, pickle_name)

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

        if isinstance(size, int):
            for i in range(size):
                for j in range(size):
                    n = i*size + j
                    plt.subplot(size, size, n+1)
                    plt.imshow(images[start_id+n])
                    plt.xticks([])
                    plt.yticks([])
                    l = labels[start_id+n]
                    plt.title(f'{l} {self.label_to_name(l)}', fontsize='small', loc='left')
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(save_path)
            print(f'CIFAR-10 view generated to {CIFAR10._out_dir}/{pickle_name}.')
