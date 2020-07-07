from tensorflow import keras
from keras import Model
from keras import layers
from keras import optimizers
from keras import backend as K
import numpy as np


def _feature_tower(input_shape):
    input = layers.Input(shape=input_shape)
    cnn = layers.Conv2D(24, (7, 7), activation='relu')(input)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(64, (5, 5), activation='relu')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(96, (3, 3), activation='relu')(cnn)
    cnn = layers.Conv2D(96, (3, 3), activation='relu')(cnn)
    cnn = layers.Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Flatten()(cnn)
    cnn = layers.Dense(128, activation='relu')(cnn)
    return Model(input, cnn)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.relu(K.sum(K.square(x - y), axis=1)))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def _metric_network(input, is_siamese):
    input_a, input_b = input
    if is_siamese:
        base_network = _feature_tower(input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
    else:
        base_network_0 = _feature_tower(input_shape)
        base_network_1 = _feature_tower(input_shape)
        processed_a = base_network_0(input_a)
        processed_b = base_network_1(input_b)
    processed = layers.concatenate([processed_a, processed_b],axis=-1)
    distance = layers.Subtract()(processed)
    fc = layers.Dense(512, activation='relu')(distance)
    if dropout > 0.:
        fc = layers.Dropout(dropout)(fc)
    fc = layers.Dense(512, activation='relu')(fc)
    if dropout > 0.:
        fc = layers.Dropout(dropout)(fc)
    fc = layers.Dense(2, activation='relu')(fc)
    distance = layers.Lambda(euclidean_distance,
                             output_shape=eucl_dist_output_shape)(layers.concatenate([processed_a, processed_b],axis=-1))
    return distance, Model([input_a, input_b], fc)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


if __name__ == '__main__':
    input_shape = ((64, 64, 1))
    dropout = 0.0
    weight_decay = 0.001

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    con_out_left, left_net = _metric_network([input_a, input_b], is_siamese=True)
    con_out_right, right_net = _metric_network([input_a, input_b], is_siamese=False)

    output_left = left_net([input_a, input_b])
    en_out_left = layers.Dense(2, activation='softmax')(output_left)

    output_right = right_net([input_a, input_b])
    en_out_right = layers.Dense(2, activation='softmax')(output_right)

    fc = layers.Concatenate()(layers.concatenate([output_left, output_right],axis=-1))
    fc = layers.Dense(2, activation='relu')(fc)
    en_out = layers.Dense(2, activation='softmax')(fc)

    model = Model([input_a, input_b],
                  [con_out_left, con_out_right, en_out_left, en_out_right, en_out])

    model.summary()

    optimizer = optimizers.SGD(lr=0.001, momentum=0.95, decay=weight_decay)
    model.compile(optimizer=optimizer,
                  loss=[contrastive_loss, contrastive_loss, 'binary_crossentropy', 'binary_crossentropy',
                        'binary_crossentropy'],
                  loss_weights=[0.001, 0.001, 1, 1, 1],
                  metrics=['accuracy'])

    std = np.load('vedai/std_mean.npy')
    test = np.load('vedai/dataset_0_test.npz')
    train = np.load('vedai/dataset_0_train.npz')
    validation = np.load('vedai/dataset_0_validation.npz')

    left_mean = std[0]
    left_std = std[1]
    right_mean = std[2]
    right_std = std[3]

    test_left_data = test['left']
    test_right_data = test['right']
    test_labels = test['labels']

    train_left_data = train['left']
    train_right_data = train['right']
    train_labels = train['labels']

    validation_left_data = validation['left']
    validation_right_data = validation['right']
    validation_labels = validation['labels']

    test_left_data = (test_left_data.astype('float32') - left_mean) / left_std
    test_right_data = (test_right_data.astype('float32') - right_mean) / right_std

    validation_left_data = (validation_left_data.astype('float32') - left_mean) / left_std
    validation_right_data = (validation_right_data.astype('float32') - right_mean) / right_std

    train_left_data = (train_left_data.astype('float32') - left_mean) / left_std
    train_right_data = (train_right_data.astype('float32') - right_mean) / right_std

    train_left_data = train_left_data[..., np.newaxis]
    train_right_data = train_right_data[..., np.newaxis]

    validation_left_data = validation_left_data[..., np.newaxis]
    validation_right_data = validation_right_data[..., np.newaxis]

    test_left_data = test_left_data[..., np.newaxis]
    test_right_data = test_right_data[..., np.newaxis]

    num_train = train_left_data.shape[0]
    indices = np.random.permutation(num_train)
    train_left_data = train_left_data[indices]
    train_right_data = train_right_data[indices]
    train_labels = train_labels[indices]

    num_validation = validation_left_data.shape[0]
    indices = np.random.permutation(num_validation)
    validation_left_data = validation_left_data[indices]
    validation_right_data = validation_right_data[indices]
    validation_labels = validation_labels[indices]

    model.fit([train_left_data, train_right_data],
              [train_labels[:, 1], train_labels[:, 1], train_labels, train_labels, train_labels],
              batch_size=256,
              epochs=20,
              validation_data=([validation_left_data, validation_right_data],
                               [validation_labels[:, 1], validation_labels[:, 1], validation_labels, validation_labels,
                                validation_labels]))

    a = model.evaluate([test_left_data, test_right_data],
                       [test_labels[:, 1], test_labels[:, 1], test_labels, test_labels, test_labels])

    print(a)
