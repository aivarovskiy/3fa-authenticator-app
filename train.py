import itertools
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Lambda
from keras.optimizers import Adam
from tqdm import tqdm
import preprocess


pt_root = "drive/MyDrive"
pt_gen = pt_root + "/cedar_dataset/full_org/original_%d_%d.png"
pt_forg = pt_root + "/cedar_dataset/full_forg/forgeries_%d_%d.png"
pt_cpkt = pt_root + "/siamese/cpkt/"
pt_model = pt_root + "/siamese/model"
pt_weights = pt_root + "/siamese/siamese_weights"

n_signs = 12
choose_two = n_signs * (n_signs - 1) // 2

img_width = 160
img_height = 320


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(
        y_true * K.square(y_pred)
        + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
    )


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def compute_threshold(predictions, labels):
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    max_acc = 0
    best_thresh = -1

    for d in np.sort(predictions[:, 0]):
        idx1 = predictions[:, 0] < d
        idx2 = predictions[:, 0] >= d

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        acc = 0.5 * (tpr + tnr)

        if acc > max_acc:
            max_acc, best_thresh = acc, d

    return max_acc, best_thresh


def create_base_network(input_shape):
    seq = Sequential()

    seq.add(Input(input_shape))

    seq.add(Conv2D(32, 7, padding="same", activation="relu"))
    seq.add(MaxPooling2D(2))

    seq.add(Conv2D(32, 5, padding="same", activation="relu"))
    seq.add(MaxPooling2D(2))

    seq.add(Conv2D(64, 3, padding="same", activation="relu"))
    seq.add(MaxPooling2D(2))

    seq.add(Flatten())

    seq.add(Dense(256, activation="relu"))

    return seq


def create_siamese():
    input_shape = (img_width, img_height, 1)

    embedding = create_base_network(input_shape)

    anc_img = Input(input_shape)
    val_img = Input(input_shape)

    anc_embedding = embedding(anc_img)
    val_embedding = embedding(val_img)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [anc_embedding, val_embedding]
    )

    siamese = Model(inputs=[anc_img, val_img], outputs=distance)

    return siamese


def train_data_prep():
    persons_gen = []
    persons_forg = []

    for person in tqdm(range(1, 41)):
        s_gen = []
        s_forg = []
        for sign in range(1, n_signs + 1):
            s_gen.append(
                np.asarray(preprocess.path(pt_gen % (person, sign))).reshape(
                    img_width, img_height, 1
                )
            )
            s_forg.append(
                np.asarray(preprocess.path(pt_forg % (person, sign))).reshape(
                    img_width, img_height, 1
                )
            )
        persons_gen.append(s_gen)
        persons_forg.append(s_forg)

    pairs_gen = []
    pairs_forg = []

    for i in range(40):
        pairs_gen.extend(list(itertools.combinations(persons_gen[i], 2)))
        pairs_forg.extend(
            list(itertools.product(persons_gen[i], persons_forg[i]))[:choose_two]
        )

    x_train = np.asarray(pairs_gen + pairs_forg)
    y_train = np.concatenate(
        [np.ones([len(pairs_gen), 1]), np.zeros([len(pairs_forg), 1])]
    )

    del persons_gen, persons_forg, pairs_gen, pairs_forg

    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]

    return x_train, y_train


def test_data_prep():
    persons_gen = []
    persons_forg = []

    for person in tqdm(range(41, 51)):
        s_gen = []
        s_forg = []
        for sign in range(1, n_signs + 1):
            s_gen.append(
                np.asarray(preprocess.path(pt_gen % (person, sign))).reshape(
                    img_width, img_height, 1
                )
            )
            s_forg.append(
                np.asarray(preprocess.path(pt_forg % (person, sign))).reshape(
                    img_width, img_height, 1
                )
            )
        persons_gen.append(s_gen)
        persons_forg.append(s_forg)

    pairs_gen = []
    pairs_forg = []

    for i in range(10):
        pairs_gen.extend(list(itertools.combinations(persons_gen[i], 2)))
        pairs_forg.extend(list(itertools.product(persons_gen[i], persons_forg[i])))

    x_test = np.asarray(pairs_gen + pairs_forg)
    y_test = np.concatenate(
        [np.ones([len(pairs_gen), 1]), np.zeros([len(pairs_forg), 1])]
    )

    del persons_gen, persons_forg, pairs_gen, pairs_forg

    return x_test, y_test


def val_data_prep():
    persons_gen = []
    persons_forg = []

    for person in tqdm(range(51, 56)):
        s_gen = []
        s_forg = []
        for sign in range(1, n_signs + 1):
            s_gen.append(
                np.asarray(preprocess.path(pt_gen % (person, sign))).reshape(
                    img_width, img_height, 1
                )
            )
            s_forg.append(
                np.asarray(preprocess.path(pt_forg % (person, sign))).reshape(
                    img_width, img_height, 1
                )
            )
        persons_gen.append(s_gen)
        persons_forg.append(s_forg)

    pairs_gen = []
    pairs_forg = []

    for i in range(5):
        pairs_gen.extend(list(itertools.combinations(persons_gen[i], 2)))
        pairs_forg.extend(list(itertools.product(persons_gen[i], persons_forg[i])))

    x_val = np.asarray(pairs_gen + pairs_forg)
    y_val = np.concatenate(
        [np.ones([len(pairs_gen), 1]), np.zeros([len(pairs_forg), 1])]
    )

    del persons_gen, persons_forg, pairs_gen, pairs_forg

    return x_val, y_val


def training():
    # snn creation
    siamese = create_siamese()
    siamese.compile(
        optimizer=Adam(learning_rate=5e-5), loss=contrastive_loss, metrics=[accuracy]
    )

    # dataset creation
    x_train, y_train = test_data_prep()
    x_test, y_test = test_data_prep()
    x_val, y_val = val_data_prep()

    # training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=pt_cpkt,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    siamese.fit(
        [x_train[:, 0], x_train[:, 1]],
        y_train,
        epochs=10,
        batch_size=32,
        shuffle=True,
        validation_data=([x_val[:, 0], x_val[:, 1]], y_val),
        callbacks=[checkpoint],
    )
    siamese.save(pt_model)
    siamese.save_weights(pt_weights)

    # testing
    siamese.evaluate([x_test[:, 0], x_test[:, 1]], y_test)

    # computing threshold
    y_pred = siamese.predict([x_test[:, 0], x_test[:, 1]])

    return compute_threshold(y_pred, y_test)
