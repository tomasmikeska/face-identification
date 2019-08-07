from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, PReLU


def VGG16(embedding_size, input_tensor):
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1_1')(input_tensor)
    x = PReLU()(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv1_2')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2_1')(x)
    x = PReLU()(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2_2')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='conv3_1')(x)
    x = PReLU()(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_2')(x)
    x = PReLU()(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='conv4_1')(x)
    x = PReLU()(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_2')(x)
    x = PReLU()(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_3')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='conv5_1')(x)
    x = PReLU()(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv5_2')(x)
    x = PReLU()(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv5_3')(x)
    x = PReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = Flatten()(x)
    x = Dense(4096, name='fc6')(x)
    x = PReLU()(x)
    x = Dense(embedding_size, name='embeddings')(x)

    return x
