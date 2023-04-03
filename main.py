import tensorflow as tf
from tensorflow import keras
from keras import layers

SIZE = 256
SIZE_IMAGE = (SIZE,SIZE)

# criando um classificador para as doenças com base nas pastas de treino
TRAINING_DIR = "/content/sample_data/doenças/Training"
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=SIZE_IMAGE,
                                                       class_mode='categorical')

# criando um validador para verificação da precisão das doenças
VALIDATION_DIR = "/content/sample_data/doenças/Testing"
Validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

validation_generator = Validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size = SIZE_IMAGE,
                                                              class_mode='categorical')

#train_generator = train_generator/SIZE
#validation_generator = validation_generator/SIZE
# definindo a quantidade de filtros por camada
filters_layer_conv1 = 64
filters_layer_conv2 = 64
filters_layer_conv3 = 128
filters_layer_conv4 = 128
filters_layer_dense = 512
filters_layer_out = 2

# instanciando o modelo 
model = tf.keras.models.Sequential(# construindo as camadas convolucionais
                                   [layers.Conv2D(filters_layer_conv1, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    layers.Conv2D(filters_layer_conv2, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    layers.Conv2D(filters_layer_conv3, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    layers.Conv2D(filters_layer_conv4, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
                                    layers.MaxPooling2D(2, 2),
                                    #Definindo a porcentagens de individuos que serão eliminados após cada geração
                                    layers.Dropout(0.5),
                                    #compactação das camadas 
                                    layers.Flatten(),
                                    # construindo a camada densa de entrada
                                    layers.Dense(filters_layer_dense, activation='relu'),
                                    #Construindo a camada densa de saída com a quantidade de filtro correspondendo a quantidade de classes
                                    layers.Dense(filters_layer_out, activation="softmax")])
#imprime as configurações das camadas, os parametros de saída e a quantidade de parametros
model.summary()

#compila as camadas do modelo e faz a otimização
history = model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

#Divide os dados em um conjunto de treinamento e um conjunto de validação, e usa o conjunto de validação para medir o progresso durante o treino.
model.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=1,validation_steps = 3)
#salva os dados do 
model.save('Dados.keras')
