from keras.layers import Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model
from sequenceAgg import audio_features

# Define the input shape for the audio features
audio_input_shape = (audio_feature_length, 1)

# Define the input shape for the text features
text_input_shape = (text_feature_length, 1)

# Create the input layers for the audio and text features
audio_input = Input(shape=audio_input_shape, name='audio_input')
text_input = Input(shape=text_input_shape, name='text_input')

# Apply a 1D convolution to the audio input
conv_audio = Conv1D(filters=32, kernel_size=3, activation='relu')(audio_input)
pool_audio = MaxPooling1D(pool_size=2)(conv_audio)
conv_audio = Conv1D(filters=64, kernel_size=3, activation='relu')(pool_audio)
pool_audio = MaxPooling1D(pool_size=2)(conv_audio)
flatten_audio = Flatten()(pool_audio)

# Apply a 1D convolution to the text input
conv_text = Conv1D(filters=32, kernel_size=3, activation='relu')(text_input)
pool_text = MaxPooling1D(pool_size=2)(conv_text)
conv_text = Conv1D(filters=64, kernel_size=3, activation='relu')(pool_text)
pool_text = MaxPooling1D(pool_size=2)(conv_text)
flatten_text = Flatten()(pool_text)

# Concatenate the flattened audio and text features
concatenated = Concatenate()([flatten_audio, flatten_text])

# Add a fully connected layer
fc = Dense(64, activation='relu')(concatenated)

# Output layer
output = Dense(num_classes, activation='softmax')(fc)

# Create the model
model = Model(inputs=[audio_input, text_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()







#====================================================================================#
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Define the input shape for the audio features
audio_input_shape = (audio_features.shape[1], 1)

# Define the input shape for the text features
text_input_shape = (text_features.shape[1], 1)

# Define the number of filters for the convolution layers
num_filters = 64

# Define the kernel size for the convolution layers
kernel_size = 3

# Define the pool size for the max pooling layers
pool_size = 2

# Define the number of neurons in the fully connected layer
num_neurons = 128

# Define the output dimension of the fully connected layer
output_dim = 1

# Define the model architecture
model = Sequential()
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=audio_input_shape))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=text_input_shape))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(num_neurons, activation='relu'))
model.add(Dense(output_dim, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([audio_features, text_features], labels, epochs=10, batch_size=32, validation_split=0.2)




