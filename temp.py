import pandas as pd
df=pd.read_csv('AggragatedData.csv')
import keras

# Create the Conv1D model.
model = keras.Sequential([
    keras.layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the data.
model.fit(df['features'], df['labels'], epochs=10)

# Create the self-attention model.
attention_model = keras.Sequential([
    keras.layers.Attention(attention_mechanism='dot', name='attention'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the attention model.
attention_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the attention model to the data.
attention_model.fit(df['features'], df['labels'], epochs=10)



