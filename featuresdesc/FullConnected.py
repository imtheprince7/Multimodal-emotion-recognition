



# not working
# Need to  discuss the ouput file of text and sequence-aggreagator.
# what should be taken for Global attention model.



































from keras.layers import Input, Concatenate, Dense
from keras.models import load_model,Model

#RNN model and the attention model
rnn_model = load_model('rnn_model.h5')
attention_model = load_model('attentionModel.h5')

rnn_outputs = rnn_model.outputs
attention_outputs = attention_model.outputs

# Concatenate the outputs along the feature dimension
concatenated_output = Concatenate(axis=-1)(rnn_outputs + attention_outputs)

# Fully connected layers
fc_output = Dense(units=128, activation='relu')(concatenated_output)
# Add more fully connected layers if needed
fc_output = Dense(units=128, activation='relu')(fc_output)

# Output layers
outputs = []
for _ in range(6):  # Assuming six outputs
    output = Dense(units=6, activation='softmax')(fc_output)
    outputs.append(output)

# Create the fully connected attention model
model = Model(inputs=[rnn_model.input, attention_model.input], outputs=outputs)
