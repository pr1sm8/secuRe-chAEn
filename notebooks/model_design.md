# Recurrent Autoencoder for detection of fruadulant transactions in blockchain

The below snippet is the model defintion of the autoencoder model used in the project

``` python
model = Sequential()
model.add(Input(shape=(a.shape[1], a.shape[2])))
model.add(CuDNNLSTM(64, return_sequences=True  ))
model.add(CuDNNLSTM(SEQUENCE_LENGTH, return_sequences=False ))
model.add(Dense(SEQUENCE_LENGTH))
model.add(RepeatVector(SEQUENCE_LENGTH))
model.add(CuDNNLSTM(SEQUENCE_LENGTH, return_sequences=True ))
model.add(CuDNNLSTM(64, return_sequences=True  ))
model.add(TimeDistributed(Dense(a.shape[2])))
```

The model is a Tensorflow Sequential Layer by layer model

After compilation of the model where a sequence length of 20 was taken

``` text
model.add(Input(shape=(a.shape[1], a.shape[2])))
```

LSTM input shape requires `A 3D tensor with shape [batch, timesteps, feature].`

this layer instantiates the tensor with the shape specified
The shape here is

*a.shape[1] = the number of vector sequences*

_a.shape[2] = the number of transactions on one such vector sequence_

``` text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 cu_dnnlstm (CuDNNLSTM)      (None, 20, 64)            19200     
                                                                 
 cu_dnnlstm_1 (CuDNNLSTM)    (None, 20)                6880      
                                                                 
 dense (Dense)               (None, 20)                420       
                                                                 
 repeat_vector (RepeatVector  (None, 20, 20)           0         
 )                                                               
                                                                 
 cu_dnnlstm_2 (CuDNNLSTM)    (None, 20, 20)            3360      
                                                                 
 cu_dnnlstm_3 (CuDNNLSTM)    (None, 20, 64)            22016     
                                                                 
 time_distributed (TimeDistr  (None, 20, 9)            585       
 ibuted)                                                         
                                                                 
=================================================================
Total params: 52,461
Trainable params: 52,461
Non-trainable params: 0
_________________________________________________________________
```

``` python
model.add(CuDNNLSTM(64, return_sequences=True  ))
model.add(CuDNNLSTM(SEQUENCE_LENGTH, return_sequences=False ))
```

The first layer here is containing 64 LSTM cells and the output from this layer also contains the outputs from these 64 LSTM cells for the next layer along with the SEQUENCE_LENGTH number of transactions

The next layer contains SEQUENCE_LENGTH number cells to apply another layer of LSTM on the given transactions

``` python
model.add(Dense(SEQUENCE_LENGTH))
```

This layer enables the model to perform a linear transformation over the SEQUENCE_LENGTH number of transactions

``` python
model.add(RepeatVector(SEQUENCE_LENGTH))
```

This layer just repeats the tensor recieved, to provide the separation of the encoder side to the decoder side (necessary while the derivation for the optimization of the weights)

``` python
model.add(CuDNNLSTM(SEQUENCE_LENGTH, return_sequences=True ))
model.add(CuDNNLSTM(64, return_sequences=True  ))
```

these layers are just a mirror flip of the encoder

``` python
model.add(TimeDistributed(Dense(a.shape[2])))
```

This wrapper allows to apply a layer to every temporal slice of an input, hence recreating the input features back as an output

``` python
history = model.fit(a, a, epochs=3, batch_size=1, validation_split=0.05, ).history
```

We then train the model to reconstruct the input to the output
While the bottleneck is a tensor of shape SEQUENCE_LENGTH
