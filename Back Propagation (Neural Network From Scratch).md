#Deep Learning : Back Propagation (Neural Network From Scratch)

**Maxime Bourgeois**





## Import some libraries


```python
from math import exp
from random import seed
from random import random
import pandas as pd 
from google.colab import drive
import numpy as np
from random import randint
from random import randrange

```

Useful video to understant what happen in this algorithm

https://www.youtube.com/watch?v=0jh-jlWfKwo&ab_channel=HugoLarochelle

## Exe. 1 

Write a function namely initialize_network(n_inputs, n_hidden,n_outputs)

this function initialize the network 

Our network is a table of layers.
Each layer contains weights.

n_input + 1 and n_hidden + 1 is to consider the bias.


```python
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
```

## Exe. 2

Test of the function.

Parameters : 
*   n_input = number of class
*   n_hidden = number of hidden layer
*   n_output = number of output class

What do we do in fact :

https://matthewmazur.files.wordpress.com/2018/03/neural_network-7.png?w=584


```python
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
	print(layer)
```

    [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
    [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
    

## Exe. 3 : Write a function namely activate(weights, inputs)

now we need to calculate activation for a neron.
This is require for the forward propagation.

activation = sum(weight * input) + bias




```python
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

length = 5
weights = np.zeros(length+1,float)
inputs = np.zeros(length,float)
for i in range(length):
  weights[i] = random()
  inputs[i] = random()
print(activate(weights, inputs))
```

    0.5936099054122081
    

## Exe. 4 : Write a function transfer(activation)

Tranfert function is the sigmoïd function apply to the activation.
this function is named logistic function.

sigmoid is a S shape and allows to give value between 0 and 1.
furthermore, sigmoïd can be easily derivative.


```python
# Transfer neuron activation
def transfer(activation):
    return 1 / (1 + np.exp(-activation))

print(transfer(activate(weights, inputs)))
```

    0.6441929962815797
    

## Exe. 5: Write a function name forward_propagate(network, row)

by using activate and transfert function, this function allows to do a forward propagation.

this one browse all neuron (double loop) and compute transfert of activation. then it add activated value on each neuron in the network.


```python
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

```

## Exe. 6 (Test forward Propagation)


```python
network = initialize_network(5,5,5)
new_inputs = forward_propagate(network, [randint(0, 1) for i in range(5)])
print(new_inputs)
print(network)
```

    [0.9380351112355052, 0.9045895257000935, 0.8898354315671796, 0.9022145799643534, 0.8794297528843628]
    [[{'weights': [0.9452706955539223, 0.9014274576114836, 0.030589983033553536, 0.0254458609934608, 0.5414124727934966, 0.9391491627785106], 'output': 0.7189277602280576}, {'weights': [0.38120423768821243, 0.21659939713061338, 0.4221165755827173, 0.029040787574867943, 0.22169166627303505, 0.43788759365057206], 'output': 0.6077555716516745}, {'weights': [0.49581224138185065, 0.23308445025757263, 0.2308665415409843, 0.2187810373376886, 0.4596034657377336, 0.28978161459048557], 'output': 0.5719426677270808}, {'weights': [0.021489705265908876, 0.8375779756625729, 0.5564543226524334, 0.6422943629324456, 0.1859062658947177, 0.9925434121760651], 'output': 0.7295900010621472}, {'weights': [0.8599465287952899, 0.12088995980580641, 0.3326951853601291, 0.7214844075832684, 0.7111917696952796, 0.9364405867994596], 'output': 0.718380112146954}], [{'weights': [0.4221069999614152, 0.830035693274327, 0.670305566414071, 0.3033685109329176, 0.5875806061435594, 0.8824790008318577], 'output': 0.9380351112355052}, {'weights': [0.8461974184283128, 0.5052838205796004, 0.5890022579825517, 0.034525830151341586, 0.24273997354306764, 0.7974042475543028], 'output': 0.9045895257000935}, {'weights': [0.4143139993007743, 0.17300740157905092, 0.548798761388153, 0.7030407620656315, 0.6744858305023272, 0.3747030205016403], 'output': 0.8898354315671796}, {'weights': [0.4389616300445631, 0.5084264882499818, 0.7784426150001458, 0.5209384176131452, 0.39325509496422606, 0.4896935204622582], 'output': 0.9022145799643534}, {'weights': [0.029574963966907064, 0.04348729035652743, 0.703382088603836, 0.9831877173096739, 0.5931837303800576, 0.393599686377914], 'output': 0.8794297528843628}]]
    

## Exe. 7 : Write the output of your neural network transfer_derivative(output)

Useful function in order to do backward propagation


```python
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
```

## Exe. 8 : write a function named backward_propagate_error()

error is calculeted between expected output and real network output value.

formula of delta:

if it's output layer:

* delta = (expected - output) * transfer_derivative(output)

if it's hidden layer: 

* error = (weight * error) * transfer_derivative(output)




```python
def backward_propagate_error(network, expected):
  # reverse allows to inverse elements in a list.
  # easier for backward propagation.
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  
    #if not input layer
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
```

## Exe. 9 Test the backward_propagate_error()


```python
backward_propagate_error(network, [0,1,2,3,4])
print(network)
```

    [[{'weights': [0.9452706955539223, 0.9014274576114836, 0.030589983033553536, 0.0254458609934608, 0.5414124727934966, 0.9391491627785106], 'output': 0.7189277602280576, 'delta': 0.024262283871714873}, {'weights': [0.38120423768821243, 0.21659939713061338, 0.4221165755827173, 0.029040787574867943, 0.22169166627303505, 0.43788759365057206], 'output': 0.6077555716516745, 'delta': 0.020553400698212338}, {'weights': [0.49581224138185065, 0.23308445025757263, 0.2308665415409843, 0.2187810373376886, 0.4596034657377336, 0.28978161459048557], 'output': 0.5719426677270808, 'delta': 0.099113231700797}, {'weights': [0.021489705265908876, 0.8375779756625729, 0.5564543226524334, 0.6422943629324456, 0.1859062658947177, 0.9925434121760651], 'output': 0.7295900010621472, 'delta': 0.0950903796050634}, {'weights': [0.8599465287952899, 0.12088995980580641, 0.3326951853601291, 0.7214844075832684, 0.7111917696952796, 0.9364405867994596], 'output': 0.718380112146954, 'delta': 0.06320584739466117}], [{'weights': [0.4221069999614152, 0.830035693274327, 0.670305566414071, 0.3033685109329176, 0.5875806061435594, 0.8824790008318577], 'output': 0.9380351112355052, 'delta': -0.05452351721179183}, {'weights': [0.8461974184283128, 0.5052838205796004, 0.5890022579825517, 0.034525830151341586, 0.24273997354306764, 0.7974042475543028], 'output': 0.9045895257000935, 'delta': 0.008234621925894674}, {'weights': [0.4143139993007743, 0.17300740157905092, 0.548798761388153, 0.7030407620656315, 0.6744858305023272, 0.3747030205016403], 'output': 0.8898354315671796, 'delta': 0.10882758565693827}, {'weights': [0.4389616300445631, 0.5084264882499818, 0.7784426150001458, 0.5209384176131452, 0.39325509496422606, 0.4896935204622582], 'output': 0.9022145799643534, 'delta': 0.1850738286504576}, {'weights': [0.029574963966907064, 0.04348729035652743, 0.703382088603836, 0.9831877173096739, 0.5931837303800576, 0.393599686377914], 'output': 0.8794297528843628, 'delta': 0.3308836204415923}]]
    

## Exe. 10 : Write a function named update_weights(network, row, l rate)

now we have our Delta, so we can update weight.

we need to define an other parameter which is the learning rate.

We will browse the network and compute new weight thanks to this formula:

weight = weight + learning_rate * error * input




```python
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
```

## Exe. 11 : Write a function named train_network(network, train, l rate, n_epoch, n_outputs)

Train network function gathers all function that we code before.

There are some parameters to define:

l_rate = learning rate
n_epoch = number of learn iteration (how much forward+backward)
n_output = number of output classes
network = an initialized network
train = the input dataset used for training

foreach epoch this algorithm browse all input row and do:

* a forward propagation
* compute the the sum of squares used for measuring the variation or deviation from the mean
* backward propagation 
* update weights




```python
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
```

## Exe. 12 : Initialize network, train on random dataset

Create the random DataSet


```python
data_set_rand = []
for i in range(10):
  sub_set = []
  for j in range(2):
    sub_set.append(random())
  sub_set.append(randrange(0,2,1))
  data_set_rand.append(sub_set)
  
data_set_rand
```




    [[0.5396174484497788, 0.8602897789205496, 0],
     [0.4044548683894549, 0.34382589125981466, 1],
     [0.45913173191066836, 0.2692794774414212, 0],
     [0.3836896328900399, 0.8569491268730604, 0],
     [0.518678283523002, 0.561357864778379, 1],
     [0.9497192655214912, 0.4811018174142402, 1],
     [0.5699993338763802, 0.19983942017714307, 1],
     [0.48492511222773416, 0.3567899645449557, 1],
     [0.0015847499555259326, 0.5401095570883858, 1],
     [0.4581468000997244, 0.027974984083842358, 0]]



Initialize variables & network


```python
learning_rate = 0.5
n_epochs = 20
n_inputs = len(data_set_rand[0]) - 1
n_outputs = len(set([row[-1] for row in data_set_rand]))
n_hidden_layer = 2

network = initialize_network(n_inputs, n_hidden_layer, n_outputs)
 
```

Train the model


```python
train_network(network, data_set_rand, learning_rate, n_epochs, n_outputs)
for layer in network:
	print(layer)
```

    >epoch=0, lrate=0.500, error=6.644
    >epoch=1, lrate=0.500, error=5.796
    >epoch=2, lrate=0.500, error=5.304
    >epoch=3, lrate=0.500, error=5.135
    >epoch=4, lrate=0.500, error=5.089
    >epoch=5, lrate=0.500, error=5.077
    >epoch=6, lrate=0.500, error=5.074
    >epoch=7, lrate=0.500, error=5.073
    >epoch=8, lrate=0.500, error=5.073
    >epoch=9, lrate=0.500, error=5.073
    >epoch=10, lrate=0.500, error=5.072
    >epoch=11, lrate=0.500, error=5.072
    >epoch=12, lrate=0.500, error=5.072
    >epoch=13, lrate=0.500, error=5.071
    >epoch=14, lrate=0.500, error=5.071
    >epoch=15, lrate=0.500, error=5.070
    >epoch=16, lrate=0.500, error=5.070
    >epoch=17, lrate=0.500, error=5.069
    >epoch=18, lrate=0.500, error=5.069
    >epoch=19, lrate=0.500, error=5.068
    [{'weights': [0.4756514740564284, 0.5052564507413251, -0.0616792513770253], 'output': 0.5408288856895984, 'delta': 0.011057400842033532}, {'weights': [0.11573704489102252, 0.48839917700401264, 0.8179614715774325], 'output': 0.7131253653430455, 'delta': -0.04288980525442849}]
    [{'weights': [0.24676127933151368, -0.7212518608874439, -0.1275954499038007], 'output': 0.34484529709549483, 'delta': 0.1480171484650492}, {'weights': [-0.13411510966076756, 0.5896453247433593, 0.15816711233103342], 'output': 0.654619250527217, 'delta': -0.1480047564773284}]
    

## Exe. 13 : Write a Predict function

By doing a forward propagation we can predict output value with a vector of input values.

return the corresponding output class (index) for the higher output probability.




```python
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
```

## Exe. 14 : Test Predict() function

we create a predict function that perform a prediction over all the dataset (test).

then, compute accuracy. here is 60 % accuracy.


```python
def predict_on_DataSet(network, dataset):
  acc = 0
  good = 0
  total = 0

  for row in dataset:
    prediction = predict(network, row)
    total = total + 1
    print('Expected=%d, Got=%d' % (row[-1], prediction))

    if row[-1] == prediction: 
      good += 1


  acc = (good/total)*100
  print("accuracy = ", acc, "%")


predict_on_DataSet(network, data_set_rand)
```

    Expected=0, Got=1
    Expected=1, Got=1
    Expected=0, Got=1
    Expected=0, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=0, Got=1
    accuracy =  60.0 %
    

## Exe. 16: Write a function load csv(filename)

this function allows to get the dataset by using read csv


```python
def load_csv(filename):
  data = pd.read_csv(filename, sep='\s+|"', header=None, usecols=range(1, 9))
  return data

```

## Exe. 17 : Suffle Normalize Split & Train 

Shuffle the dataset, normalize all columns except the labels (final
column) w.r.t maximum and minimum value of each column. Take 80% and 20% to train and test set respectively.

Load csv:


```python
file = '/content/sample_data/seeds_dataset.csv'

data_csv = load_csv(file)
data_csv

```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:2: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.26</td>
      <td>14.84</td>
      <td>0.8710</td>
      <td>5.763</td>
      <td>3.312</td>
      <td>2.221</td>
      <td>5.220</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.88</td>
      <td>14.57</td>
      <td>0.8811</td>
      <td>5.554</td>
      <td>3.333</td>
      <td>1.018</td>
      <td>4.956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.29</td>
      <td>14.09</td>
      <td>0.9050</td>
      <td>5.291</td>
      <td>3.337</td>
      <td>2.699</td>
      <td>4.825</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.84</td>
      <td>13.94</td>
      <td>0.8955</td>
      <td>5.324</td>
      <td>3.379</td>
      <td>2.259</td>
      <td>4.805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.14</td>
      <td>14.99</td>
      <td>0.9034</td>
      <td>5.658</td>
      <td>3.562</td>
      <td>1.355</td>
      <td>5.175</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>12.19</td>
      <td>13.20</td>
      <td>0.8783</td>
      <td>5.137</td>
      <td>2.981</td>
      <td>3.631</td>
      <td>4.870</td>
      <td>3</td>
    </tr>
    <tr>
      <th>206</th>
      <td>11.23</td>
      <td>12.88</td>
      <td>0.8511</td>
      <td>5.140</td>
      <td>2.795</td>
      <td>4.325</td>
      <td>5.003</td>
      <td>3</td>
    </tr>
    <tr>
      <th>207</th>
      <td>13.20</td>
      <td>13.66</td>
      <td>0.8883</td>
      <td>5.236</td>
      <td>3.232</td>
      <td>8.315</td>
      <td>5.056</td>
      <td>3</td>
    </tr>
    <tr>
      <th>208</th>
      <td>11.84</td>
      <td>13.21</td>
      <td>0.8521</td>
      <td>5.175</td>
      <td>2.836</td>
      <td>3.598</td>
      <td>5.044</td>
      <td>3</td>
    </tr>
    <tr>
      <th>209</th>
      <td>12.30</td>
      <td>13.34</td>
      <td>0.8684</td>
      <td>5.243</td>
      <td>2.974</td>
      <td>5.637</td>
      <td>5.063</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 8 columns</p>
</div>



Normalize data

this function allows to compute normalization on the data by using the formula:

z = (x-x_min)/(x_max-x_min)

* x_min = minimum value of a column
* x_max = maximum value of a column
* x = value to normalize




```python
def normalize(data_csv):
  for i in range(1, len(data_csv.columns)):
    x_min = min(data_csv[i])
    x_max = max(data_csv[i])
    data_csv[i] = data_csv[i].apply(lambda x: (x - x_min)/(x_max-x_min))
  return data_csv

data_csv = normalize(data_csv)
data_csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.440982</td>
      <td>0.502066</td>
      <td>0.570780</td>
      <td>0.486486</td>
      <td>0.486101</td>
      <td>0.189302</td>
      <td>0.345150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.405099</td>
      <td>0.446281</td>
      <td>0.662432</td>
      <td>0.368806</td>
      <td>0.501069</td>
      <td>0.032883</td>
      <td>0.215165</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.349386</td>
      <td>0.347107</td>
      <td>0.879310</td>
      <td>0.220721</td>
      <td>0.503920</td>
      <td>0.251453</td>
      <td>0.150665</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.306893</td>
      <td>0.316116</td>
      <td>0.793103</td>
      <td>0.239302</td>
      <td>0.533856</td>
      <td>0.194243</td>
      <td>0.140817</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.524079</td>
      <td>0.533058</td>
      <td>0.864791</td>
      <td>0.427365</td>
      <td>0.664291</td>
      <td>0.076701</td>
      <td>0.322994</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>0.151086</td>
      <td>0.163223</td>
      <td>0.637024</td>
      <td>0.134009</td>
      <td>0.250178</td>
      <td>0.372635</td>
      <td>0.172821</td>
      <td>3</td>
    </tr>
    <tr>
      <th>206</th>
      <td>0.060434</td>
      <td>0.097107</td>
      <td>0.390200</td>
      <td>0.135698</td>
      <td>0.117605</td>
      <td>0.462872</td>
      <td>0.238306</td>
      <td>3</td>
    </tr>
    <tr>
      <th>207</th>
      <td>0.246459</td>
      <td>0.258264</td>
      <td>0.727768</td>
      <td>0.189752</td>
      <td>0.429081</td>
      <td>0.981667</td>
      <td>0.264402</td>
      <td>3</td>
    </tr>
    <tr>
      <th>208</th>
      <td>0.118036</td>
      <td>0.165289</td>
      <td>0.399274</td>
      <td>0.155405</td>
      <td>0.146828</td>
      <td>0.368344</td>
      <td>0.258493</td>
      <td>3</td>
    </tr>
    <tr>
      <th>209</th>
      <td>0.161473</td>
      <td>0.192149</td>
      <td>0.547187</td>
      <td>0.193694</td>
      <td>0.245189</td>
      <td>0.633463</td>
      <td>0.267848</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 8 columns</p>
</div>



Shuffle

sample function allow to shuffle.


```python
data_csv = data_csv.sample(frac=1)
data_csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>154</th>
      <td>0.072710</td>
      <td>0.132231</td>
      <td>0.273140</td>
      <td>0.155405</td>
      <td>0.089095</td>
      <td>0.426855</td>
      <td>0.366322</td>
      <td>3</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.583569</td>
      <td>0.663223</td>
      <td>0.505445</td>
      <td>0.578829</td>
      <td>0.575909</td>
      <td>0.540236</td>
      <td>0.628262</td>
      <td>2</td>
    </tr>
    <tr>
      <th>157</th>
      <td>0.145420</td>
      <td>0.272727</td>
      <td>0.000000</td>
      <td>0.278716</td>
      <td>0.081967</td>
      <td>0.527884</td>
      <td>0.345150</td>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.332389</td>
      <td>0.365702</td>
      <td>0.670599</td>
      <td>0.361486</td>
      <td>0.421240</td>
      <td>0.258604</td>
      <td>0.255539</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.078376</td>
      <td>0.092975</td>
      <td>0.546279</td>
      <td>0.061374</td>
      <td>0.156807</td>
      <td>0.251583</td>
      <td>0.043328</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>118</th>
      <td>0.783758</td>
      <td>0.789256</td>
      <td>0.841198</td>
      <td>0.747748</td>
      <td>0.811832</td>
      <td>0.373675</td>
      <td>0.712457</td>
      <td>2</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0.196412</td>
      <td>0.188017</td>
      <td>0.813067</td>
      <td>0.047860</td>
      <td>0.359943</td>
      <td>0.199574</td>
      <td>0.111275</td>
      <td>3</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.199245</td>
      <td>0.206612</td>
      <td>0.719601</td>
      <td>0.159910</td>
      <td>0.328582</td>
      <td>1.000000</td>
      <td>0.236829</td>
      <td>3</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.612842</td>
      <td>0.613636</td>
      <td>0.905626</td>
      <td>0.525338</td>
      <td>0.750535</td>
      <td>0.284869</td>
      <td>0.475135</td>
      <td>1</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.911237</td>
      <td>0.929752</td>
      <td>0.740472</td>
      <td>0.797297</td>
      <td>0.949394</td>
      <td>0.667789</td>
      <td>0.821763</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>210 rows × 8 columns</p>
</div>



Csv To List 2D

we need a List 2D for using our algorithm.


```python
data = np.array(data_csv).tolist()

for row in data:
  # keep integer value insteed of float for output column
  row[-1] = round(row[-1])

data[0:2]
```




    [[0.07271010387157692,
      0.1322314049586778,
      0.27313974591651463,
      0.15540540540540532,
      0.08909479686386312,
      0.426855114485951,
      0.36632200886262917,
      3],
     [0.5835694050991501,
      0.6632231404958676,
      0.5054446460980035,
      0.5788288288288291,
      0.5759087669280114,
      0.5402358631629588,
      0.6282619399310684,
      2]]



Spliting to train = 80%  and  test = 20%


```python
# get counts
n_train = round(len(data_csv)*0.8)
n_test = len(data_csv) - n_train

# print counts
print(n_train)
print(n_test)

# code to split it into 2 lists 
train_data = [data[i] for i in range(n_train)] 
test_data = [data[i] for i in range(n_train, n_train + n_test)] 

print(train_data[0:2])
print(test_data[0:2])

# verifications : 
print(len(train_data))
print(len(test_data))
# Ok Good ! 
```

    168
    42
    [[0.07271010387157692, 0.1322314049586778, 0.27313974591651463, 0.15540540540540532, 0.08909479686386312, 0.426855114485951, 0.36632200886262917, 3], [0.5835694050991501, 0.6632231404958676, 0.5054446460980035, 0.5788288288288291, 0.5759087669280114, 0.5402358631629588, 0.6282619399310684, 2]]
    [[0.6978281397544854, 0.7107438016528925, 0.8275862068965515, 0.6081081081081082, 0.7533856022808265, 0.19398249879727988, 0.6893156080748398, 2], [0.5221907459867801, 0.5351239669421487, 0.8339382940108894, 0.4560810810810809, 0.6094084105488238, 0.1956728081238867, 0.4549483013293942, 1]]
    168
    42
    

Train the Model


```python
# get parameters
n_inputs = len(train_data[0]) - 1
n_outputs = len(set([row[-1] for row in train_data])) + 1

# initialize network 
network = initialize_network(n_inputs, 4, n_outputs)

# train the model
train_network(network, train_data, 0.5, 20, n_outputs)

for layer in network:
	print(layer)
```

    >epoch=0, lrate=0.500, error=135.745
    >epoch=1, lrate=0.500, error=105.033
    >epoch=2, lrate=0.500, error=87.099
    >epoch=3, lrate=0.500, error=72.805
    >epoch=4, lrate=0.500, error=65.628
    >epoch=5, lrate=0.500, error=61.219
    >epoch=6, lrate=0.500, error=57.241
    >epoch=7, lrate=0.500, error=52.655
    >epoch=8, lrate=0.500, error=47.222
    >epoch=9, lrate=0.500, error=41.453
    >epoch=10, lrate=0.500, error=36.266
    >epoch=11, lrate=0.500, error=32.184
    >epoch=12, lrate=0.500, error=29.128
    >epoch=13, lrate=0.500, error=26.832
    >epoch=14, lrate=0.500, error=25.070
    >epoch=15, lrate=0.500, error=23.686
    >epoch=16, lrate=0.500, error=22.572
    >epoch=17, lrate=0.500, error=21.657
    >epoch=18, lrate=0.500, error=20.891
    >epoch=19, lrate=0.500, error=20.239
    [{'weights': [2.989117999234495, 2.335155673499777, 0.9107358855311519, 2.206852306411716, 2.2990401726516136, -2.054594127065679, -0.17069643981591334, -2.160074773650044], 'output': 0.8813426946121891, 'delta': -0.00148516554937075}, {'weights': [3.082651112695033, 2.381140028895888, -1.887048667033515, 1.1420933835593898, 1.925700668457285, 0.8282911551084087, 3.3708756603776586, -6.209153573429914], 'output': 0.05164211360096801, 'delta': -0.0023180752344039773}, {'weights': [-0.3775029029175826, -0.010441870126216373, -1.0174470588111948, -0.35547545539660763, -0.18118934629042485, 2.5251747520466687, 2.3791630681720997, -0.4242551718414272], 'output': 0.38763588958512857, 'delta': 1.672466320046384e-05}, {'weights': [2.4689084023124135, 1.5946875290968496, 2.0304023064690955, 0.8916314754935482, 2.7054153947530266, -3.3204446589140217, -1.8151796091393133, -0.6505240414934472], 'output': 0.9372370098452493, 'delta': 0.0006700596432940353}]
    [{'weights': [-0.6926419461761412, -0.5929562455950033, -1.5571614239475464, -1.1649438036824662, -2.1007380957286506], 'output': 0.011692284450753467, 'delta': -0.0001351110691329072}, {'weights': [2.0063102231839816, -5.412575877018794, -3.2549281759966995, 3.8112087027829067, -0.61075186835223], 'output': 0.9603090856054861, 'delta': 0.0015128408618377376}, {'weights': [2.7091738878758553, 5.893468111522868, -0.8114198875505062, -0.6890808150159384, -4.101589881463914], 'output': 0.0862464175525897, 'delta': -0.0067969053468604846}, {'weights': [-4.385690046088722, -3.31489903680152, 2.863382475997035, -3.2026374299266265, 1.8089000578370957], 'output': 0.016001987261729616, 'delta': -0.00025196606991797864}]
    

Got Predictions for test_data and train_data : 


```python
print("\n\nPredict on train ----------------------")
predict_on_DataSet(network, train_data)

print("\n\nPredict on test ----------------------")
predict_on_DataSet(network, test_data)
```

    
    
    Predict on train ----------------------
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=2, Got=1
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=1
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=1
    Expected=2, Got=2
    Expected=3, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=3, Got=1
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=3
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=1
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=2
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=2, Got=1
    Expected=2, Got=2
    Expected=1, Got=1
    accuracy =  92.85714285714286 %
    
    
    Predict on test ----------------------
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=2
    Expected=2, Got=2
    Expected=1, Got=1
    Expected=3, Got=1
    Expected=1, Got=1
    Expected=2, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=1
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=1, Got=1
    Expected=3, Got=3
    Expected=2, Got=2
    Expected=3, Got=3
    Expected=2, Got=1
    Expected=2, Got=2
    Expected=3, Got=1
    Expected=3, Got=3
    Expected=1, Got=1
    Expected=2, Got=2
    accuracy =  88.09523809523809 %
    

Good Accuracy ! Close to 90 %

In order to do better we can play with parameters like learning rate, epochs, etc...
