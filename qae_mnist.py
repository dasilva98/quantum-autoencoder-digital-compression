import json
import os
import time
#import warnings
#import traceback
import mnist_loader

import random
import matplotlib.pyplot as plt

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from pathlib import Path
from IPython.display import clear_output
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit.exceptions import QiskitError

algorithm_globals.random_seed = 42

# Ansatz Function
def ansatz(num_qubits):
    return RealAmplitudes(num_qubits, reps=5)

# Autoencoder Function
def auto_encoder_circuit(num_latent, num_trash):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier() # not mandatory
    auxiliary_qubit = num_latent + 2 * num_trash
    # swap test
    circuit.h(auxiliary_qubit)# h gate
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)

    circuit.h(auxiliary_qubit)# h gate
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit


# Interpret for SamplerQNN
def identity_interpret(x):
    return x

# Cost Function
def cost_func_digits(params_values):

    probabilities = qnn.forward(x_axis, params_values)
    cost = np.sum(probabilities[:, 1]) / x_axis.shape[0]

    objective_func_vals.append(cost)

    return cost

def cost_func_digits_2(params_values):

    x_axis_sample
    probabilities = qnn.forward(x_axis_sample, params_values)
    cost = np.sum(probabilities[:, 1]) / len(x_axis_sample)

    objective_func_vals.append(cost)

    return cost

def cost_func_digits_sample(params_values):
    length = len(x_axis_sample) 
    try:
        probabilities = qnn.forward(x_axis_sample, params_values)
    except QiskitError as e:
        print("A QiskitError occurred: ", e)
        # Handle the normalization issue if that's the problem
        #if 'amplitudes-squared is not 1' in str(e):
        print("There is a normalization issue with the input state.")
        # Add additional debugging information or corrective measures here
        # For example, print out the parameters you used
        print("Parameters used:", params_values)
        # Check if parameters contain nan
        if np.isnan(params_values).any():
            print("The parameters contain nan values.")
        # Normalize the parameters if that's the issue
        norm = np.linalg.norm(params_values)
        if norm == 0:
            print("Cannot normalize a zero vector.")
        else:
            params_values = params_values / norm
            print("Normalized parameters:", params_values)
            # Retry the initialization with the normalized parameters
            # result = some_qiskit_function_that_might_fail(parameters)
        cost = np.sum(probabilities[:, 1]) / np.array(x_axis_sample).shape[0] # maybe 'len(x_axis_sample)'?
        objective_func_vals.append(cost)
        return cost
    except Exception as e:
        # Handle other exceptions that are not QiskitErrors
        print("An unexpected error occurred: ", e)

    

input_path = 'input\\'
training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte\\train-images-idx3-ubyte')
training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte\\train-labels-idx1-ubyte')
#test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte\\t10k-images-idx3-ubyte')
#test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(12,4))
    index = 1  
    #plt.colorbar(mappable=plt.cm.gray)
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        plt.colorbar()
        if (title_text != ''):
            plt.title(title_text, fontsize = 10);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = mnist_loader.MnistDataloader(training_images_filepath, training_labels_filepath)
x_axis, y_axis = mnist_dataloader.load_data()

print('x_axis: ' + str(np.array(x_axis).shape))
print('y_axis: ' + str(np.array(y_axis).shape))

#
# Rescale the images from [0,255] to the [0.0,1.0] range.
#

x_axis = np.array(x_axis)  # Convert x_axis to a NumPy array
#x_axis = x_axis[..., np.newaxis]/255.0
x_axis = x_axis / 255.0

# Show some random images 
x_axis_sample = []
sample_indexes = []
y_axis_sample = []
images_2_show = []
titles_2_show = []
    
for i in range(0, 5):
    r = random.randint(1, len(y_axis))
    print("x_axis[r]: \n", np.array(x_axis[r]))
    images_2_show.append(x_axis[r])
    titles_2_show.append('image [' + str(r) + '] = ' + str(y_axis[r]))  
    x_axis_sample.append(x_axis[r])
    y_axis_sample.append(y_axis[r])
    sample_indexes.append(r)

show_images(images_2_show, titles_2_show)
plt.show()

#
# Quantum Autoenconder Section
#

# Latent and trash qubits
num_latent = 3
num_trash = 1

# A quantum feature map encodes classical data 
# to the quantum state space by using a quantum circuit

# fm = ZZFeatureMap(8, entanglement='linear')

fm = RawFeatureVector(2 ** (num_latent + num_trash))
# fm.decompose().draw("mpl", style="iqp")
plt.show()
ae = auto_encoder_circuit(num_latent, num_trash)

qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
qc = qc.compose(fm, range(num_latent + num_trash))
qc = qc.compose(ae)

qc.draw("mpl", style="iqp")
plt.show()

qnn = SamplerQNN(
    circuit=qc,
    input_params=fm.parameters,
    weight_params=ae.parameters,
    interpret=identity_interpret,
    output_shape=2,
)

# By minimizing this cost function, we can determine 
# the required parameters to compress our noisy images. 
opt = COBYLA(maxiter=150)
initial_point = algorithm_globals.random.random(ae.num_parameters)

# initial_point = np.random.random(10*(num_layers))
#with open("12_qae_initial_point.json", "r") as f:
#    initial_point = json.load(f)
print("ae.num_parameters", ae.num_parameters)
print("initial_point: ",initial_point)
objective_func_vals = []

plt.rcParams["figure.figsize"] = (12, 6) # make the plot nicer

start = time.time()
print("Running optimization..")
#np.seterr(divide='ignore', invalid='ignore')
opt_result = opt.minimize(fun=cost_func_digits_2,x0=initial_point) #change 'fun=' parameter to change the sample used
elapsed = time.time() - start
print(f"Fit in {elapsed:0.2f} seconds")

# plotting of the Objective function value against Iteration
clear_output(wait=True)
plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()