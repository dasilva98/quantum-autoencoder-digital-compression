import json
import os
import time
#import warnings
#import traceback
import mnist_loader

import random
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from IPython.display import clear_output
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN


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
    circuit.barrier() # not mandatory, still unsure if I will need it in the future
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

#
# Rescale the images from [0,255] to the [0.0,1.0] range.
#

#x_axis = np.array(x_axis)  # Convert x_axis to a NumPy array
#x_axis = x_axis[..., np.newaxis]/255.0

#
# Show some random images 
#

images_2_show = []
titles_2_show = []
for i in range(0, 5):
    r = random.randint(1, len(y_axis))
    images_2_show.append(x_axis[r])
    titles_2_show.append('image [' + str(r) + '] = ' + str(y_axis[r]))    

show_images(images_2_show, titles_2_show)
plt.show()

#
# Quantum Autoenconder Section
#

# Latent and trash qubits
num_latent = 3
num_trash = 2

# A quantum feature map encodes classical data 
# to the quantum state space by using a quantum circuit
fm = RawFeatureVector(2 ** (num_latent + num_trash))

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

objective_func_vals = []

plt.rcParams["figure.figsize"] = (12, 6) # make the plot nicer

start = time.time()
opt_result = opt.minimize(fun=cost_func_digits,x0=initial_point)
elapsed = time.time() - start
print(f"Fit in {elapsed:0.2f} seconds")

# plotting of the Objective function value against Iteration
clear_output(wait=True)
plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()