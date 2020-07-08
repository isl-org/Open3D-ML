import re
import os
import numpy as np
import tensorflow as tf
from pprint import pprint
import torch
def load_tf_weights(model, weight_path):


	# Retrieve weights from TF checkpoint
	tf_path = os.path.abspath(weight_path)
	init_vars = tf.train.list_variables(tf_path)
	tf_vars = []
	for name, shape in init_vars:
	    #print("Loading TF weight {} with shape {}".format(name, shape))
	    array = tf.train.load_variable(tf_path, name)
	    tf_vars.append((name, array.squeeze()))


	# FOr each variable in the PyTorch model
	for name, array in tf_vars:
	    # skip the prefix ('model/') and split the path-like variable name in a list of sub-path
	    

	    whole_name = name
	    name = name.split('/')

	    if (name[0]=="optimizer"):
	    	continue
	    name = name[1:]


	    # Initiate the pointer from the main model class
	    pointer = model

	    # We iterate along the scopes and move our pointer accordingly
	    for m_name in name:
	
	        l = [m_name]

	        #print(l[0])

	        # Convert parameters final names to the PyTorch modules equivalent names
	        last_pointer = pointer
	        if l[0] == 'beta':
	            pointer = getattr(pointer, 'bias')
	        elif l[0] == 'gamma':
	            pointer = getattr(pointer, 'weight')
	        elif l[0] == 'moving_mean':
	            pointer = getattr(pointer, 'running_mean')
	        elif l[0] == 'moving_variance':
	            pointer = getattr(pointer, 'running_var')
	        elif l[0] == 'kernel':
	            pointer = getattr(pointer, 'weight')
	        elif l[0] == 'w' or l[0] == 'g':
	            pointer = getattr(pointer, 'weight')
	        elif l[0] == 'b':
	            pointer = getattr(pointer, 'bias')
	        else:
	            #pprint(vars(pointer))
	            pointer = getattr(pointer, l[0])
	   
	    if (l[0]=='kernel'):
	        array = np.transpose(array)

	    if (len(pointer.shape)==4 and pointer.shape[2]==pointer.shape[3] and pointer.shape[3]==1):
	        array = np.transpose(array)
	        array = np.expand_dims(array, axis=2)
	        array = np.expand_dims(array, axis=2)
	     

	    try:
	        assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
	    except AssertionError as e:
	        print(whole_name)
	        #pprint(vars(last_pointer))
	        #print(last_pointer.weights.shape)
	        e.args += (pointer.shape, array.shape)
	        raise

	    #print("Initialize PyTorch weight {}".format(name))
	    pointer.data = torch.from_numpy(array)
	print("Initialize torch weights from {}".format(weight_path))