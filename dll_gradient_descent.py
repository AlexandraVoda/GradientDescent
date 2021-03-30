import numpy as np
import matplotlib.pyplot as plt


def param_initialization(layer_dimensions):
    """
    Arguments: 
    layer_dimensions: array of integeres representing the number of units in each layer
    
    Returns:
    dictionary of initialized weights (using Xavier initialization) and biases for each layer
    """
    
    params = {}
    
    for i in range(1, len(layer_dimensions)):
        params["W" + str(i)] = np.random.randn(layer_dimensions[i], layer_dimensions[i-1]) / np.sqrt(layer_dimensions[i-1])
        params["b" + str(i)] = np.zeros([layer_dimensions[i], 1])
        
    return params


def sigmoid(Z):
    """
    Computes Sigmoid activation function for a given input
    """
    
    return 1/(1+np.exp(-Z))


def relu(Z):
    """
    Computes ReLU activation function for a given input
    """
    return np.maximum(0, Z)


def forward_propagation_one_step(A_prev, W, b, activation):
    """
    Arguments: 
    A_prev: activations from previous layer; shape: (previous_layer_size, number_of_examples)
    W: current layer weight; shape: (current_layer_size, previous_layer_size)
    b: current layer bias; shape: (current_layer_size, 1)
    activation: current layer's activation function; can be "relu" or "sigmoid"
    
    Returns:
    Z = W * A_prev + b, the activation input
    A = g(Z), the activation function result; g = relu/sigmoid
    """
    
    Z = np.dot(W, A_prev) + b
    A = None
    if activation == 'sigmoid':
        A = sigmoid(Z)
    else:
        A = relu(Z)
    
    return A, Z
        
    
def forward_propagation_multiple_steps(X, params):
    """
    Arguments: 
    X: training examples; shape: (number_of_features, number_of_examples)
    params: dictionary of weights and biases
    
    Returns:
    AL: the activation of the last layer; shape: (1, no_of_examples)
    outputs: array of partial forward propagation results and parameters
    """
    outputs = []
    L = len(params)//2
    A_prev = X
    
    # For the first no_of_layers-1 apply ReLU activation function
    for l in range(1, L):
        A, Z = forward_propagation_one_step(A_prev, params["W" + str(l)], 
                                              params["b" + str(l)], "relu")
        
        step_vars = (params["W" + str(l)], params["b" + str(l)], A_prev, Z)
        outputs.append(step_vars)
        A_prev = A
    
    # Apply sigmoid for the last layer
    AL, Z = forward_propagation_one_step(A_prev, params["W" + str(L)], 
                                          params["b" + str(L)], "sigmoid")
    step_vars = (params["W" + str(L)], params["b" + str(L)], A_prev, Z)
    outputs.append(step_vars)
    
    return AL, outputs


def compute_cost(A, Y):
    """
    Arguments: 
    A: the output of the neural net; shape: (1, no_of_examples)
    Y: correct labels; shape: (1, no_of_examples)
    
    Returns:
    cost: cross-entropy cost; scalar
    """
    
    m = Y.shape[1]
    
    cost = -1/m * np.sum( np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A)))
    cost = np.squeeze(cost) # to obtain a scalar
    
    return cost


def sigmoid_backward(Z, dA):
    """
    Arguments:
    dA: previous gradient
    Z: variable with respect to which the gradient is calculated
    
    Returns:
    the gradient of a sigmoid hidden unit with respect to Z
    """
    # dZ = dA * sigma'(Z) = dA * sigma(Z) * ( 1 - sigma(Z))
    A = sigmoid(Z)
    result = dA * A * (1-A)
    
    return result


def relu_backward(Z, dA):
    """
    Arguments:
    dA: previous gradient
    Z: variable with respect to which the gradient is calculated
    
    Returns:
    the gradient of a relu hidden unit with respect to Z
    """
    # dZ = dA * relu'(Z) = dA * (1 if Z > 0 else 0)
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0 
    
    return dZ


def backward_propagation_linear(dZ, cached_data):
    '''
    Arguments:
    dZ: gradient of the cost with respect to current Z; shape = same of Z
    cached_data: weights and activation result from the forward step
    
    Returns:
    dA_prev: gradient of the cost with respect to the activation of the previous layer; shape: same of A_prev
    dW: gradient of the cost with respect to the current layer's weight; shape: same of W
    db: gradient of the cost with respect to the current layer's bias; shape: same of b
    '''
    
    A_prev, W, b = cached_data
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=1)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def backward_propagation_activation(dA, cache, activation):
    '''
    Arguments:
    dA: gradient of the cost with respect to current activation; shape = same of A
    cache: weights and activation result from the forward step
    activation: activation function of current layer; can be "sigmoid" or "relu"
    
    Returns:
    dA_prev: gradient of the cost with respect to the activation of the previous layer; shape: same of A_prev
    dW: gradient of the cost with respect to the current layer's weight; shape: same of W
    db: gradient of the cost with respect to the current layer's bias; shape: same of b
    '''
    
    W, b, A_prev, Z = cache
    dZ = None
    if activation == 'sigmoid':
        dZ = sigmoid_backward(Z, dA)
        dA_prev, dW, db = backward_propagation_linear(dZ, (A_prev, W, b))
    else:
        dZ = relu_backward(Z, dA)
        dA_prev, dW, db = backward_propagation_linear(dZ, (A_prev, W, b))

    return dA_prev, dW, db


def backward_propagation_multiple_steps(Y, AL, caches):
    '''
    Arguments:
    Y: correct labels
    A: network output labels
    caches: array of forward propagation partial results that will be used for gradient computing
    
    Returns:
    grads: dictionary of gradients computed in the backprop step for each layer; used in parameter update step
    '''
    
    dAL = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    L = len(caches)
    
    grads = { "dA" + str(L): dAL }

    current_cache = caches[L-1]
    W, b, A_prev, Z = current_cache
    dA_prev, dW, db = backward_propagation_activation(dAL, current_cache, "sigmoid")
    grads['dA' + str(L-1)] = dA_prev
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = backward_propagation_activation(dA_prev, current_cache, "relu")
        grads['dA' + str(l)] = dA_prev
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db
    
    return grads


def update_params(params, grads, alfa):
    '''
    Arguments:
    params: dictionary of current network parameters that need to be updated after each gradient descent iteration
    grads: dictionary of gradients computed in the backprop step
    alfa: learning rate
    
    Returns:
    params: updated parameters
    '''
    
    L = len(params)//2
    
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - alfa * grads["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - alfa * grads["db" + str(l+1)]
    
    return params

def predict (X, Y, params):
    '''
    Arguments:
    X: testing set used for prediction
    Y: testing correct labels used for accuracy score calculation
    params: trained neural net parameters
    
    Returns:
    predicted_classes: array of predicted labels for input X
    '''
    
    m = X.shape[1] #number of examples
    predicted_classes = np.zeros((1,m))

    AL, outputs = forward_propagation_multiple_steps(X, params) #AL = resulted probabilities; has (1,m) shape

    for i in range(m):
        predicted_classes[0, i] = 1 if AL[0, i] > 0.5 else 0
        
    accuracy = np.sum(predicted_classes == Y)/m
    
    print("Accuracy = {}".format(accuracy))
    
    return predicted_classes
    
    
def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predicted and correct label did not match
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    num_images = len(mislabeled_indices[0])
    plt.rcParams['figure.figsize'] = (40.0, 40.0) 
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))