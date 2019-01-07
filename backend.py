'''
Contains the classes required to implement the neural network
models.
'''
import numpy as np

def xavier_initializer(*dims):
    '''
    given the input and output dimensions,
    gives a weight matrix using Xavier Initialization.
    uses the version with uniform distribution
    can accept more than one dimensions
    basically adds all given dimensions
    '''
    shape = []

    # fill shape
    for dim in dims:
        shape.append(dim)

    if len(shape) == 2:
        boundary = np.sqrt(6/(shape[0]+shape[1]))
    elif len(shape) == 4:
        boundary = np.sqrt(6/(shape[0]+shape[1]*shape[2]*shape[3]))
    else:
        print("Error Occurred")
        return None

    return np.random.uniform(-1*boundary, boundary, size=shape)


class PlaceHolder(object):
    '''
    placeholders for training and test data
    they only have output fields, wont take inputs
    '''
    def __init__(self, dim):
        '''
        takes the dimension of one data point, to initialize the variables
        '''
        self.out_dim = dim
        # for multidimensional input, dim should be a tuple or list

    def load(self, data=None):
        '''
        function to load value to placeholders
        '''
        self.output = data


class SoftMaxXent(object):
    '''
    The combo layer that encompasses both softmax and xent
    '''
    def __init__(self, precursor=None):
        '''
        adds a connection to the precursor
        default precursor in None
        '''
        self.precursor = precursor

    def fpass(self, labels):
        '''
        takes an N x D matrix as input (x) (output of precursor)
        takes an N x D label matrix (l)
        computes the loss (y)
        also stores the sigma value to be used for backprop
        '''
        # if precursor is not a placeholder, do fpass before using output
        if self.precursor.__class__.__name__ != "PlaceHolder":
            self.precursor.fpass()

        # now use the updated value of precursor.output
        logits = np.copy(self.precursor.output)
        # subtracting the max to prevent overflow
        logits = logits - np.amax(logits, axis=1)[:, np.newaxis]
        # take log and then take exp to prevent numerical issues
        numerator = logits
        # denominator is tiled, log of sum of numerator
        denominator = np.log(np.sum(np.exp(logits), axis=1))[:, np.newaxis]
        self.sigma = np.exp(numerator - denominator)
        # the following is not really required, so not calculated
        #self.output = -1*np.sum(np.log(np.sum(self.sigma*labels, axis=1)))

    def bprop(self, l_rate, labels):
        '''
        there are no variables
        just finds gradients
        '''
        # if the input is not a placeholder
        if self.precursor.__class__.__name__ != "PlaceHolder":
            # find the gradient wrt to input
            grad_wrt_in = self.sigma - labels
            # call the backprop function of precursor
            self.precursor.bprop(l_rate, grad_wrt_in)


class Linear(object):
    '''
    The linear layer with weight and bias
    '''
    def __init__(self, out_dim, precursor=None):
        '''
        initializes the variables
        out_dim is the number of output dimensions
        adds a connection to the precursor
        default precursor in None
        '''
        self.out_dim = out_dim
        self.precursor = precursor
        self.weight = xavier_initializer(self.precursor.out_dim,
                                         self.out_dim)
        self.bias = np.zeros(out_dim)

    def fpass(self):
        '''
        takes N x A matrix as input (output of precursor)
        gives N x out_dim matrix as output - B is an argument
        '''
        # if precursor is not a placeholder, do fpass before using output
        if self.precursor.__class__.__name__ != "PlaceHolder":
            self.precursor.fpass()

        # now use the updated value of precursor.output
        self.output = self.precursor.output @ self.weight + self.bias

    def bprop(self, l_rate, grad_wrt_out):
        '''
        updates variables
        andfinds the gradients
        '''
        # find the gradients before updating the parameters
        # if the input is not a placeholder
        if self.precursor.__class__.__name__ != "PlaceHolder":
            # find the gradient wrt to input
            grad_wrt_in = grad_wrt_out @ np.transpose(self.weight)
            # call the backprop function of precursor
            self.precursor.bprop(l_rate, grad_wrt_in)

        # update the paramters based on grad_wrt_out
        w_grad = np.transpose(self.precursor.output) @ grad_wrt_out
        self.weight = self.weight - l_rate * w_grad
        b_grad = np.sum(grad_wrt_out, axis=0)
        self.bias = self.bias - l_rate * b_grad


class Relu(object):
    '''
    this is simply an activation layer
    '''
    def __init__(self, precursor=None):
        '''
        adds a connection to precursor
        '''
        self.precursor = precursor
        self.out_dim = precursor.out_dim

    def fpass(self):
        '''
        performs relu activation
        makes negative elements of the matrix 0
        '''
        # if precursor is not a placeholder, do fpass before using output
        if self.precursor.__class__.__name__ != "PlaceHolder":
            self.precursor.fpass()

        self.output = np.copy(self.precursor.output)
        self.output[np.where(self.output < 0)] = 0
        # this mask will be used during backprop
        self.mask = np.copy(self.output)
        self.mask[np.where(self.output != 0)] = 1

    def bprop(self, l_rate, grad_wrt_out):
        '''
        no variables to update
        simply finds the gradient
        '''
        # if the input is not a placeholder
        if self.precursor.__class__.__name__ != "PlaceHolder":
            # find the gradient wrt to input
            grad_wrt_in = grad_wrt_out * self.mask
            # call the backprop function of precursor
            self.precursor.bprop(l_rate, grad_wrt_in)


class Flatten(object):
    '''
    this is a simple flattening layer
    '''
    def __init__(self, precursor=None):
        '''
        assigns the precursor
        finds the flattened out_dim
        '''
        self.precursor = precursor
        self.out_dim = np.prod(precursor.out_dim)

    def fpass(self):
        '''
        fill the self.output
        the way in which flattennig happens does not matter
        '''
        # if precursor is not a placeholder, do fpass before using output
        if self.precursor.__class__.__name__ != "PlaceHolder":
            self.precursor.fpass()

        # now use the updated output
        self.output = np.reshape(self.precursor.output, (-1, self.out_dim))

    def bprop(self, l_rate, grad_wrt_out):
        '''
        no variables to update
        simply finds the gradient
        '''
        # if the input is not a placeholder
        if self.precursor.__class__.__name__ != "PlaceHolder":
            # find grad_wrt_in from grad_wrt_out
            grad_wrt_in = np.reshape(grad_wrt_out,
                                     (-1,
                                      self.precursor.out_dim[0],
                                      self.precursor.out_dim[1],
                                      self.precursor.out_dim[2]))

            # call the backprop function of precursor
            self.precursor.bprop(l_rate, grad_wrt_in)


class Maxpool(object):
    '''
    this is simply an activation layer (for convolution)
    with 2x2 kernel (fixed) - does not generalize
    no overlap
    '''
    def __init__(self, precursor=None):
        '''
        adds a connection to precursor
        '''
        self.precursor = precursor
        self.out_dim = precursor.out_dim
        # no of rows and columns get divided by 2
        self.out_dim[0] = (self.out_dim[0]/2).astype("int")
        self.out_dim[1] = (self.out_dim[1]/2).astype("int")
        # no. of channels are the same

    def fpass(self):
        '''
        performs maxpool activation
        find and store max_max (for bprop)
        '''
        # if precursor is not a placeholder, do fpass before using output
        if self.precursor.__class__.__name__ != "PlaceHolder":
            self.precursor.fpass()

        # change order to bring channel before row,col
        input_data = np.einsum('ijkl->iljk', self.precursor.output)
        # split into 2x2 blocks
        stride = input_data.strides
        mem_view = np.lib.stride_tricks.as_strided(
                        input_data,
                        (self.precursor.output.shape[0],  # batch size
                         self.precursor.output.shape[3],  # channel size
                         self.out_dim[0],  # downsampled row_size
                         self.out_dim[1],  # downsampled col_size
                         2, 2),  # window size
                        (stride[0],
                         stride[1],
                         stride[2]*2,
                         stride[3]*2,
                         stride[2],
                         stride[3]))
        # find max in each block
        max_view = np.amax(mem_view, axis=(4, 5), keepdims=True)
        # squeeze to get output
        self.output = np.squeeze(max_view)
        # rearrange output to usual order
        self.output = np.einsum('abcd->acdb', self.output)

        # prepare mask for use with backprop
        max_view = np.broadcast_to(max_view,
                                   (max_view.shape[0],
                                    max_view.shape[1],
                                    max_view.shape[2],
                                    max_view.shape[3],
                                    2, 2))
        temp_max_mask = np.equal(max_view, mem_view).astype("int")

        # remove duplicates in mask
        temp = np.reshape(temp_max_mask, (-1, 4))
        mask = np.zeros_like(temp)
        mask[np.arange(temp.shape[0]), np.argmax(temp, axis=1)] = 1
        self.max_mask = np.reshape(mask,
                                   (max_view.shape[0],
                                    max_view.shape[1],
                                    max_view.shape[2],
                                    max_view.shape[3],
                                    2, 2))

    def bprop(self, l_rate, grad_wrt_out):
        '''
        no variables to update
        simply finds the gradient
        '''
        # if the input is not a placeholder
        if self.precursor.__class__.__name__ != "PlaceHolder":
            # find the gradient wrt to input - following steps
            # reorder the gradient_wrt_out
            grad = np.einsum('abcd->adbc', grad_wrt_out)
            # separate the individual elements
            grad = np.reshape(grad,
                              (grad.shape[0],
                               grad.shape[1],
                               grad.shape[2],
                               grad.shape[3],
                               1, 1))
            # broadcast
            grad = np.broadcast_to(grad,
                                   (grad.shape[0],
                                    grad.shape[1],
                                    grad.shape[2],
                                    grad.shape[3],
                                    2, 2))
            # applymask
            grad_wrt_in = grad * self.max_mask
            # concatenate
            grad_wrt_in = np.einsum('abcdef->cdefab',grad_wrt_in)
            grad_wrt_in = np.concatenate(grad_wrt_in, axis=1)
            grad_wrt_in = np.concatenate(grad_wrt_in, axis=1)
            # reorder
            grad_wrt_in = np.einsum('abcd->cabd', grad_wrt_in)

            # call the backprop function of precursor
            self.precursor.bprop(l_rate, grad_wrt_in)


class Conv(object):
    '''
    The Convolutional Layer
    '''
    def __init__(self, k_index, k_row, k_col, precursor=None):
        '''
        initializes the variables
        k_index = no. of kernels to consider, same as output channels
        k_row = no. of rows in kernel (should be odd)
        k_col = no. of columns in kernel (should be odd)
        k_channel = number of channels in the input image (eg, RGB = 3)
        adds a connection to the precursor
        default precursor in None
        precursor should have data in the format
        [batch_no, row_no, col_no, channel_no]
        '''
        self.k_index = k_index
        self.k_row = k_row
        self.k_col = k_col
        self.precursor = precursor
        self.out_dim = np.copy(self.precursor.out_dim)
        self.out_dim[2] = self.k_index
        self.k_channel = self.precursor.out_dim[2]

        # initialize the kernel using xavier initializer
        # we simply have to add dimensions to mimic a fully connected layer
        self.kernel = xavier_initializer(self.k_index,
                                         self.k_row,
                                         self.k_col,
                                         self.k_channel)

    def fpass(self):
        '''
        takes output of precursor as input
        performs convolutions and gives output
        '''
        # if precursor is not a placeholder, do fpass before using output
        if self.precursor.__class__.__name__ != "PlaceHolder":
            self.precursor.fpass()

        # now use the updated value of precursor.output
        # STEP 1: Add padding to rows and cols
        input_data = np.copy(self.precursor.output)
        input_data = np.pad(input_data,
                            ((0, 0),  # no padding for batch
                             (self.k_row-np.ceil(self.k_row/2).astype(int),
                              self.k_row-np.ceil(self.k_row/2).astype(int)),
                             (self.k_col-np.ceil(self.k_col/2).astype(int),
                              self.k_col-np.ceil(self.k_col/2).astype(int)),
                             (0, 0)),  # no padding for channel
                            "constant")  # 0 is the default constant

        # STEP 2: create a convolution friendly memory view using stride tricks
        stride = input_data.strides
        mem_view = np.lib.stride_tricks.as_strided(
                        input_data,
                        # size of output
                        (self.precursor.output.shape[0],
                         self.precursor.output.shape[1],
                         self.precursor.output.shape[2],
                         self.k_row,
                         self.k_col,
                         self.k_channel),
                        # stride lengths
                        (stride[0],
                         stride[1],
                         stride[2],
                         stride[1],
                         stride[2],
                         stride[3]))

        # STEP 3: perform actual convolution using einstein summation
        self.output = np.einsum('abcdef,idef->abci',
                                mem_view,
                                self.kernel)

    def grad_kernel(self, grad_wrt_out):
        '''
        find the loss gradient with respect to kernel
        we need the input (output of precursor) to calculate this
        '''
        # STEP 1: Add padding to rows and cols of input
        input_data = np.copy(self.precursor.output)
        input_data = np.pad(input_data,
                            ((0, 0),  # no padding for batch
                             (self.k_row-np.ceil(self.k_row/2).astype(int),
                              self.k_row-np.ceil(self.k_row/2).astype(int)),
                             (self.k_col-np.ceil(self.k_col/2).astype(int),
                              self.k_col-np.ceil(self.k_col/2).astype(int)),
                             (0, 0)),  # no padding for channel
                            "constant")  # 0 is the default constant

        # STEP 2: create a memview to suit the equation
        stride = input_data.strides
        mem_view = np.lib.stride_tricks.as_strided(
                        input_data,
                        # size of output
                        (self.precursor.output.shape[0],
                         self.k_row,
                         self.k_col,
                         self.precursor.output.shape[1],
                         self.precursor.output.shape[2],
                         self.k_channel),
                        # stride lengths
                        (stride[0],
                         stride[1],
                         stride[2],
                         stride[1],
                         stride[2],
                         stride[3]))

        # STEP 3: perform actual convolution using einstein summation
        grad_wrt_kernel = np.einsum('aqrmns,amnp->pqrs',
                                    mem_view,
                                    grad_wrt_out)
        return grad_wrt_kernel

    def grad_input(self, grad_wrt_out):
        '''
        find the loss gradient with respect to input
        we need the input (output of precursor) to calculate this
        '''
        # STEP 1: Add padding to rows and cols of input
        grad_data = np.copy(grad_wrt_out)
        grad_data = np.pad(grad_data,
                           ((0, 0),  # no padding for batch
                            (self.k_row-np.ceil(self.k_row/2).astype(int),
                             self.k_row-np.ceil(self.k_row/2).astype(int)),
                            (self.k_col-np.ceil(self.k_col/2).astype(int),
                             self.k_col-np.ceil(self.k_col/2).astype(int)),
                            (0, 0)),  # no padding for channel
                           "constant")  # 0 is the default constant

        # STEP 2: create a memview to suit the equation
        stride = grad_data.strides
        mem_view = np.lib.stride_tricks.as_strided(
                        grad_data,
                        # size of output
                        (self.precursor.output.shape[0],
                         self.k_row,
                         self.k_col,
                         self.precursor.output.shape[1],
                         self.precursor.output.shape[2],
                         self.k_index),
                        # stride lengths
                        (stride[0],
                         stride[1],
                         stride[2],
                         stride[1],
                         stride[2],
                         stride[3]))

        # STEP 3: swap the rows and columns of kernel
        swap_kernel = np.copy(self.kernel)
        swap_kernel = np.flip(swap_kernel, 1)
        swap_kernel = np.flip(swap_kernel, 2)

        # STEP 4: perform actual convolution using einstein summation
        grad_wrt_input = np.einsum('pmnqri,imns->pqrs',
                                   mem_view,
                                   swap_kernel)
        return grad_wrt_input

    def bprop(self, l_rate, grad_wrt_out):
        '''
        updates variables and send back gradients
        using the grad_kernel and grad_input functions
        '''
        # if the input is not a placeholder
        if self.precursor.__class__.__name__ != "PlaceHolder":
            # get gradient wrt to input
            grad_wrt_in = self.grad_input(grad_wrt_out)
            # call the backprop function of precursor
            self.precursor.bprop(l_rate, grad_wrt_in)

        # update the paramters based on grad_wrt_out
        grad_wrt_kernel = self.grad_kernel(grad_wrt_out)
        self.kernel = self.kernel - l_rate * grad_wrt_kernel


class Model(object):
    '''
    used to create complete models. Persistently stores varaibles
    values between updates. Exposes handles for training the model
    and for checking accuracy
    '''
    def __init__(self, input_dim, *layers):
        '''
        Builds the model based on the given layers
        Hosts Placeholders
        input_dim is the dimension of one batch
        '''
        # Data Placeholders
        self.x = PlaceHolder(input_dim)

        # Building the model based on inputs
        # last_layer is continuously updated
        # we loose references to some layers
        # but, that doesn't matter in out case
        last_layer = self.x
        for layer in layers:
            # build the model here
            if layer == "linear":
                last_layer = Linear(32, last_layer)
            if layer == "relu":
                last_layer = Relu(last_layer)
            if layer == "conv":
                last_layer = Conv(8, 3, 3, precursor=last_layer)
            if layer == "maxpool":
                last_layer = Maxpool(last_layer)
            if layer == "flatten":
                last_layer = Flatten(last_layer)
            # last_layer has been updated accordingly

        # all models end with linear -> softmaxXent
        # output of this linear layer should be 10
        self.pred_layer = Linear(10, last_layer)
        # linear layer is connected to softmaxXent layer
        self.loss_layer = SoftMaxXent(self.pred_layer)

    def train(self, l_rate, x_batch, y_batch):
        '''
        trains the models based on the given x_batch and y_batch data.
        updates the variables
        '''
        # update placeholders
        self.x.load(x_batch)

        # do fpass before training - bprop wont call fpass
        self.loss_layer.fpass(y_batch)

        # train using bprop
        self.loss_layer.bprop(l_rate, y_batch)
        pass

    def evaluate(self, x_batch, y_batch):
        '''
        predicts output for x_batch and uses y_batch for comparison.
        returns a single preiction accuracy float value
        '''
        # update placeholders
        self.x.load(x_batch)

        # finds the accuracy and returns it
        self.pred_layer.fpass()
        return np.mean(np.equal(np.argmax(self.pred_layer.output, 1),
                       np.argmax(y_batch, 1)).astype(float))
