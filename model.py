"""
Define various models in the paper

Author: Huan Minh Luu
Magnetic Resonance Imaging Laboratory
KAIST
luuminhhuan@kaist.ac.kr
"""
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model

def create_qMTNet_fit_model():
    """
    Create qMTNet_fit model described in the paper

    Returns
    -------
    qMTNet_fit model

    """
    inp = Input((14,))

    out = inp
    for i in range(4):
        out = Dense(100,activation='relu')(out)
        out = BatchNormalization()(out)
		
    dropout = Dropout(0.2)(out)
    output = Dense(2,activation='sigmoid')(dropout)

    qMTNet_fit_model = Model(inputs=inp,outputs = output)

    return qMTNet_fit_model

def create_qMTNet_1_model():
    """
    Create qMTNet_1 model described in the paper

    Returns
    -------
    qMTNet_1 model

    """
    inp = Input((6,))

    out = inp
    for i in range(7):
        out = Dense(100,activation='relu')(out)
        out = BatchNormalization()(out)
		
    dropout = Dropout(0.4)(out)
    output = Dense(2,activation='sigmoid')(dropout)

    qMTNet_1_model = Model(inputs=inp,outputs = output)

    return qMTNet_1_model
    
    
def create_qMTNet_acq_model():
    """
    Create qMTNet_acq model described in the paper

    Returns
    -------
    qMTNet_acq model

    """
    
    inp = Input((128,128,4))

    def conv2d(inputs,filters,stride,kernel_size=(1,1),has_act=True):
        l = Conv2D(filters, strides=stride, kernel_size=kernel_size, padding='same')(inputs)
        if has_act:
            l = Activation('relu')(l)
        return l

    def module2(inputs):
        l1 = conv2d(inputs, 256, stride=1, kernel_size=(3,3))

        l2 = conv2d(l1, 128, stride = 1, kernel_size=(3,3))
        l2_ = concatenate([l1, l2], 3)

        l3 = conv2d(l2_, 32, stride = 1, kernel_size=(3,3))

        l5 = conv2d(l3, 8, 1, (3,3), has_act=False)
        return l5

    output = module2(inp)
    return Model(inputs=inp,outputs=output)
    
def make_model(model_type):
    """
    Create the appropriate model according to model_type

    Parameters
    ----------
    model_type : type of model to generate (qMTNet_1,qMTNet_fit,qMTNet_acq)
        DESCRIPTION.


    Returns
    -------
    A qMTNet model specified by model_type 

    """
    if model_type == 'qMTNet_fit':
        return create_qMTNet_fit_model()
    elif model_type == 'qMTNet_1':
        return create_qMTNet_1_model()
    elif model_type == 'qMTNet_acq':
        return create_qMTNet_acq_model()
    else:
        raise ValueError('Undefined model type: {}'.format(model_type))
