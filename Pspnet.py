from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
# Pspnet
from keras.models import *
from keras.layers import *

from keras_segmentation.models.model_utils import get_segmentation_model

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def resize_image( inp ,  s , data_format ):
	import tensorflow as tf
    

	return Lambda( 
		lambda x: tf.image.resize(
			x , ( K.int_shape(x)[1]*s[0] ,K.int_shape(x)[2]*s[1] ))  
		)( inp )
def pool_block( feats , pool_factor ):


	if IMAGE_ORDERING == 'channels_first':
		h = K.int_shape( feats )[2]
		w = K.int_shape( feats )[3]
	elif IMAGE_ORDERING == 'channels_last':
		h = K.int_shape( feats )[1]
		w = K.int_shape( feats )[2]

	# strides = [18,18],[9,9],[6,6],[3,3]
	pool_size = strides = [int(np.round( float(h) /  pool_factor)), int(np.round(  float(w )/  pool_factor))]
 
	# 进行不同程度的平均
	x = AveragePooling2D(pool_size , data_format=IMAGE_ORDERING , strides=strides, padding='same')( feats )
	
	# 进行卷积
	x = Conv2D(32, (1 ,1 ), data_format=IMAGE_ORDERING , padding='same' , use_bias=False, kernel_regularizer=keras.regularizers.l2(0.00001) )( x )
	x = BatchNormalization()(x)
	x = Activation('relu' )(x)

	x = resize_image( x , strides , data_format=IMAGE_ORDERING ) 

	return x
def _pspnet( n_classes , encoder ,  input_height=384, input_width=576  ):

	assert input_height%192 == 0
	assert input_width%192 == 0

	img_input , levels = encoder( input_height=input_height,input_width=input_width)
	[f1 , f2 , f3 , f4 , f5 ] = levels 

	o = f5
	# 对f5进行不同程度的池化
	pool_factors = [ 1,2,3,6]
	pool_outs = [o ]
	for p in pool_factors:
		pooled = pool_block(o,p)
		pool_outs.append( pooled )
	# 连接
	o = Concatenate( axis=MERGE_AXIS)(pool_outs )
	o = ( Conv2D(32, (1, 1), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=keras.regularizers.l2(0.001)))(o)
	o = ( BatchNormalization())(o)
	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)
	o = Concatenate( axis=MERGE_AXIS)([o,f4])

	o = ( Conv2D(32, (1, 1), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=keras.regularizers.l2(0.001)))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu' )(o)
	pool_outs = [o ]
	# 对f4进行不同程度的池化
	for p in pool_factors:
		pooled = pool_block(o,p)
		pool_outs.append( pooled )
	# 连接
	o = Concatenate( axis=MERGE_AXIS)(pool_outs )
	o = ( Conv2D(32, (1, 1), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=keras.regularizers.l2(0.001)))(o)
	o = ( BatchNormalization())(o)
	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)

	o = Concatenate( axis=MERGE_AXIS)([o,f3])
	o = ( Conv2D(32, (1, 1), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=keras.regularizers.l2(0.001)))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu' )(o)
	pool_outs = [o ]
	# 对f3进行不同程度的池化
	for p in pool_factors:
		pooled = pool_block(o,p)
		pool_outs.append( pooled )
	# 连接
	o = Concatenate( axis=MERGE_AXIS)(pool_outs )

	# 卷积
	o = Conv2D(32, (1,1), data_format=IMAGE_ORDERING, use_bias=False ,kernel_regularizer=keras.regularizers.l2(0.001))(o)
	o = BatchNormalization()(o)
	o = Activation('relu' )(o)

	# 此时输出为[144,144,nclasses]
	o = Conv2D( n_classes,(3,3),data_format=IMAGE_ORDERING, padding='same', kernel_regularizer=keras.regularizers.l2(0.001) )(o)
	o = resize_image(o,(2,2),data_format=IMAGE_ORDERING)
	o = Reshape((-1,n_classes))(o)
	o = Softmax()(o)
	model = Model(img_input,o)
	return model



def mobilenet_pspnet( n_classes ,  input_height=224, input_width=224 ):

	model =  _pspnet( n_classes , get_mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_pspnet"
	return model