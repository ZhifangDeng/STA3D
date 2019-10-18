import io
import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.framework import ops

growth_rate = 32
growth_rate_factor = 4
reduction  = 0.5
keep_prob = 0.8

def dense_layer(layer, is_training):
    with tf.name_scope("dense_layer") :
	bottle_BN = tl.layers.BatchNormLayer(layer,
				             act = tf.nn.relu,
				             is_train = is_training,
				             name = 'bott_BN')
        in_features = int(bottle_BN.outputs.get_shape()[-1])
        bottle_Conv  = tl.layers.Conv3dLayer(bottle_BN,
				             act = tf.identity,
				             shape = [1,1,1,in_features,growth_rate*growth_rate_factor],
				             strides = [1,1,1,1,1],
				             padding = 'VALID',
				             name = 'bott_Conv')
        com_BN = tl.layers.BatchNormLayer(bottle_Conv,
			                          act = tf.nn.relu,
			                          is_train = is_training,
			                          name = 'com_BN')
	com_Conv = tl.layers.Conv3dLayer(com_BN,
			                         act = tf.identity,
			                         shape = [3,3,3,growth_rate*growth_rate_factor,growth_rate],
			                         strides = [1,1,1,1,1],
			                         padding = 'SAME',
			                         name = 'com_Conv')
	dense_layer = tl.layers.DropoutLayer(com_Conv,
			                     keep = keep_prob,
			                     is_train = is_training,
			                     name = 'dense_layer')
	concat_layer = tl.layers.ConcatLayer([layer, dense_layer],
			                            concat_dim = 4,
			                            name = 'concat_layer')
	return concat_layer

def se_dense_layer(layer, is_training):
    with tf.name_scope("se_dense_layer") :
	TCB_layer = CBAM_seq(layer)
        bottle_BN = tl.layers.BatchNormLayer(TCB_layer,
				             act = tf.nn.relu,
				             is_train = is_training,
				             name = 'bott_BN')
        in_features = int(bottle_BN.outputs.get_shape()[-1])
        bottle_Conv  = tl.layers.Conv3dLayer(bottle_BN,
				             act = tf.identity,
				             shape = [1,1,1,in_features,growth_rate*growth_rate_factor],
				             strides = [1,1,1,1,1],
				             padding = 'VALID',
				             name = 'bott_Conv')
        com_BN = tl.layers.BatchNormLayer(bottle_Conv,
			                          act = tf.nn.relu,
			                          is_train = is_training,
			                          name = 'com_BN')
	com_Conv = tl.layers.Conv3dLayer(com_BN,
			                         act = tf.identity,
			                         shape = [3,3,3,growth_rate*growth_rate_factor,growth_rate],
			                         strides = [1,1,1,1,1],
			                         padding = 'SAME',
			                         name = 'com_Conv')
        dense_layer = tl.layers.DropoutLayer(com_Conv,
			                     keep = keep_prob,
			                     is_train = is_training,
			                     name = 'dense_layer')
        
        
	concat_layer = tl.layers.ConcatLayer([layer, dense_layer],
			                            concat_dim = 4,
			                            name = 'concat_layer')
        
	return concat_layer


def dense_block(layer, is_training, layers_per_block):
	for layer_num in range(layers_per_block):
		with tf.variable_scope("layer_%d" % layer_num):
			layer = dense_layer(layer, is_training)
	return layer

def se_dense_block(layer, is_training, layers_per_block):
	for layer_num in range(layers_per_block):
		with tf.variable_scope("layer_%d" % layer_num):
			layer = se_dense_layer(layer, is_training)
	return layer

def transition_layer(layer, is_training):
	with tf.name_scope("transition_layer") :
		in_channel = int(layer.outputs.get_shape()[-1])
		out_channel = int(int(layer.outputs.get_shape()[-1]) * reduction)
		tran_BN = tl.layers.BatchNormLayer(layer,
			                               act = tf.nn.relu,
			                               is_train = is_training,
			                               name = 'tran_BN')
		tran_Conv = tl.layers.Conv3dLayer(tran_BN,
			                              act = tf.identity,
			                              shape = [1,1,1,in_channel,out_channel],
			                              strides = [1,1,1,1,1],
			                              padding = 'VALID',
			                              name = 'tran_Conv')
		tran_Pool = tl.layers.PoolLayer(tran_Conv,
			                            ksize = [1,2,2,2,1],
			                            strides = [1,2,2,2,1],
			                            padding = 'VALID',
			                            pool = tf.nn.avg_pool3d,
			                            name = 'tran_Pool')
		return tran_Pool

def tran_to_class(layer, is_training, num_classes):
	tran_to_class_BN = tl.layers.BatchNormLayer(layer,
		                                        act = tf.nn.relu,
		                                        is_train = is_training,
		                                        name = 'tran_to_class_BN')
	last_temporal = int(tran_to_class_BN.outputs.get_shape()[1])
	last_spatio = int(tran_to_class_BN.outputs.get_shape()[2])
	tran_to_class_Pool = tl.layers.PoolLayer(tran_to_class_BN,
		                                     ksize = [1,last_temporal,last_spatio,last_spatio,1],
		                                     strides = [1,last_temporal,last_spatio,last_spatio,1],
		                                     padding = 'VALID',
		                                     pool = tf.nn.avg_pool3d,
		                                     name = 'tran_to_class_Pool')
	flatten_last = tl.layers.FlattenLayer(tran_to_class_Pool,
		                                  name = 'flatten_last')
	classes = tl.layers.DropconnectDenseLayer(flatten_last,
		                                      keep = 0.8,
		                                      n_units = num_classes,
		                                      act = tf.identity,
		                                      name = 'Linear')
	return classes


def TTL1(layer, is_training):
	with tf.name_scope("TTL1_layer") :
		in_channel = int(layer.outputs.get_shape()[-1])
		out_channel = int(int(layer.outputs.get_shape()[-1]) * reduction)
		layer_BN = tl.layers.BatchNormLayer(layer,
			                                  act = tf.nn.relu,
			                                  is_train = is_training,
			                                  name = 'layer_BN')
		layer_F_Conv = tl.layers.Conv3dLayer(layer_BN,
			                                 act = tf.identity,
			                                 shape = [1,1,1,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'First_layer_Conv')
		layer_S_Conv = tl.layers.Conv3dLayer(layer_BN,
			                                 act = tf.identity,
			                                 shape = [3,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'Second_layer_Conv')
		layer_T_Conv = tl.layers.Conv3dLayer(layer_BN,
			                                 act = tf.identity,
			                                 shape = [6,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'Third_layer_Conv')
		TTL1_Concat = tl.layers.ConcatLayer([layer_F_Conv,layer_S_Conv,layer_T_Conv],
			                               concat_dim = 4,
			                               name = 'TTL1_Concat_layer')
		TTL1_Pool = tl.layers.PoolLayer(TTL1_Concat,
			                            ksize = [1,2,2,2,1],
			                            strides = [1,2,2,2,1],
			                            padding = 'VALID',
			                            pool = tf.nn.avg_pool3d,
			                            name = 'TTL1_Pool_layer')
		return TTL1_Pool

def TTL2(layer, is_training):
	with tf.name_scope("TTL2_layer") :
		in_channel = int(layer.outputs.get_shape()[-1])
		out_channel = int(int(layer.outputs.get_shape()[-1]) * reduction)
		layer_BN = tl.layers.BatchNormLayer(layer,
			                                  act = tf.nn.relu,
			                                  is_train = is_training,
			                                  name = 'layer_BN')
		layer_F_Conv = tl.layers.Conv3dLayer(layer_BN,
			                                 act = tf.identity,
			                                 shape = [1,1,1,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'First_layer_Conv')
		layer_S_Conv = tl.layers.Conv3dLayer(layer_BN,
			                                 act = tf.identity,
			                                 shape = [3,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'Second_layer_Conv')
		layer_T_Conv = tl.layers.Conv3dLayer(layer_BN,
			                                 act = tf.identity,
			                                 shape = [4,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'Third_layer_Conv')
		TTL2_Concat = tl.layers.ConcatLayer([layer_F_Conv,layer_S_Conv,layer_T_Conv],
			                               concat_dim = 4,
			                               name = 'TTL2_Concat_layer')
		TTL2_Pool = tl.layers.PoolLayer(TTL2_Concat,
			                            ksize = [1,2,2,2,1],
			                            strides = [1,2,2,2,1],
			                            padding = 'VALID',
			                            pool = tf.nn.avg_pool3d,
			                            name = 'TTL2_Pool_layer')
		return TTL2_Pool

def RTTL1(layer, is_training):
	with tf.name_scope("RTTL1_layer") :
		in_channel = int(layer.outputs.get_shape()[-1])
		out_channel = int(int(layer.outputs.get_shape()[-1]) * reduction)
		layer_F_Conv = tl.layers.Conv3dLayer(layer,
			                                 act = tf.identity,
			                                 shape = [3,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'First_layer_Conv')
		layer_F_BN = tl.layers.BatchNormLayer(layer_F_Conv,
			                                  act = tf.nn.relu,
			                                  is_train = is_training,
			                                  name = 'First_layer_BN')
		layer_S_Conv = tl.layers.Conv3dLayer(layer,
			                                 act = tf.identity,
			                                 shape = [6,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'Second_layer_Conv')
		layer_S_BN = tl.layers.BatchNormLayer(layer_S_Conv,
			                                  act = tf.nn.relu,
			                                  is_train = is_training,
			                                  name = 'Second_layer_BN')
		RTTL1_Concat = tl.layers.ConcatLayer([layer_F_BN,layer_S_BN],
			                                 concat_dim = 4,
			                                 name = 'RTTL1_Concat_layer')
		RTTL1_Pool = tl.layers.PoolLayer(RTTL1_Concat,
			                             ksize = [1,2,2,2,1],
			                             strides = [1,2,2,2,1],
			                             padding = 'VALID',
			                             pool = tf.nn.avg_pool3d,
			                             name = 'RTTL1_Pool_layer')
		return RTTL1_Pool

def RTTL2(layer, is_training):
	with tf.name_scope("RTTL2_layer") :
		in_channel = int(layer.outputs.get_shape()[-1])
		out_channel = int(int(layer.outputs.get_shape()[-1]) * reduction)
		layer_F_Conv = tl.layers.Conv3dLayer(layer,
			                                 act = tf.identity,
			                                 shape = [3,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'First_layer_Conv')
		layer_F_BN = tl.layers.BatchNormLayer(layer_F_Conv,
			                                  act = tf.nn.relu,
			                                  is_train = is_training,
			                                  name = 'First_layer_BN')
		layer_S_Conv = tl.layers.Conv3dLayer(layer,
			                                 act = tf.identity,
			                                 shape = [6,3,3,in_channel,out_channel],
			                                 strides = [1,1,1,1,1],
			                                 padding = 'SAME',
			                                 name = 'Second_layer_Conv')
		layer_S_BN = tl.layers.BatchNormLayer(layer_S_Conv,
			                                  act = tf.nn.relu,
			                                  is_train = is_training,
			                                  name = 'Second_layer_BN')
		RTTL2_Concat = tl.layers.ConcatLayer([layer_F_BN,layer_S_BN],
			                                 concat_dim = 4,
			                                 name = 'RTTL2_Concat_layer')
		RTTL2_Pool = tl.layers.PoolLayer(RTTL2_Concat,
			                             ksize = [1,2,2,2,1],
			                             strides = [1,2,2,2,1],
			                             padding = 'VALID',
			                             pool = tf.nn.avg_pool3d,
			                             name = 'RTTL2_Pool_layer')
		return RTTL2_Pool

def TCB_avg(layer, ratio):
	with tf.name_scope("TCB_avg") :
		TCB_temporal = int(layer.outputs.get_shape()[1])
		TCB_spatio = int(layer.outputs.get_shape()[2])
		out_channel = int(layer.outputs.get_shape()[-1])
		TCB_avg_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               strides = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.avg_pool3d,
			                               name = 'TCB_avg_Pool_layer')
		TCB_avg_Flatten = tl.layers.FlattenLayer(TCB_avg_Pool,
			                                     name = 'TCB_avg_Flatten_layer')
		TCB_avg_F_Dense = tl.layers.DenseLayer(TCB_avg_Flatten,
			                                   n_units = out_channel//ratio,
			                                   act = tf.nn.relu,
			                                   name = 'TCB_avg_First_Dense_layer')
		TCB_avg_S_Dense = tl.layers.DenseLayer(TCB_avg_F_Dense,
			                                   n_units = out_channel,
			                                   act = tf.nn.sigmoid,
			                                   name = 'TCB_avg_Second_Dense_layer')
		TCB_avg_Rescale = tl.layers.ReshapeLayer(TCB_avg_S_Dense,
			                                     shape = [-1,1,1,1,out_channel],
			                                     name = 'TCB_avg_Rescale_layer')
		net = tl.layers.ElementwiseLayer([layer, TCB_avg_Rescale], combine_fn=tf.multiply, name='multiply')
                #layer.outputs = layer.outputs * TCB_avg_Rescale.outputs
		return net

def TCB_max(layer, ratio):
	with tf.name_scope("TCB_max") :
		TCB_temporal = int(layer.outputs.get_shape()[1])
		TCB_spatio = int(layer.outputs.get_shape()[2])
		out_channel = int(layer.outputs.get_shape()[-1])
		TCB_max_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               strides = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.max_pool3d,
			                               name = 'TCB_max_Pool_layer')
		TCB_max_Flatten = tl.layers.FlattenLayer(TCB_max_Pool,
			                                     name = 'TCB_max_Flatten_layer')
		TCB_max_F_Dense = tl.layers.DenseLayer(TCB_max_Flatten,
			                                   n_units = out_channel//ratio,
			                                   act = tf.nn.relu,
			                                   name = 'TCB_max_First_Dense_layer')
		TCB_max_S_Dense = tl.layers.DenseLayer(TCB_max_F_Dense,
			                                   n_units = out_channel,
			                                   act = tf.nn.sigmoid,
			                                   name = 'TCB_max_Second_Dense_layer')
		TCB_max_Rescale = tl.layers.ReshapeLayer(TCB_max_S_Dense,
			                                     shape = [-1,1,1,1,out_channel],
			                                     name = 'TCB_max_Rescale_layer')
		layer.outputs = layer.outputs * TCB_max_Rescale.outputs
		return layer

def TCB(layer, ratio):
	with tf.name_scope("TCB") :
		TCB_temporal = int(layer.outputs.get_shape()[1])
		TCB_spatio = int(layer.outputs.get_shape()[2])
		out_channel = int(layer.outputs.get_shape()[-1])
		TCB_avg_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               strides = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.avg_pool3d,
			                               name = 'TCB_avg_Pool_layer')
		TCB_avg_Flatten = tl.layers.FlattenLayer(TCB_avg_Pool,
			                                     name = 'TCB_avg_Flatten_layer')
		TCB_avg_F_Dense = tl.layers.DenseLayer(TCB_avg_Flatten,
			                                   n_units = out_channel//ratio,
			                                   act = tf.nn.relu,
			                                   name = 'TCB_avg_First_Dense_layer')
		TCB_avg_S_Dense = tl.layers.DenseLayer(TCB_avg_F_Dense,
			                                   n_units = out_channel,
			                                   act = tf.identity,
			                                   name = 'TCB_avg_Second_Dense_layer')
		TCB_max_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               strides = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.max_pool3d,
			                               name = 'TCB_max_Pool_layer')
		TCB_max_Flatten = tl.layers.FlattenLayer(TCB_max_Pool,
			                                     name = 'TCB_max_Flatten_layer')
		TCB_max_F_Dense = tl.layers.DenseLayer(TCB_max_Flatten,
			                                   n_units = out_channel//ratio,
			                                   act = tf.nn.relu,
			                                   name = 'TCB_max_First_Dense_layer')
		TCB_max_S_Dense = tl.layers.DenseLayer(TCB_max_F_Dense,
			                                   n_units = out_channel,
			                                   act = tf.identity,
			                                   name = 'TCB_max_Second_Dense_layer')
		TCB_Add_Layer = tl.layers.ElementwiseLayer([TCB_avg_S_Dense,TCB_max_S_Dense],
			                                       act = tf.nn.sigmoid,
			                                       combine_fn = tf.add,
			                                       name = 'TCB_Add_Layer')
		TCB_Rescale = tl.layers.ReshapeLayer(TCB_Add_Layer,
			                                 shape = [-1,1,1,1,out_channel],
			                                 name = 'TCB_Rescale_layer')
		net = tl.layers.ElementwiseLayer([layer, TCB_Rescale], combine_fn=tf.multiply, name='multiply')
		return net

def MLP(layer, ratio, reuse):
	with tf.variable_scope('MLP', reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		MLP_out_channel = int(layer.outputs.get_shape()[-1])
		MLP_F_layer = tl.layers.DenseLayer(layer,
			                               n_units = MLP_out_channel//ratio,
			                               act = tf.nn.relu,
			                               name = 'MLP_First_layer')
		MLP_S_layer = tl.layers.DenseLayer(MLP_F_layer,
			                               n_units = MLP_out_channel,
			                               act = tf.identity,
			                               name = 'MLP_Second_layer')
	return MLP_S_layer

def TCB_Reuse(layer, ratio):
	with tf.name_scope("TCB_Reuse") :
		TCB_temporal = int(layer.outputs.get_shape()[1])
		TCB_spatio = int(layer.outputs.get_shape()[2])
		TCB_out_channel = int(layer.outputs.get_shape()[-1])
		TCB_avg_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               strides = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.avg_pool3d,
			                               name = 'TCB_avg_Pool_layer')
		TCB_avg_Flatten = tl.layers.FlattenLayer(TCB_avg_Pool,
			                                     name = 'TCB_avg_Flatten_layer')
		TCB_avg_MLP = MLP(TCB_avg_Flatten,
			              ratio = ratio,
			              reuse = False)
		TCB_max_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               strides = [1,TCB_temporal,TCB_spatio,TCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.max_pool3d,
			                               name = 'TCB_max_Pool_layer')
		TCB_max_Flatten = tl.layers.FlattenLayer(TCB_max_Pool,
			                                     name = 'TCB_max_Flatten_layer')
		TCB_max_MLP = MLP(TCB_max_Flatten,
			              ratio = ratio,
			              reuse = True)
		TCB_Add_Layer = tl.layers.ElementwiseLayer(layer = [TCB_avg_MLP,TCB_max_MLP],
			                                       act = tf.nn.sigmoid,
			                                       combine_fn = tf.add,
			                                       name = 'TCB_Add_Layer')
		TCB_Rescale = tl.layers.ReshapeLayer(TCB_Add_Layer,
			                                 shape = [-1,1,1,1,TCB_out_channel],
			                                 name = 'TCB_Rescale_layer')
		layer.outputs = TCB_Rescale.outputs * layer.outputs
		return layer

def SCB(layer, ratio):
	with tf.name_scope("SCB") :
		SCB_temporal = int(layer.outputs.get_shape()[1])
		SCB_spatio = int(layer.outputs.get_shape()[2])
		SCB_out_channel = int(layer.outputs.get_shape()[-1])
		SCB_avg_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,1,SCB_spatio,SCB_spatio,1],
			                               strides = [1,1,SCB_spatio,SCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.avg_pool3d,
			                               name = 'SCB_avg_Pool_layer')
		SCB_avg_Flatten = tl.layers.FlattenLayer(SCB_avg_Pool,
			                                     name = 'SCB_avg_Flatten_layer')
		SCB_avg_F_Dense = tl.layers.DenseLayer(SCB_avg_Flatten,
			                                   n_units = SCB_out_channel//ratio,
			                                   act = tf.nn.relu,
			                                   name = 'SCB_avg_First_Dense_layer')
		SCB_avg_S_Dense = tl.layers.DenseLayer(SCB_avg_F_Dense,
			                                   n_units = SCB_out_channel,
			                                   act = tf.nn.sigmoid,
			                                   name = 'SCB_avg_Second_Dense_layer')
		SCB_avg_Rescale = tl.layers.ReshapeLayer(SCB_avg_S_Dense,
			                                     shape = [-1,SCB_temporal,1,1,SCB_out_channel],
			                                     name = 'SCB_avg_Rescale_layer')
		layer.outputs = layer.outputs * SCB_avg_Rescale.outputs
		return layer

def RSCB(layer, ratio):
	with tf.name_scope("RSCB") :
		RSCB_temporal = int(layer.outputs.get_shape()[1])
		RSCB_spatio = int(layer.outputs.get_shape()[2])
		RSCB_out_channel = int(layer.outputs.get_shape()[-1])
		RSCB_avg_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,1,RSCB_spatio,RSCB_spatio,1],
			                               strides = [1,1,RSCB_spatio,RSCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.avg_pool3d,
			                               name = 'RSCB_avg_Pool_layer')
		RSCB_avg_Flatten = tl.layers.FlattenLayer(RSCB_avg_Pool,
			                                     name = 'RSCB_avg_Flatten_layer')
		RSCB_avg_MLP = MLP(RSCB_avg_Flatten,
			              ratio = ratio,
			              reuse = False)
		RSCB_max_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,1,RSCB_spatio,RSCB_spatio,1],
			                               strides = [1,1,RSCB_spatio,RSCB_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.max_pool3d,
			                               name = 'RSCB_max_Pool_layer')
		RSCB_max_Flatten = tl.layers.FlattenLayer(RSCB_max_Pool,
			                                     name = 'RSCB_max_Flatten_layer')
		RSCB_max_MLP = MLP(RSCB_max_Flatten,
			              ratio = ratio,
			              reuse = True)
		RSCB_Add_Layer = tl.layers.ElementwiseLayer(layer = [RSCB_avg_MLP,RSCB_max_MLP],
			                                       act = tf.nn.sigmoid,
			                                       combine_fn = tf.add,
			                                       name = 'RSCB_Add_Layer')
		RSCB_Rescale = tl.layers.ReshapeLayer(RSCB_Add_Layer,
			                                 shape = [-1,RSCB_temporal,1,1,RSCB_out_channel],
			                                 name = 'RSCB_Rescale_layer')
		layer.outputs = RSCB_Rescale.outputs * layer.outputs
		return layer

def SSCA(layer):
	with tf.name_scope("SSCA") :
		SSCA_temporal = int(layer.outputs.get_shape()[1])
		SSCA_spatio = int(layer.outputs.get_shape()[2])
		SSCA_out_channel = int(layer.outputs.get_shape()[-1])
		SSCA_avg_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,1,SSCA_spatio,SSCA_spatio,1],
			                               strides = [1,1,SSCA_spatio,SSCA_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.avg_pool3d,
			                               name = 'SSCA_avg_Pool_layer')
		SSCA_max_Pool = tl.layers.PoolLayer(layer,
			                               ksize = [1,1,SSCA_spatio,SSCA_spatio,1],
			                               strides = [1,1,SSCA_spatio,SSCA_spatio,1],
			                               padding = 'VALID',
			                               pool = tf.nn.max_pool3d,
			                               name = 'SSCA_max_Pool_layer')

		SSCA_Pool = tl.layers.ConcatLayer([SSCA_avg_Pool,SSCA_max_Pool],
			                            concat_dim = 4,
			                            name = 'SCA_concat_pool')
		in_channel = int(SSCA_Pool.outputs.get_shape()[-1])
		out_channel = int(layer.outputs.get_shape()[-1])
		SSCA_Conv = tl.layers.Conv3dLayer(SSCA_Pool,
			                            act = tf.nn.sigmoid,
			                            shape = [1,7,7,in_channel,out_channel],
			                            strides = [1,1,1,1,1],
			                            padding = 'SAME',
			                            name = 'SA_Conv_layer')
		layer.outputs = layer,outputs * SSCA_Conv.outputs
		return layer

def SCA(layer):
	with tf.name_scope("SCA") :
		SCA_temporal = int(layer.outputs.get_shape()[1])
		SCA_spatio = int(layer.outputs.get_shape()[2])
		SCA_out_channel = int(layer.outputs.get_shape()[-1])
		SCA_avg_Pool = tl.layers.ChanPoolLayer(layer,
			                                  pool = tf.reduce_mean,
			                                  name = 'SCA_mean_pool')
		SCA_max_Pool = tl.layers.ChanPoolLayer(layer,
			                                  pool = tf.reduce_max,
			                                  name = 'SCA_max_pool')
		SCA_Pool = tl.layers.ConcatLayer([SCA_avg_Pool,SCA_max_Pool],
			                            concat_dim = 4,
			                            name = 'SCA_concat_pool')
		in_channel = int(SCA_Pool.outputs.get_shape()[-1])
		out_channel = int(layer.outputs.get_shape()[-1])
		SCA_Conv = tl.layers.Conv3dLayer(SCA_Pool,
			                            act = tf.nn.sigmoid,
			                            shape = [1,7,7,in_channel,out_channel],
			                            strides = [1,1,1,1,1],
			                            padding = 'SAME',
			                            name = 'SA_Conv_layer')
		net = tl.layers.ElementwiseLayer([layer, SCA_Conv], combine_fn=tf.multiply, name='multiply')
		return net

def CBAM_seq(layer):
	with tf.name_scope('CBAM_seq'):
		TCB_layer = TCB(layer, 4)
		CBAM_seq_layer = SCA(TCB_layer)
		return CBAM_seq_layer

def CBAM_para1(layer):
	with tf.name_scope('CBAM_para1'):
		TCB_layer = TCB_Reuse(layer, 4)
		SCA_layer = SCA(layer, 4)
		CBAM_layer = tl.layers.ElementwiseLayer(layer = [TCB_layer,SCA_layer],
				                                combine_fn = tf.add,
				                                name = 'CBAM_layer')
		CBAM_layer.outputs = CBAM_layer.outputs//2
		return CBAM_layer



def STC(layer):
	with tf.name_scope('STC'):
		TCB_layer = TCB_avg(layer, 4)
		SCB_layer = SCB(layer,4)
		STCB_layer = tl.layers.ElementwiseLayer(layer = [TCB_layer,SCB_layer],
				                                combine_fn = tf.add,
				                                name = 'STCB_layer')
		STCB_layer.outputs = STCB_layer.outputs//2
		return STCB_layer

def T3D(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('T3D',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,1,1,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
		with tf.variable_scope('transition_1'):
			tran_1 = TTL1(block_1, is_training)

		with tf.variable_scope('block_2'):
			block_2 = dense_block(tran_1, is_training, 3)
		with tf.variable_scope('transition_2'):
			tran_2 = TTL2(block_2, is_training)

		with tf.variable_scope('block_3'):
			block_3 = dense_block(tran_2, is_training, 6)
		with tf.variable_scope('transition_3'):
			tran_3 = TTL2(block_3, is_training)

		with tf.variable_scope('block_4'):
			block_4 = dense_block(tran_3, is_training, 4)
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph

def RT3D(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('RT3D',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,2,2,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
		with tf.variable_scope('transition_1'):
			tran_1 = RTTL1(block_1, is_training)

		with tf.variable_scope('block_2'):
			block_2 = dense_block(tran_1, is_training, 3)
		with tf.variable_scope('transition_2'):
			tran_2 = RTTL2(block_2, is_training)

		with tf.variable_scope('block_3'):
			block_3 = dense_block(tran_2, is_training, 6)
		with tf.variable_scope('transition_3'):
			tran_3 = RTTL2(block_3, is_training)

		with tf.variable_scope('block_4'):
			block_4 = dense_block(tran_3, is_training, 4)
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph

def RTA3D(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('RT3D',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
                                                               strides = [1,1,2,2,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = se_dense_block(initial_pool, is_training, 2)
		with tf.variable_scope('transition_1'):
			tran_1 = RTTL1(block_1, is_training)

		with tf.variable_scope('block_2'):
			block_2 = se_dense_block(tran_1, is_training, 6)
		with tf.variable_scope('transition_2'):
			tran_2 = RTTL2(block_2, is_training)

		with tf.variable_scope('block_3'):
			block_3 = se_dense_block(tran_2, is_training, 12)
		with tf.variable_scope('transition_3'):
			tran_3 = RTTL2(block_3, is_training)

		with tf.variable_scope('block_4'):
			block_4 = se_dense_block(tran_3, is_training, 8)
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return tran_1,tran_2,tran_3,class_graph


def DenseNet3D(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('DenseNet3D',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,2,2,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
		with tf.variable_scope('transition_1'):
			tran_1 = transition_layer(block_1, is_training)

		with tf.variable_scope('block_2'):
			block_2 = dense_block(tran_1, is_training, 3)
		with tf.variable_scope('transition_2'):
			tran_2 = transition_layer(block_2, is_training)

		with tf.variable_scope('block_3'):
			block_3 = dense_block(tran_2, is_training, 6)
		with tf.variable_scope('transition_3'):
			tran_3 = transition_layer(block_3, is_training)

		with tf.variable_scope('block_4'):
			block_4 = dense_block(tran_3, is_training, 4)
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph

def SE_DenseNet3D(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('SE_DenseNet3D',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,1,1,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
			
		with tf.variable_scope('block_1'):
			block_1 = se_dense_block(initial_pool, is_training, 2)
			
		with tf.variable_scope('transition_1'):
			tran_1 = transition_layer(block_1, is_training)
            
		with tf.variable_scope('block_2'):
			block_2 = se_dense_block(tran_1, is_training, 3)
			
		with tf.variable_scope('transition_2'):
			tran_2 = transition_layer(block_2, is_training)
            
		with tf.variable_scope('block_3'):
			block_3 = se_dense_block(tran_2, is_training, 6)
			
		with tf.variable_scope('transition_3'):
			tran_3 = transition_layer(block_3, is_training)
			
		with tf.variable_scope('block_4'):
			block_4 = se_dense_block(tran_3, is_training, 4)
			
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph



def DenseNet3D_avg(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('DenseNet3D_avg',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,1,1,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
                        
		with tf.variable_scope('transition_1'):
			tran_1 = transition_layer(block_1, is_training)
                        TCB_avg_tran_1 = TCB_avg(tran_1, 4)

		with tf.variable_scope('block_2'):
			block_2 = dense_block(TCB_avg_tran_1, is_training, 3)
                        
		with tf.variable_scope('transition_2'):
			tran_2 = transition_layer(block_2, is_training)
                        TCB_avg_tran_2 = TCB_avg(tran_2, 4)

		with tf.variable_scope('block_3'):
			block_3 = dense_block(TCB_avg_tran_2, is_training, 6)
                        
		with tf.variable_scope('transition_3'):
			tran_3 = transition_layer(block_3, is_training)
                        TCB_avg_tran_3 = TCB_avg(tran_3, 4)

		with tf.variable_scope('block_4'):
			block_4 = dense_block(TCB_avg_tran_3, is_training, 4)
                       
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph
	

def DenseNet3D_STC(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('DenseNet3D_STC',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,1,1,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
                        
		with tf.variable_scope('transition_1'):
			tran_1 = transition_layer(block_1, is_training)
                        STC_1 = STC(tran_1)

		with tf.variable_scope('block_2'):
			block_2 = dense_block(STC_1, is_training, 3)
                        
		with tf.variable_scope('transition_2'):
			tran_2 = transition_layer(block_2, is_training)
                        STC_2 = STC(tran_2)

		with tf.variable_scope('block_3'):
			block_3 = dense_block(STC_2, is_training, 6)
                        
		with tf.variable_scope('transition_3'):
			tran_3 = transition_layer(block_3, is_training)
                        STC_3 = STC(tran_3)

		with tf.variable_scope('block_4'):
			block_4 = dense_block(STC_3, is_training, 4)
                        
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph


def DenseNet3D_SCA(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('DenseNet3D_avg',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,1,1,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
                        
		with tf.variable_scope('transition_1'):
			tran_1 = transition_layer(block_1, is_training)
                        SCA_tran_1 = SCA(tran_1)

		with tf.variable_scope('block_2'):
			block_2 = dense_block(SCA_tran_1, is_training, 3)
                       
		with tf.variable_scope('transition_2'):
			tran_2 = transition_layer(block_2, is_training)
                        SCA_tran_2 = SCA(tran_2)

		with tf.variable_scope('block_3'):
			block_3 = dense_block(SCA_tran_2, is_training, 6)
                        
		with tf.variable_scope('transition_3'):
			tran_3 = transition_layer(block_3, is_training)
                        SCA_tran_3 = SCA(tran_3)
                        
		with tf.variable_scope('block_4'):
			block_4 = dense_block(SCA_tran_3, is_training, 4)
                        
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph

def DenseNet3D_TSCA(inputs, num_classes, reuse, is_training):
	with tf.variable_scope('DenseNet3D_TSCA',reuse = reuse):
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs, name = 'input_layer')
		with tf.variable_scope('Initial_Conv'):
			initial_conv =tl.layers.Conv3dLayer(input_layer,
				                                act = tf.identity,
				                                shape = [3,7,7,3,2*growth_rate],
				                                strides = [1,2,2,2,1],
				                                padding = 'SAME',
				                                name = 'initial_conv_layer')
			initial_bn = tl.layers.BatchNormLayer(initial_conv,
				                                  act = tf.nn.relu,
				                                  is_train = is_training,
				                                  name = 'initial_bn_layer')
			initial_pool = tl.layers.PoolLayer(initial_bn,
				                               ksize = [1,3,3,3,1],
				                               strides = [1,1,1,1,1],
				                               padding = 'SAME',
				                               pool = tf.nn.max_pool3d,
				                               name = 'initial_pool_layer')
		with tf.variable_scope('block_1'):
			block_1 = dense_block(initial_pool, is_training, 2)
                        
		with tf.variable_scope('transition_1'):
			tran_1 = transition_layer(block_1, is_training)
                        TCB_avg_tran_1 = TCB_avg(tran_1, 4)
                        SCA_tran_1 = SCA(TCB_avg_tran_1)
		with tf.variable_scope('block_2'):
			block_2 = dense_block(SCA_tran_1, is_training, 3)
                        
		with tf.variable_scope('transition_2'):
			tran_2 = transition_layer(block_2, is_training)
                        TCB_avg_tran_2 = TCB_avg(tran_2, 4)
                        SCA_tran_2 = SCA(TCB_avg_tran_2)
		with tf.variable_scope('block_3'):
			block_3 = dense_block(SCA_tran_2, is_training, 6)
                        
		with tf.variable_scope('transition_3'):
			tran_3 = transition_layer(block_3, is_training)
                        TCB_avg_tran_3 = TCB_avg(tran_3, 4)
                        SCA_tran_3 = SCA(TCB_avg_tran_3)
		with tf.variable_scope('block_4'):
			block_4 = dense_block(SCA_tran_3, is_training, 4)
                        
		with tf.variable_scope('transition_to_class'):
			class_graph = tran_to_class(block_4, is_training, num_classes)
	return class_graph


