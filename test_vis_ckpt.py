import tensorflow as tf
import numpy as np
reader = tf.train.NewCheckpointReader('/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt')
all_variables = reader.get_variable_to_shape_map()
w1 = reader.get_tensor("MobilenetV1/Conv2d_0/weights")
print(type(w1))
print(w1.shape)
#print(w1[0])

for v in all_variables:
    #print v
    if v.startswith('MobilenetV1/Conv2d_9_depthwise/depthwise_weights'):
        print v
