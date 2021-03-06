import os
import sys
sys.path.append('./lib/') ## !!!
sys.path.append('./lib/slim/')

import functools
import numpy as np
import tensorflow as tf
from core import trainer_test, trainer_seq, input_reader
from core.model_builder import build_man_model
from google.protobuf import text_format

from lib.object_detection.protos import pipeline_pb2
import functools
from lib.object_detection.builders import preprocessor_builder, optimizer_builder
from lib.object_detection.utils import variables_helper
from lib.object_detection.core import standard_fields as fields
from lib.object_detection.core import preprocessor, batcher
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os
from core.preprocessor import  preprocess

from train.data_loader import otb_dataset,otb_collate
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(0)
#print np.random.rand(2, 2, 300, 300, 3)
#print '\n','\n'
#print np.random.rand(2, 2, 300, 300, 3)
#exit()

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags

flags.DEFINE_string('train_dir', 'model/ssd_mobilenet_video1/',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('pipeline_config_path', '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/ssd_mobilenet_video.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_string('image_root', '/media/2TB/Research/DataSet/ILSVRC2015/Data/VID/train/',
                    'Root path to input images')

FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_configs_from_pipeline_file():
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model.ssd
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader

  return model_config, train_config, input_config

def main(_):
    global_step = tf.train.create_global_step()

    model_config, train_config, input_config = get_configs_from_pipeline_file()
    model_fn = functools.partial(
        build_man_model,
        model_config=model_config,
        is_training=True)

    batchsize = 3
    num_seq = 2
    search_region = tf.placeholder(tf.float32, [batchsize, num_seq, 300, 300, 3])
    template = tf.placeholder(tf.float32, [batchsize, 1, 128, 128, 3])  ## each sequene only has one template
    groundtruth_boxes = tf.placeholder(tf.float32, [batchsize, num_seq, 4])
    groundtruth_classes = tf.placeholder(tf.int32, [batchsize, num_seq, 1])

    otb_data = otb_dataset('/home/yuzhe/Downloads/part_vot_seq/')
    otb_loader = torch.utils.data.DataLoader(otb_data,
                                             batch_size=train_config.batch_size, shuffle=True, num_workers=1,
                                             collate_fn=otb_collate, drop_last=True)

    detection_model = model_fn()

    detection_model.provide_groundtruth(groundtruth_boxes,
                                        groundtruth_classes)
    prediction = detection_model.predict(template, search_region)

    losses_dict = detection_model.loss(prediction)

    for loss_tensor in losses_dict.values():
        tf.losses.add_loss(loss_tensor)
    total_loss = tf.losses.get_total_loss()

    training_optimizer = optimizer_builder.build(train_config.optimizer, set())
    train_op = training_optimizer.minimize(total_loss, global_step=global_step)

    """
    ## for test feature_extract
    feat_extra = detection_model.extract_feature(search_region)
    """

    #init_restore_fn = None
    var_map = detection_model.restore_map(
        from_detection_checkpoint=train_config.from_detection_checkpoint)
    var_map_init = detection_model.restore_init_map(
        from_detection_checkpoint=train_config.from_detection_checkpoint)

    available_var_map = (variables_helper.
        get_variables_available_in_checkpoint(
        var_map, train_config.fine_tune_checkpoint))
    available_var_map_init = (variables_helper.
        get_variables_available_in_checkpoint(
        var_map_init, train_config.fine_tune_checkpoint))

    #for k in tf.all_variables():
    #    print k.op.name

    """
    print '\n','\n','\n'
    for key,value in available_var_map.items():
       print key
       print value
    print '\n','\n','\n'
    """
    feat_extract_saver = tf.train.Saver(available_var_map)
    init_saver = tf.train.Saver(available_var_map_init)






    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        feat_extract_saver.restore(sess, train_config.fine_tune_checkpoint)
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
        """
        v = None
        for variable in tf.all_variables():
            print variable.op.name
            
            if variable.op.name.startswith('InitFeatureExtractor/MobilenetV1/Conv2d_0/weights'): #FeatureExtractor/MobilenetV1/Conv2d_0/weights
                print '\n','\n','\n',
                print variable
                v = variable
                print '\n', '\n', '\n'

        w1 = sess.run(v)
        print w1.shape
        print w1

        #feat = sess.run(feat_extra, feed_dict={search_region: np.ones((batchsize, num_seq, 300, 300, 3),dtype=np.float32)})
        #print feat['feature_maps0']
        #sess.run(restore_op)

        """
        for i in xrange(100):
            print 'iter {}'.format(i)
            sess.run(train_op, feed_dict={search_region: np.random.rand(batchsize, num_seq, 300, 300, 3),
                                                    template: np.random.rand(batchsize, 1, 128, 128, 3),
                                                    groundtruth_boxes: np.random.rand(batchsize, num_seq, 4),
                                                    groundtruth_classes: np.ones([batchsize, num_seq, 1])})

        """
        loss = sess.run(losses_dict, feed_dict={search_region: np.random.rand(batchsize, num_seq, 300, 300, 3),
                                                     template: np.random.rand(batchsize, 1, 128, 128, 3),
                                                     groundtruth_boxes: np.random.rand(batchsize, num_seq, 4),
                                                     groundtruth_classes: np.zeros([batchsize, num_seq, 1])})
        print loss
        
        
        
        print prediction['box_encodings'].shape
        print prediction['class_predictions_with_background'].shape
        for feat in prediction['feature_maps']:
            print feat.shape
        
        (12, 4110, 4)
        (12, 4110, 2)
        (12, 19, 19, 512)
        (12, 10, 10, 512)"""


if __name__ == '__main__':
  tf.app.run()