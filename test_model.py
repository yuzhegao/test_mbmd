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


os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
    model_config, train_config, input_config = get_configs_from_pipeline_file()
    model_fn = functools.partial(
        build_man_model,
        model_config=model_config,
        is_training=True)

    batchsize = 3
    num_seq = 4
    search_region = tf.placeholder(tf.float32, [batchsize, num_seq, 300, 300, 3])
    template = tf.placeholder(tf.float32, [batchsize, 1, 128, 128, 3])  ## each sequene only has one template
    groundtruth_boxes = tf.placeholder(tf.float32, [batchsize, num_seq, 4])
    groundtruth_classes = tf.placeholder(tf.int32, [batchsize, num_seq, 1])

    detection_model = model_fn()
    detection_model.provide_groundtruth(groundtruth_boxes,
                                        groundtruth_classes)
    prediction = detection_model.predict(template, search_region)

    losses_dict = detection_model.loss(prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss = sess.run(losses_dict, feed_dict={search_region: np.random.rand(batchsize, num_seq, 300, 300, 3),
                                                     template: np.random.rand(batchsize, 1, 128, 128, 3),
                                                     groundtruth_boxes: np.random.rand(batchsize, num_seq, 4),
                                                     groundtruth_classes: np.zeros([batchsize, num_seq, 1])})
        print loss
        """
        print prediction['box_encodings'].shape
        print prediction['class_predictions_with_background'].shape
        for feat in prediction['feature_maps']:
            print feat.shape
        
        (12, 4110, 4)
        (12, 4110, 2)
        (12, 19, 19, 512)
        (12, 10, 10, 512)
        """

if __name__ == '__main__':
  tf.app.run()