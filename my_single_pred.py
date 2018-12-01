import sys
sys.path.append('./')
sys.path.append('./lib/')
sys.path.append('./lib/slim/')

import functools
import numpy as np

import torch
import tensorflow as tf
from core import trainer_test, trainer_seq, input_reader
from core.model_builder import build_man_model
from google.protobuf import text_format

from lib.object_detection.core import box_list
from lib.object_detection.core import box_list_ops
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
from train.data_utils import draw_box,draw_mulitbox

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(0)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_configs_from_pipeline_file():
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile( '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/ssd_mobilenet_tracking.config', 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model.ssd
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    return model_config, train_config, input_config

def restore_model(sess, checkpoint_path, variables_to_restore):
    # variables_to_restore = tf.global_variables()
    name_to_var_dict = dict([(var.op.name, var) for var in variables_to_restore
                             if not var.op.name.endswith('Momentum')])

    # for k,v in name_to_var_dict.items():
    #     print k

    saver = tf.train.Saver(name_to_var_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

def build_init_graph(model, reuse=None):
    input_init_image = tf.placeholder(dtype=tf.uint8, shape=[128,128,3],name='template') ## template patch
    float_init_image = tf.to_float(input_init_image)
    float_init_image = tf.expand_dims(tf.expand_dims(float_init_image, axis=0), axis=0)

    preprocessed_init_image = model.preprocess(float_init_image, [128,128])  ## resize + mobilenet.preprocess
    init_feature_maps = model.extract_init_feature(preprocessed_init_image) ## mobilenet.extract_features
    return init_feature_maps,input_init_image

def build_box_predictor(model,init_feature_maps,init_gt_box,groundtruth_classes,reuse=None):
    input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[300, 300, 3],name='search_region') ## should feed
    images = tf.expand_dims(input_cur_image, axis=0)
    float_images = tf.to_float(images)
    preprocessed_images = model.preprocess(float_images)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=0)

    # input_init_gt_box = tf.constant(np.zeros((1, 4)), dtype=tf.float32)
    # init_gt_box = tf.reshape(input_init_gt_box, shape=[1,1,4])
    # groundtruth_classes = tf.ones(dtype=tf.float32, shape=[1, 1, 1])

    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)  ## the gt box(es), MAYBE for compute loss ???
                                ## I think, it make no sense during inferencce
    prediction_dict = model.predict_box_with_init(init_feature_maps, preprocessed_images, istraining=False)

    detections = model.postprocess(prediction_dict)  ## NMS

    return detections, detections['detection_scores'], input_cur_image

def main():
    ## dataset

    ## tf graph
    global_step = tf.train.create_global_step()

    model_config, train_config, input_config = get_configs_from_pipeline_file()
    model_fn = functools.partial(
        build_man_model,
        model_config=model_config,
        is_training=True)

    epochs = 100
    batchsize = 1
    num_seq = 1

    otb_data = otb_dataset('/home/yuzhe/Downloads/part_vot_seq/')
    otb_loader = torch.utils.data.DataLoader(otb_data,
                                             batch_size=batchsize, shuffle=True, num_workers=1,
                                             collate_fn=otb_collate, drop_last=True)


    search_region = tf.placeholder(dtype=tf.uint8, shape=[batchsize,num_seq, 300, 300, 3],name='search_region')
    template = tf.placeholder(dtype=tf.uint8, shape=[batchsize,num_seq, 128, 128, 3],name='template')  ## each sequene only has one template
    groundtruth_boxes = tf.placeholder(dtype=tf.float32, shape=[batchsize, num_seq, 4],name='groundtruth_boxes')
    groundtruth_classes = tf.placeholder(dtype=tf.int32, shape=[batchsize, num_seq, 1],name='groundtruth_classes')




    detection_model = model_fn()

    ## IN training --->failure
    # detection_model.provide_groundtruth(groundtruth_boxes,
    #                                     groundtruth_classes)
    # prediction = detection_model.predict(template, search_region)
    #
    # detections = detection_model.postprocess(prediction)
    #
    # losses_dict = detection_model.loss(prediction)
    #
    #
    # total_loss = losses_dict['localization_loss'] + losses_dict['classification_loss']
    # training_optimizer = optimizer_builder.build(train_config.optimizer, set())
    # train_op = training_optimizer.minimize(total_loss, global_step=global_step)



    ## IN test (myself) -->failure
    # init_feature_maps = detection_model.extract_init_feature(template)
    # detection_model.provide_groundtruth(groundtruth_boxes,
    #                                     groundtruth_classes)
    # prediction = detection_model.predict_box_with_init(init_feature_maps, search_region, istraining=False)
    # detections = detection_model.postprocess(prediction)


    ## IN test (offical)  --> success
    # initFeatOp, template = build_init_graph(detection_model, reuse=None)
    # pre_box_tensor, scores_tensor, search_region = build_box_predictor(detection_model, initFeatOp,
    #                                                                    init_gt_box=groundtruth_boxes,
    #                                                                    groundtruth_classes=groundtruth_classes,
    #                                                                    reuse=None)


    ## Outside ---> success
    float_init_image = tf.to_float(template)
    float_init_image = tf.squeeze(float_init_image,axis=1)
    preprocessed_init_image = detection_model.preprocess(float_init_image, [128, 128])  ## resize + mobilenet.preprocess
    preprocessed_init_image = tf.expand_dims(preprocessed_init_image,axis=1)


    float_images = tf.to_float(search_region)
    float_images = tf.squeeze(float_images,axis=1)
    preprocessed_images = detection_model.preprocess(float_images)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=1)

    detection_model.provide_groundtruth(groundtruth_boxes,
                              groundtruth_classes,
                              None)
    ## important
    #prediction_dict = detection_model.predict_box_with_init(init_feature_maps, preprocessed_images, istraining=False)
    ## replace with model.predict()
    #print preprocessed_init_image,preprocessed_images ##(1, 1, 128, 128, 3)  (1, 1, 300, 300, 3)
    prediction_dict = detection_model.predict(preprocessed_init_image,preprocessed_images,istraining=True)

    detections = detection_model.postprocess(prediction_dict)

    losses_dict = detection_model.loss(prediction_dict)
    total_loss = losses_dict['localization_loss'] + losses_dict['classification_loss']
    training_optimizer = optimizer_builder.build(train_config.optimizer, set())
    train_op = training_optimizer.minimize(total_loss, global_step=global_step)


    test_op = detection_model.test_dict['batch_reg_targets']


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # feat_extract_saver.restore(sess, train_config.fine_tune_checkpoint)
        # init_saver.restore(sess, train_config.fine_tune_checkpoint)

        variables_to_restore = tf.global_variables()
        restore_model(sess, '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/dump', variables_to_restore)

        for i in xrange(epochs):
            for idx, (templates_batch, imgs_batch, gts_batch, labels_batch) in enumerate(otb_loader):
                print ('iter {}'.format(idx))


                # imgs_batch = np.squeeze(imgs_batch)
                # templates_batch = np.squeeze(templates_batch)

                #labels_batch = np.zeros_like(labels_batch,dtype=np.int32)

                test, detection_dict,pred_dict,loss_value = sess.run([test_op, detections,prediction_dict,losses_dict], feed_dict={search_region: imgs_batch,
                                                                                        template: templates_batch,
                                                                                        groundtruth_boxes: gts_batch,
                                                                                        groundtruth_classes: labels_batch})
                # for k, v in detection_dict.items():
                #     print k
                #     print v.shape
                # print pred_dict['box_encodings'].shape
                #
                # print detection_dict['detection_boxes'].shape
                # print detection_dict['detection_scores']
                # print detection_dict['detection_classes']
                # boxes = detection_dict['detection_boxes']
                #
                # for idx,box in enumerate(boxes):
                #     det = box[:5]
                #     detect = np.zeros_like(det)
                #     detect[:, 0], detect[:, 1], detect[:, 2], detect[:, 3] = det[:, 1], det[:, 0], det[:, 3], det[:, 2]
                #     detect = (detect * 300).astype(np.int32)
                #     draw_mulitbox(Image.fromarray(imgs_batch[idx][0]), detect, '/home/yuzhe/tmp/test_{}.jpg'.format(idx))
                #
                #     gt1 = [gts_batch[idx][0][1],gts_batch[idx][0][0],gts_batch[idx][0][3],gts_batch[idx][0][2]]
                #     draw_box(Image.fromarray(imgs_batch[idx][0]),gt1,'/home/yuzhe/tmp/test_{}gt.jpg'.format(idx))
                #
                #     #print templates_batch.shape
                #     tem = Image.fromarray(templates_batch[idx][0])
                #     tem.save('/home/yuzhe/tmp/test_{}t.jpg'.format(idx))

                # print labels_batch
                # print labels_batch.shape

                # np.set_printoptions(threshold='nan')
                print loss_value
                # print test
                # print test.shape
                # print '\n'
                # print test2
                # print test2.shape















                exit()
if __name__ == '__main__':
    main()


