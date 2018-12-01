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
# print np.random.rand(2, 2, 300, 300, 3)
# print '\n','\n'
# print np.random.rand(2, 2, 300, 300, 3)
# exit()


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

def build_box_predictor(model,init_feature_maps,reuse=None):
    input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[300, 300, 3],name='search_region') ## should feed
    images = tf.expand_dims(input_cur_image, axis=0)
    float_images = tf.to_float(images)
    preprocessed_images = model.preprocess(float_images)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=0)

    input_init_gt_box = tf.constant(np.zeros((1, 4)), dtype=tf.float32)
    init_gt_box = tf.reshape(input_init_gt_box, shape=[1,1,4])
    groundtruth_classes = tf.ones(dtype=tf.float32, shape=[1, 1, 1])

    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)  ## the gt box(es), MAYBE for compute loss ???
                                ## I think, it make no sense during inferencce
    prediction_dict = model.predict_box_with_init(init_feature_maps, preprocessed_images, istraining=False)  ## ** can throw away the model_scope

    detections = model.postprocess(prediction_dict)  ## NMS
    original_image_shape = tf.shape(preprocessed_images)
    absolute_detection_boxlist = box_list_ops.to_absolute_coordinates(
                            box_list.BoxList(tf.squeeze(detections['detection_boxes'], axis=0)),
                            original_image_shape[2],  ## 300
                            original_image_shape[3]   ## 300                                             ## ** can just multiply 300 directly
                            )
    return absolute_detection_boxlist.get(), detections['detection_scores'], input_cur_image

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


    # search_region = tf.placeholder(dtype=tf.uint8, shape=[300, 300, 3])
    # template = tf.placeholder(dtype=tf.uint8, shape=[128, 128, 3])  ## each sequene only has one template
    # groundtruth_boxes = tf.placeholder(dtype=tf.float32, shape=[batchsize, num_seq, 4])
    # groundtruth_classes = tf.placeholder(dtype=tf.int32, shape=[batchsize, num_seq, 1])




    detection_model = model_fn()

    #input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[300, 300, 3])  ## should feed
    # search_region = tf.to_float(search_region)
    # search_region = tf.squeeze(search_region,axis=1)
    # search_region = detection_model.preprocess(search_region)
    # search_region = tf.expand_dims(search_region,axis=1)
    #
    # template = tf.to_float(template)
    # template = tf.squeeze(template,axis=1)
    # template = detection_model.preprocess(template, [128, 128])
    # template = tf.expand_dims(template,axis=1)



    ## IN training
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



    ## IN test (myself)
    # init_feature_maps = detection_model.extract_init_feature(template)
    # detection_model.provide_groundtruth(groundtruth_boxes,
    #                                     groundtruth_classes)
    # prediction = detection_model.predict_box_with_init(init_feature_maps, search_region, istraining=False)
    # detections = detection_model.postprocess(prediction)


    ## IN test (offical)
    initFeatOp, template = build_init_graph(detection_model, reuse=None)
    pre_box_tensor, scores_tensor, search_region = build_box_predictor(detection_model, initFeatOp, reuse=None)





    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # feat_extract_saver.restore(sess, train_config.fine_tune_checkpoint)
        # init_saver.restore(sess, train_config.fine_tune_checkpoint)

        variables_to_restore = tf.global_variables()
        restore_model(sess,'/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/dump',variables_to_restore)

        for i in xrange(epochs):
            for idx,(templates_batch, imgs_batch, gts_batch, labels_batch) in enumerate(otb_loader):
                print ('iter {}'.format(idx))

                # prediction_dict,detect_box = sess.run([prediction,detections],
                #                                            feed_dict={search_region: imgs_batch,
                #                                                       template: templates_batch,
                #                                                       groundtruth_boxes: gts_batch,
                #                                                       groundtruth_classes: labels_batch})
                #
                # #print ('loss {}'.format(loss))
                # print (prediction_dict['box_encodings'].shape)
                # print (prediction_dict['class_predictions_with_background'].shape)

                # for k,v in detect_box.items():
                #     print k
                #     detection_classes
                #     detection_boxes
                #     detection_scores
                #     num_detections

                # det = detect_box['detection_boxes'][0][0]*300.0 ## (y1,x1,y2,x2)
                # print det
                # det = det.astype(np.int32)
                # img1 = Image.fromarray(np.squeeze(imgs_batch))
                # draw_box(img1,[det[1],det[0],det[3],det[2]],'/home/yuzhe/test.jpg')

                # det = detect_box['detection_boxes'][0][:10] * 300.0  ## (y1,x1,y2,x2)
                # print det
                # det = det.astype(np.int32)
                #
                # detect = np.zeros_like(det)
                # detect[:,0],detect[:,1],detect[:,2],detect[:,3] = det[:,1], det[:,0], det[:,3], det[:,2]
                # img1 = Image.fromarray(np.squeeze(imgs_batch))
                # draw_mulitbox(img1, detect, '/home/yuzhe/test.jpg')



                imgs_batch = np.squeeze(imgs_batch)
                templates_batch = np.squeeze(templates_batch)

                boxes = sess.run(pre_box_tensor,feed_dict={search_region: imgs_batch,
                                                template: templates_batch,})

                print (boxes.shape)

                det = boxes[:1]
                detect = np.zeros_like(det)
                detect[:,0],detect[:,1],detect[:,2],detect[:,3] = det[:,1], det[:,0], det[:,3], det[:,2]
                draw_mulitbox(Image.fromarray(imgs_batch),detect,'/home/yuzhe/test.jpg')







                exit()
if __name__ == '__main__':
    main()


