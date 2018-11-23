import functools
import tensorflow as tf
from lib.object_detection.builders import preprocessor_builder, optimizer_builder
from lib.object_detection.utils import variables_helper
from lib.object_detection.core import standard_fields as fields
from lib.object_detection.core import preprocessor, batcher
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import numpy as np
import os
from core.preprocessor import  preprocess
slim=tf.contrib.slim



def _concat_tensor_dicts(tensor_dicts):
    concat_tensor_dict = dict()
    for k in tensor_dicts[0].keys():
        tensor_list = [t[k] for t in tensor_dicts]
        concat_tensor_dict[k] = tf.concat(tensor_list, axis=0)
    return concat_tensor_dict

def _split_tensor_dict(in_tensor_dict, seq_length):
    for key, value in in_tensor_dict.items():
        in_tensor_dict[key] = tf.split(in_tensor_dict[key], seq_length, axis=0)
    out_tensor_dicts = list()
    for i in range(seq_length):
        tensor_dict = dict()
        for key, value in in_tensor_dict.items():
            tensor_dict[key] = value[i]
        out_tensor_dicts.append(tensor_dict)

        ## input: {'image': (20,300,300,3); 'gt_box':(20,4)}
        ## output: [{'image': (300,300,3); 'gt_box':(4,)}, {same}, {same} .......{same}]
    return out_tensor_dicts



def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn, detection_model,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, data_augmentation_options, image_rootpath, seq_length=20):

    tensor_dict = create_tensor_dict_fn()
    #print tensor_dict
    #for i in tensor_dict:
    #    print i

    def _read_image(folder, im_names, groundtruth_boxes, seq_length=10):
        num_frames = len(im_names)
        size = 300
        template_size = 128

        if num_frames >= seq_length:
            start_id = np.random.randint(0,num_frames-seq_length+1)
            frame_ids = range(start_id, start_id+seq_length)
        else:
            frame_ids = np.random.randint(0, num_frames, seq_length)

        imgs = np.zeros([seq_length, size, size, 3], dtype=np.uint8)
        template = np.zeros([1, template_size, template_size, 3], dtype=np.uint8)
        first_gt = groundtruth_boxes[frame_ids[0],:]

        for ind, frame_id in enumerate(frame_ids):
            img = Image.open(im_names[frame_id]) ## im_names contain absolute path

            ## write the temlate img 128*128
            if ind == 0:
                template_array = img.crop(first_gt)
                template_array = template_array.resize(np.int32([template_size,template_size]))
                template_array = np.array(template_array).astype(np.uint8)
                if template_array.ndim < 3:
                    template_array = np.repeat(np.expand_dims(template_array, axis=2), repeats=3, axis=2)
                template[0] = template_array

            img = img.resize(np.int32([size, size]))
            img = np.array(img).astype(np.uint8)
            if img.ndim < 3:
                img = np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)
            imgs[ind] = img


        groundtruth_boxes = groundtruth_boxes[frame_ids,:]
        groundtruth_classes = np.ones([seq_length, 1], dtype=np.float32)
        return imgs, template, groundtruth_boxes, groundtruth_classes
    #
    # sess = tf.Session()
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    # out_dict = sess.run(tensor_dict)
    # for i in range(100):
    #     out_dict = sess.run(tensor_dict)
    #     _read_image(out_dict['folder'], out_dict['filename'], out_dict['groundtruth_boxes'], seq_length)

    images, template, groundtruth_boxes, groundtruth_classes = tf.py_func(_read_image,
                                                                [tensor_dict['folder'],tensor_dict['filename'],tensor_dict['groundtruth_boxes'], seq_length],
                                                                [tf.uint8, tf.uint8, tf.float32, tf.float32])

    images.set_shape([seq_length, 300, 300, 3])
    template.set_shape([1, 128, 128, 3])
    float_images, float_template = tf.to_float(images), tf.to_float(template)
    groundtruth_boxes.set_shape([seq_length, 4])
    groundtruth_classes.set_shape([seq_length, 1])
    #print '\n',images,'\n'

    tensor_dict = dict()
    tensor_dict[fields.InputDataFields.image] = float_images
    #tensor_dict['template'] = float_template  should add template to dict  after pre-process
    tensor_dict[fields.InputDataFields.groundtruth_boxes] = groundtruth_boxes
    tensor_dict[fields.InputDataFields.groundtruth_classes] = groundtruth_classes

    """
    tensor_dicts = _split_tensor_dict(tensor_dict, seq_length) ## a seq
    if data_augmentation_options:
        tensor_dicts = [preprocess(tensor_dict.copy()) for tensor_dict in tensor_dicts]

    for i in range(seq_length):
        tensor_dicts[i][fields.InputDataFields.image] = \
            detection_model.preprocess(tensor_dicts[i][fields.InputDataFields.image])
        tensor_dicts[i][fields.InputDataFields.groundtruth_classes].set_shape([1, 1])
        tensor_dicts[i][fields.InputDataFields.groundtruth_boxes].set_shape([1, 4])

    concat_tensor_dict = _concat_tensor_dicts(tensor_dicts)
    concat_tensor_dict['template'] = float_template
    #print '\n',concat_tensor_dict,'\n'
    """
    tensor_dict['template'] = float_template

    ## tensor_dict['images']  (20, 300, 300, 3)
    ## tensor_dict['groundtruth_boxes']  (20, 4)
    ## tensor_dict['groundtruth_classes']  (20, 1)

    #batched_tensor = tf.train.batch(concat_tensor_dict,
    batched_tensor = tf.train.batch(tensor_dict,
                                    capacity=batch_queue_capacity,
                                    batch_size=batch_size_per_clone,
                                    num_threads=num_batch_queue_threads,
                                    dynamic_pad=True
                                    )

    dtypes = [t.dtype for t in batched_tensor.values()]
    shapes = [t.get_shape() for t in batched_tensor.values()]
    names = list(batched_tensor.keys())


    prefetch_queue = tf.FIFOQueue(capacity=prefetch_queue_capacity, dtypes=dtypes, shapes=shapes, names=names)
    init_prefetch = prefetch_queue.enqueue(batched_tensor)
    tf.train.add_queue_runner(tf.train.QueueRunner(prefetch_queue, [init_prefetch] * num_batch_queue_threads))
    # x = prefetch_queue.dequeue()
    # sess = tf.Session()
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # def _normalize(x):
    #     return (x-x.min()) / (x.max()-x.min())
    #
    # fig = plt.figure()
    # for i in range(100):
    #     plt.clf()
    #     a = sess.run(x)
    #     for j in range(2):
    #         image = _normalize(a['image'][0,j])
    #         box = a['groundtruth_boxes'][0, j] * 300
    #         ax = plt.subplot(1,2,j+1)
    #         plt.imshow(image)
    #         ax.add_patch(
    #             patches.Rectangle(
    #                 (box[1], box[0]),  # (x,y)
    #                 box[3]-box[1],  # width
    #                 box[2]-box[0],  # height
    #                 fill=False,
    #                 edgecolor="red"
    #             )
    #         )
    return prefetch_queue

def _get_inputs(input_queue):
    tensor_dict = input_queue.dequeue() ## dequeue data

    images = tensor_dict[fields.InputDataFields.image]
    template = tensor_dict['template']
    groundtruth_box = tensor_dict[fields.InputDataFields.groundtruth_boxes]
    groundtruth_class = tensor_dict[fields.InputDataFields.groundtruth_classes]

    return images, template, groundtruth_box, groundtruth_class, None

def _create_losses(input_queue, create_model_fn):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
  """
  detection_model = create_model_fn()
  (images, template, groundtruth_boxes, groundtruth_classes,
   groundtruth_masks
  ) = _get_inputs(input_queue)
  #print _get_inputs(input_queue)

  detection_model.provide_groundtruth(groundtruth_boxes,
                                      tf.to_int32(groundtruth_classes),
                                      groundtruth_masks)
  #print images
  #print detection_model
  prediction_dict = detection_model.predict(template, images)
  print prediction_dict,'\n'

  losses_dict = detection_model.loss(prediction_dict)
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)  ## here we add all the loss -> total loss



def train(create_model_fn, create_tensor_dict_fn, train_config, train_dir, img_root):

    ## this fn maybe train the compute graph which build by Siamese-network
    ## maybe use some fn of the network class

    ## create the siamses network
    detection_model = create_model_fn()  ## a MANMetaArch    which object containing some model config
    data_augmentation_options = [ preprocessor_builder.build(step)
                                    for step in train_config.data_augmentation_options]

    ## prepare input
    with tf.device('cpu:0'):
        #global_step = slim.create_global_step()
        global_step = tf.train.create_global_step()

        input_queue = _create_input_queue(train_config.batch_size,
                                          create_tensor_dict_fn,
                                          detection_model,
                                          train_config.batch_queue_capacity,
                                          train_config.num_batch_queue_threads,
                                          train_config.prefetch_queue_capacity,
                                          data_augmentation_options,
                                          img_root)
    with tf.device('gpu:0'):
        _create_losses(input_queue, create_model_fn)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    training_optimizer = optimizer_builder.build(train_config.optimizer,set())



    # create initial restore op
    ## restore the checkpoint , notice !!!!!
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint)
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn



    # loss and grads
    total_loss = tf.losses.get_total_loss()
    ## ??? maybe use tf.add_loss in network class   ->  in fn:_create_losses

    grads_and_vars = training_optimizer.compute_gradients(total_loss, tf.trainable_variables())
    # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
    if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

    # Optionally freeze some layers by setting their gradients to be zero.
    if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

    # Optionally clip gradients
    if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
            grads_and_vars = slim.learning.clip_gradient_norms(
                grads_and_vars, train_config.gradient_clipping_by_norm)

    # Create gradient updates.
    grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                      global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')
    # create summary
    summaries = set()
    for loss_tensor in tf.losses.get_losses():
        summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.allow_growth = True
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    slim.learning.train(
        train_tensor,  ##train_tensor = tf.identity(total_loss, name='train_op')
        logdir=train_dir,
        session_config=session_config,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        saver=saver)
    


    """
    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())

        tf.train.start_queue_runners(sess=sess)

        tensor_dict = input_queue.dequeue()
        dic = sess.run(tensor_dict)
        print dic
    """



    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # training_optimizer = optimizer_builder.build(train_config.optimizer,
    #                                              global_summaries)
    # loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
    # optimize_op = training_optimizer.minimize(loss, global_step=slim.get_global_step())
    # sess.run(tf.global_variables_initializer())
    # init_fn(sess)
    # for i in range(train_config.num_steps):
    #     cur_loss, _ = sess.run([loss, optimize_op])
    #     print("Iteration %d, Loss: %f"%(i, cur_loss))
