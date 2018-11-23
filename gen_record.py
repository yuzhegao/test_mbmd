import os
import sys
sys.path.append('./lib/') ## !!!
sys.path.append('./lib/slim/')

import numpy as np
import glob
from object_detection.utils import ops as util_ops
from PIL import Image


import tensorflow as tf

#print sorted(glob.glob('/home/yuzhe/Downloads/part_vot_seq/Bird1/img/*'))

def gen_otb_record():
    data_root = '/home/yuzhe/Downloads/part_vot_seq/'
    seq_list = sorted(glob.glob(data_root + '*'))
    for i, seq in enumerate(seq_list):
        seq_list[i] = os.path.basename(seq)

    # print seq_list
    seq_img_dict = dict()
    label_dict = dict()
    for seq in seq_list:
        #print seq
        img_list = sorted(glob.glob(os.path.join(data_root, seq) + '/img/*.jpg'))
        seq_img_dict[seq] = img_list

        with open(os.path.join(data_root, seq, 'groundtruth_rect.txt'), 'r') as fl:
            lines = fl.readlines()

            if len(lines[0].rstrip().split(',')) > 1:
                for i, line in enumerate(lines):
                    lines[i] = line.rstrip().split(',')
            else:
                for i, line in enumerate(lines):
                    lines[i] = line.rstrip().split()

            lines = np.array(lines, dtype=np.float32)
            label_dict[seq] = lines


    #for keys,values in seq_img_dict.items():
    #    print(keys), len(values)

    #for keys,values in label_dict.items():
    #    print(keys), values.shape

    writer = tf.python_io.TFRecordWriter('data_folder/otb.tfrecords')
    for seq in seq_list:
        print seq

        if len(seq_img_dict[seq]) != len(label_dict[seq]):
            print len(seq_img_dict[seq])
            print len(label_dict[seq])

        assert len(seq_img_dict[seq]) == len(label_dict[seq])


        #print len(seq_img_dict[seq])
        #print label_dict[seq].shape
        #print label_dict[seq][:,0].shape
        #print (label_dict[seq][:,2] + label_dict[seq][:,0]).shape

        x_max = label_dict[seq][:,2] + label_dict[seq][:,0]
        y_max= label_dict[seq][:,3] + label_dict[seq][:,1]

        example = tf.train.Example(features=tf.train.Features(feature={
            'folder': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seq])),
            'image_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=seq_img_dict[seq])),
            'bndbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=label_dict[seq][:,0])),
            'bndbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=label_dict[seq][:,1])),
            'bndbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=x_max)),
            'bndbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=y_max)),
            }))
        ## notice: each seq is a example,so we use parse_single_example to parse it !
        writer.write(example.SerializeToString())

    writer.close()

def read_record():
    data_path = 'data_folder/otb.tfrecords'
    input_record_queue = tf.train.string_input_producer([data_path])
    record_reader = tf.TFRecordReader()
    _, serialized_example = record_reader.read(input_record_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'folder': tf.FixedLenFeature([], tf.string),
            'image_name': tf.VarLenFeature(tf.string),
            'bndbox/xmin': tf.VarLenFeature(tf.float32),
            'bndbox/ymin': tf.VarLenFeature(tf.float32),
            'bndbox/xmax': tf.VarLenFeature(tf.float32),
            'bndbox/ymax': tf.VarLenFeature(tf.float32),
        })
    tensor_dict = dict()
    tensor_dict['folder'] = features['folder']

    tensor_dict['filename'] = features['image_name'].values
    bndbox = tf.stack([features['bndbox/ymin'].values, features['bndbox/xmin'].values,
                       features['bndbox/ymax'].values, features['bndbox/xmax'].values], axis=1)
    # bndbox = tf.expand_dims(bndbox, axis=0)

    tensor_dict['groundtruth_boxes'] = bndbox

    classes_gt = tf.ones_like(features['bndbox/ymin'].values, dtype=tf.int32)
    label_id_offset = 1
    num_classes = 1
    classes_gt -= label_id_offset
    classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt,
                                                  depth=num_classes, left_pad=0)
    tensor_dict['groundtruth_classes'] = tf.to_int32(classes_gt)

    ## output dict: 'floder', 'filename', 'groundtruth_boxes'

    ## ---------------------------------------------------------------------------------

    seq_length = 20

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
    tensor_dict['image'] = float_images
    tensor_dict['groundtruth_boxes'] = groundtruth_boxes
    tensor_dict['groundtruth_classes'] = groundtruth_classes
    tensor_dict['template'] = float_template

    ## ---------------------------------------------------------------------------------


    batched_tensor = tf.train.batch(tensor_dict,
                                    capacity=30,
                                    batch_size=16,
                                    num_threads=1,
                                    dynamic_pad=True
                                    )

    dtypes = [t.dtype for t in batched_tensor.values()]
    names = list(batched_tensor.keys())

    prefetch_queue = tf.FIFOQueue(capacity=30, dtypes=dtypes, names=names)
    init_prefetch = prefetch_queue.enqueue(batched_tensor)
    tf.train.add_queue_runner(tf.train.QueueRunner(prefetch_queue, [init_prefetch] * 1))

    return  prefetch_queue

if __name__ == '__main__':
    pass
    #gen_otb_record()

    with tf.Session() as sess:
        input_queue = read_record()

        init = tf.global_variables_initializer()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        for iter in range(10):
            tensor_dict = input_queue.dequeue()
            dict = sess.run(tensor_dict)

            print '-------------------------'
            print dict['image'].shape,dict['template'].shape
            print dict['groundtruth_boxes'].shape,dict['groundtruth_classes'].shape,
            print #len(dict['template'])
            print '-------------------------'
            print 'iter {}'.format(iter)
