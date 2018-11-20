# -- coding: utf-8 --

import cv2
import os
import sys
sys.path.append('./lib')
sys.path.append('./lib/slim')

from lib.object_detection.protos import pipeline_pb2
from lib.object_detection.core import box_list
from lib.object_detection.core import box_list_ops
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import vot
from vggm import vggM
from sample_generator import *
from tracking_utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/float(np.size(new_distances)) * 100.0

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/float(np.size(new_distances))

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


def get_configs_from_pipeline_file(config_file):
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  model_config = pipeline_config.model.ssd
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader
  eval_config = pipeline_config.eval_config

  return model_config, train_config, input_config, eval_config


def restore_model(sess, model_scope, checkpoint_path, variables_to_restore):
    # variables_to_restore = tf.global_variables()
    name_to_var_dict = dict([(var.op.name.lstrip(model_scope+'/'), var) for var in variables_to_restore
                             if not var.op.name.endswith('Momentum')])
    saver = tf.train.Saver(name_to_var_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

def crop_search_region(img, gt, win_size, scale=4, mean_rgb=128, offset=None):
    ## crop the search region, which is four times the size of the target,and centering in gt's center
    # gt: [ymin, xmin, ymax, xmax]

    bnd_ymin, bnd_xmin, bnd_ymax, bnd_xmax = gt
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    # cx, cy = gt[:2] + gt[2:] / 2
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    diag = np.sum( bnd_h** 2 + bnd_w**2) ** 0.5
    origin_win_size = diag * scale
    origin_win_size_h, origin_win_size_w = bnd_h * scale, bnd_w * scale
    # origin_win_size_h = origin_win_size
    # origin_win_size_w = origin_win_size
    #print "che {} {}".format(img.size,img.size[1::-1]) ##(576, 432) (432, 576)
    im_size = img.size[1::-1] ##[H,W]
    min_x = np.round(cx - origin_win_size_w / 2).astype(np.int32)
    max_x = np.round(cx + origin_win_size_w / 2).astype(np.int32)
    min_y = np.round(cy - origin_win_size_h / 2).astype(np.int32)
    max_y = np.round(cy + origin_win_size_h / 2).astype(np.int32)

    if offset is not None:
        min_offset_y, max_offset_y = (bnd_ymax - max_y, bnd_ymin - min_y)
        min_offset_x, max_offset_x = (bnd_xmax - max_x, bnd_xmin - min_x)
        offset[0] = np.clip(offset[0] * origin_win_size_h, min_offset_y, max_offset_y)
        offset[1] = np.clip(offset[1] * origin_win_size_w, min_offset_x, max_offset_x)
        offset = np.int32(offset)
        min_y += offset[0]
        max_y += offset[0]
        min_x += offset[1]
        max_x += offset[1]

    print "che {} / {}".format(gt, [min_y,min_x,max_y,max_x])

    win_loc = np.array([min_y, min_x]) ## what if min_x/min_y <0 ???

    gt_x_min, gt_y_min = ((bnd_xmin-min_x)/origin_win_size_w, (bnd_ymin - min_y)/origin_win_size_h)
    gt_x_max, gt_y_max = [(bnd_xmax-min_x)/origin_win_size_w, (bnd_ymax - min_y)/origin_win_size_h]
    # coordinates on window
    #relative coordinates of gt bbox to the search region

    unscaled_w, unscaled_h = [max_x - min_x + 1, max_y - min_y + 1] ## before scaled to 300*300
    min_x_win, min_y_win, max_x_win, max_y_win = (0, 0, unscaled_w, unscaled_h)
    ## in search region coordinate
    min_x_im, min_y_im, max_x_im, max_y_im = (min_x, min_y, max_x+1, max_y+1)
    ## in origin img coordinate   (useless)

    img = img.crop([min_x_im, min_y_im, max_x_im, max_y_im]) ## crop the search region
    ## from the code below: if the min/max out of origin img bound, then just padding
    img_array = np.array(img)

    if min_x < 0:
        min_x_im = 0
        min_x_win = 0 - min_x
    if min_y < 0:
        min_y_im = 0
        min_y_win = 0 - min_y
    if max_x+1 > im_size[1]:
        max_x_im = im_size[1]
        max_x_win = unscaled_w - (max_x + 1 - im_size[1])
    if max_y+1 > im_size[0]:
        max_y_im = im_size[0]
        max_y_win = unscaled_h - (max_y + 1 - im_size[0])
    ## after padding

    unscaled_win = np.ones([unscaled_h, unscaled_w, 3], dtype=np.uint8) * np.uint8(mean_rgb)
    unscaled_win[min_y_win:max_y_win, min_x_win:max_x_win] = img_array[min_y_win:max_y_win, min_x_win:max_x_win]
    ## here padding with 128(mean value)

    unscaled_win = Image.fromarray(unscaled_win)
    height_scale, width_scale = np.float32(unscaled_h)/win_size, np.float32(unscaled_w)/win_size
    win = unscaled_win.resize([win_size, win_size], resample=Image.BILINEAR)
    ## now　resize and get "resize_scale_rate"

    # win = sp.misc.imresize(unscaled_win, [win_size, win_size])
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max]), win_loc, [height_scale, width_scale]
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)



class MobileTracker(object):
    def __init__(self, image, region):
        init_training = True
        config_file = '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/ssd_mobilenet_tracking.config'
        checkpoint_dir = '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/dump'

        model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
        model = build_man_model(model_config=model_config, is_training=False)
        ## fn build_man_model: return a MANMetaArch netwrk object

        model_scope = 'model'

        ##-------------------------------------------------------------------------------------------------

        self.initFeatOp, self.initInputOp = build_init_graph(model, model_scope, reuse=None)
        ## self.initFeatOp -> template feature vector
        ## self.initFeatOp :using mobilenet process 127*127 template patch

        self.initConstantOp = tf.placeholder(tf.float32, [1,1,1,512]) ## maybe the feature map of  template
        ## notice: one-shot learning !

        ## maybe:  build_box_predictor -> using siameseRPN to predict proposal box
        self.pre_box_tensor, self.scores_tensor, self.input_cur_image = build_box_predictor(model, model_scope, self.initConstantOp, reuse=None)
        ## predict bbox

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #if not init_training:
        variables_to_restore = tf.global_variables()
        restore_model(self.sess, model_scope, checkpoint_dir, variables_to_restore) ## restore the siamese network checkpoint

        ##-------------------------------------------------------------------------------------------------

        init_img = Image.fromarray(image) ## the first frame
        init_img_array = np.array(init_img)
        self.expand_channel = False
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)
            self.expand_channel = True

        init_gt1 = [region.x,region.y,region.width,region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax


        gt_boxes = np.zeros((1,4))
        gt_boxes[0,0] = init_gt[0] / float(init_img.height)
        gt_boxes[0,1] = init_gt[1] / float(init_img.width)
        gt_boxes[0,2] = init_gt[2] / float(init_img.height)
        gt_boxes[0,3] = init_gt[3] / float(init_img.width)  ## ymin xmin ymax xmax  -> ratio  (very confused )

        img1_xiaobai = np.array(init_img)
        pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img.width
        pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img.height ## ???
        #cx = (gt_boxes[0, 3] + gt_boxes[0, 1]) / 2.0 * init_img.width
        #cy = (gt_boxes[0, 2] + gt_boxes[0, 0]) / 2.0 * init_img.height
        startx = gt_boxes[0, 1] * init_img.width - pad_x
        starty = gt_boxes[0, 0] * init_img.height - pad_y
        endx = gt_boxes[0, 3] * init_img.width + pad_x
        endy = gt_boxes[0, 2] * init_img.height + pad_y

        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - init_img.width + 1))
        bottom_pad = max(0, int(endy - init_img.height + 1)) ## prevent bbox out of init_img

        ## re-compute the x1,x2,y1,y2 after padding
        startx = int(startx + left_pad)
        starty = int(starty + top_pad)
        endx = int(endx + left_pad)
        endy = int(endy + top_pad)


        if top_pad or left_pad or bottom_pad or right_pad:
            r = np.pad(img1_xiaobai[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            g = np.pad(img1_xiaobai[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)
            b = np.pad(img1_xiaobai[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                       constant_values=128)  ## padding value=128 ?
            r = np.expand_dims(r, 2)
            g = np.expand_dims(g, 2)
            b = np.expand_dims(b, 2)

            img1_xiaobai = np.concatenate((r, g, b), axis=2)

        img1_xiaobai = Image.fromarray(img1_xiaobai)
        im = np.array(init_img) ## backup for crop pos/neg sample region for training

        # gt_boxes resize
        init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
        init_img_crop = init_img_crop.resize([128,128], resample=Image.BILINEAR)
        self.last_gt = init_gt  # ymin xmin ymax xmax

        self.init_img_array = np.array(init_img_crop)
        self.init_feature_maps = self.sess.run(self.initFeatOp, feed_dict={self.initInputOp:self.init_img_array}) ## input the template
        ## self.init_feature_maps-> [1,1,1,512] for template
        ## here we get the init_feature_maps of template
        ## notice: just maintain as a filter !!!

        ##-------------------------------------------------------------------------------------------------

        ## here we can build VGGM (binary classification net )
        self.mdnet = vggM()
        self.imageOp = tf.placeholder(dtype=tf.float32, shape=(20, 107, 107, 3)) ## Nr=20 proposal
        self.outputsOp = self.mdnet.vggM(self.imageOp)
        self.researchImageOp = tf.placeholder(dtype=tf.float32, shape=(256, 107, 107, 3))
        self.researchOutputsOp = self.mdnet.vggM(self.researchImageOp, reuse=True) ## maybe for init training ???

        self.imageSingleOp = tf.placeholder(dtype=tf.float32, shape=(1, 107, 107, 3))
        self.outputsSingleOp = self.mdnet.vggM(self.imageSingleOp, reuse=True) ## only one, maybe for first highest score box

        self.featInputOp = tf.placeholder(dtype=tf.float32, shape=(250, 3, 3, 512)) ## the output of vggm-layer3
        self.labelOp = tf.placeholder(dtype=tf.float32, shape=(250, 2))
        self.lrOp = tf.placeholder(tf.float32, )
        self.logitsOp,_ = self.mdnet.classification(self.featInputOp)
        self.lossOp,_ = self.mdnet.loss(self.logitsOp, self.labelOp)


        self.optimizer_vggm1 = tf.train.MomentumOptimizer(learning_rate=self.lrOp, momentum=0.9)
        trainable_vars_vggm = tf.trainable_variables()
        vggMTrainableVars1 = [var for var in trainable_vars_vggm if (var.name.startswith("VGGM"))]
        trainVGGMGradOp1 = self.optimizer_vggm1.compute_gradients(self.lossOp, var_list=vggMTrainableVars1)

        ## train the MD-Net op
        self.trainVGGMOp = self.optimizer_vggm1.apply_gradients(trainVGGMGradOp1)

        self.imageOp1 = tf.placeholder(dtype=tf.float32, shape=(256, 107, 107, 3)) ## a batch=256 ,for init md-net training
        self.featOp = self.mdnet.extractFeature(self.imageOp1)

        all_vars = tf.global_variables()
        vggMVars = [var for var in all_vars if (var.name.startswith("VGGM"))]
        vggMVarsRestore = [var for var in all_vars if
                           (var.name.startswith("VGGM") and not var.name.endswith("Momentum:0"))]
        vggMSaver = tf.train.Saver(var_list=vggMVarsRestore)

        ##-------------------------------------------------------------------------------------------------
        ## 2018/11/16

        init_fn = tf.variables_initializer(var_list=vggMVars)
        self.sess.run(init_fn) ## init variable of vggm

        pos_examples = gen_samples(SampleGenerator('gaussian', init_img.size, trans_f=0.1, scale_f=1.2), bbox=np.array(init_gt1), n=500, overlap_range=[0.7, 1])
        ## positive samples -> overlap_range=[0.7, 1]
        pos_regions = extract_regions(im, pos_examples)
        pos_regions = pos_regions[:, :, :, ::-1]

        neg_examples = np.concatenate([
            gen_samples(SampleGenerator('uniform', init_img.size, 1, 2, 1.1), np.array(init_gt1), 5000 // 2, [0, 0.5]),
            gen_samples(SampleGenerator('whole', init_img.size, 0, 1.2, 1.1), np.array(init_gt1), 5000 // 2, [0, 0.5])])
        ## positive samples -> overlap_range=[0, 0.5]
        neg_regions = extract_regions(im, neg_examples)
        neg_regions = neg_regions[:, :, :, ::-1]

        vggMSaver.restore(self.sess, '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/ckpt/VGGM/vggMParams.ckpt')

        neg_features = np.zeros((5000, 3, 3, 512))
        pos_features = np.zeros((500, 3, 3, 512))
        num_iter = 5000 / 256
        for t in range(num_iter):
            neg_features[t * 256:(t + 1) * 256, :, :, :] = self.sess.run(self.featOp, feed_dict={
                self.imageOp1: neg_regions[t * 256:(t + 1) * 256, :, :, :]})
        residual = 5000 - 256 * num_iter
        tmp = 256 / residual + 1
        tmp1 = np.tile(neg_regions[num_iter * 256:, :, :, :], (tmp, 1, 1, 1))
        tmp1 = self.sess.run(self.featOp, feed_dict={self.imageOp1: tmp1[:256, :, :, :]})
        neg_features[num_iter * 256:, :, :, :] = tmp1[:residual, :, :, :]

        num_iter = 500 / 256
        for t in range(num_iter):
            pos_features[t * 256:(t + 1) * 256, :, :, :] = self.sess.run(self.featOp, feed_dict={
                self.imageOp1: pos_regions[t * 256:(t + 1) * 256, :, :, :]})
        residual = 500 - 256 * num_iter
        tmp = 256 / residual + 1
        tmp1 = np.tile(pos_regions[num_iter * 256:, :, :, :], (tmp, 1, 1, 1))
        tmp1 = self.sess.run(self.featOp, feed_dict={self.imageOp1: tmp1[:256, :, :, :]})
        pos_features[num_iter * 256:, :, :, :] = tmp1[:residual, :, :, :]
        ## from pos/neg sample , get pos/neg feature


        labels1 = np.array([0, 1])
        labels1 = np.reshape(labels1, (1, 2))
        labels1 = np.tile(labels1, (50, 1))
        labels2 = np.array([1, 0])
        labels2 = np.reshape(labels2, (1, 2))
        labels2 = np.tile(labels2, (200, 1))
        self.labels = np.concatenate((labels1, labels2), axis=0)  ## (50+200,2)

        for iter in range(30):
            ## 30 iter, each use 50+200 samples
            pos_feat = np.random.randint(0, 500, 50)
            pos_feat = pos_features[pos_feat]
            neg_feat = np.random.randint(0, 5000, 200)
            neg_feat = neg_features[neg_feat] ## here feed the pos/neg feature

            featInputs = np.concatenate((pos_feat, neg_feat), axis=0)

            _, loss1, logits1 = self.sess.run([self.trainVGGMOp, self.lossOp, self.logitsOp],
                                         feed_dict={self.featInputOp: featInputs, self.labelOp: self.labels, self.lrOp: 0.0001})

        logits1 = logits1[:50,1] ## only pos samples ?
        self.first_score = np.max(logits1) ## notice: the scores of target in the first img !!!

        ##-------------------------------------------------------------------------------------------------

        tmp1 = np.random.randint(0, 500, 50)
        self.pos_feat_record = pos_features[tmp1, :, :, :]
        tmp1 = np.random.randint(0, 5000, 200)
        self.neg_feat_record = neg_features[tmp1, :, :, :] ## ?

        self.target_w = init_gt[3] - init_gt[1]
        self.target_h = init_gt[2] - init_gt[0] ## record current w,h

        self.first_w = init_gt[3] - init_gt[1]
        self.first_h = init_gt[2] - init_gt[0]
        self.pos_regions_record = []
        self.neg_regions_record = []
        self.i = 0

        self.startx = 0
        self.starty = 0

    ##-------------------------------------------------------------------------------------------------

    def track(self, image):
        self.i += 1
        cur_ori_img = Image.fromarray(image)
        # if self.expand_channel:
        #     cur_ori_img = np.array(cur_ori_img)
        #     cur_ori_img = np.expand_dims(cur_ori_img, axis=2)
        #     cur_ori_img = np.repeat(cur_ori_img, repeats=3, axis=2)
        #     cur_ori_img = Image.fromarray(cur_ori_img)
        cur_ori_img_array = np.array(cur_ori_img)

        cropped_img, last_gt_norm, win_loc, scale = crop_search_region(cur_ori_img, self.last_gt, 300, mean_rgb=128)
        ## not read fn: crop_search_region
        cur_img_array = np.array(cropped_img)
        detection_box_ori, scores = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                             feed_dict={self.input_cur_image: cur_img_array,
                                                        self.initConstantOp: self.init_feature_maps})
        # detection_box = detection_box[0]
        #print scores.shape ## the R scores
        #print scores[0]
        #print detection_box_ori
        #print '\n'


        detection_box_ori[:, 0] = detection_box_ori[:, 0] * scale[0] + win_loc[0]
        detection_box_ori[:, 1] = detection_box_ori[:, 1] * scale[1] + win_loc[1]
        detection_box_ori[:, 2] = detection_box_ori[:, 2] * scale[0] + win_loc[0]
        detection_box_ori[:, 3] = detection_box_ori[:, 3] * scale[1] + win_loc[1]
        ## look fn:crop_search_region
        ## just turn the bbox  in search region coordinate --> origin img coordinate

        rank = np.argsort(scores)
        k = 20
        candidates = rank[0, -k:] ## top-20
        pixel_count = np.zeros((k,))
        for ii in range(k):
            bb = detection_box_ori[candidates[ii], :].copy()

            ## 0-y1   1-x1   2-y2   3-x2
            x1 = max(self.last_gt[1], bb[1])
            y1 = max(self.last_gt[0], bb[0])
            x2 = min(self.last_gt[3], bb[3])
            y2 = min(self.last_gt[2], bb[2])
            ## IOU ?
            pixel_count[ii] = (x2 - x1) * (y2 - y1) / float(
                                                        (self.last_gt[2] - self.last_gt[0]) * (self.last_gt[3] -self.last_gt[1]) +
                                                        (bb[3] - bb[1]) * (bb[2] - bb[0]) -
                                                        (x2 - x1) * (y2 - y1)
                                                    )

        threshold = 0.4
        passed = pixel_count > (threshold)  ## throw out the too far proposals (w.r.t last gt)
        if np.sum(passed) > 0:
            candidates_left = candidates[passed]
            max_idx = candidates_left[np.argmax(scores[0, candidates_left])]
        else:
            max_idx = 0

        search_box1 = detection_box_ori[max_idx]
        search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
        search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
        search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
        search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)  ## y_min x_min y_max x_max

        if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]): ## ???
            score_max = -1
        else:
            search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                           search_box1[2] - search_box1[0]] ## [x,y,w,h]
            search_box1 = np.reshape(search_box1, (1, 4))
            search_regions = extract_regions(cur_ori_img_array, search_box1)
            search_regions = search_regions[:,:,:,::-1]
            score_max = self.sess.run(self.outputsSingleOp, feed_dict={self.imageSingleOp: search_regions}) ## (1,2)
            score_max = score_max[0, 1]

        ##-------------------------------------------------------------------------------------------------
        ## now we can make decision (from the scores)

        ## at the present, score_max belongs to the proposal bbox of highest similiarty scores

        ## case 1
        ## MAYBE for top-20 proposals
        if score_max < 0:
            search_box1 = detection_box_ori[:20]  ## just NMS, then get 20 (just the top-20)
            search_box = np.zeros_like(search_box1)
            search_box[:, 1] = search_box1[:, 0]
            search_box[:, 0] = search_box1[:, 1]
            search_box[:, 2] = search_box1[:, 3]
            search_box[:, 3] = search_box1[:, 2]
            ## x_min y_min x_max y_max (fuck!)

            haha = np.ones_like(search_box[:, 2]) * 3 ##(20,)
            search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
            search_box[:, 3] = search_box[:, 3] - search_box[:, 1] ## x_min y_min w h
            search_box[:, 2] = np.maximum(search_box[:, 2], haha)
            search_box[:, 3] = np.maximum(search_box[:, 3], haha)
            haha2 = np.zeros_like(search_box[:, 0])
            search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
            search_box[:, 1] = np.maximum(search_box[:, 1], haha2)

            haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
            search_box[:, 0] = np.minimum(search_box[:, 0], haha)
            haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
            search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

            search_regions = extract_regions(cur_ori_img_array, search_box)
            search_regions = search_regions[:, :, :, ::-1]
            mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
            mdnet_scores = mdnet_scores[:, 1]
            mdnet_scores1 = passed * mdnet_scores  ## pasased: binary
            max_idx1 = np.argmax(mdnet_scores1)
            if mdnet_scores1[max_idx1] > 0:
                max_idx = max_idx1
                score_max = mdnet_scores1[max_idx1]
            elif np.max(mdnet_scores) > 0:
                max_idx = np.argmax(mdnet_scores)
                score_max = mdnet_scores[max_idx] ## temporary score_max
            else:
                score_max = -1
        detection_box = detection_box_ori[max_idx] ## temporary detection_box



        ## case 2
        ## MAYBE if top-20 is not proper, just sample 256 box around last_gt
        if score_max < 0:
            gt_tmp = np.array([self.last_gt[1], self.last_gt[0], self.last_gt[3] - self.last_gt[1],
                               self.last_gt[2] - self.last_gt[0]])
            candidates_samples = gen_samples(SampleGenerator('gaussian', cur_ori_img.size, trans_f=0.6, scale_f=1.05, valid=True),
                                             gt_tmp, 256)
            candidates_regions = extract_regions(cur_ori_img_array, candidates_samples)
            candidates_regions = candidates_regions[:, :, :, ::-1]
            researchScores = self.sess.run(self.researchOutputsOp, feed_dict={self.researchImageOp: candidates_regions}) ## 256 sample
            researchScores = researchScores[:, 1]
            top_idx = np.argsort(-researchScores)
            top_scores = researchScores[top_idx[:5]]
            score_max = top_scores.mean()
            target_bbox = candidates_samples[top_idx[:5]].mean(axis=0)  ## maybe a soft trick, use mean value of top-5 box
            if score_max > 0:
                detection_box = np.array(
                    [target_bbox[1], target_bbox[0], target_bbox[3] + target_bbox[1], target_bbox[2] + target_bbox[0]])



        ## case 3
        if scores[0, max_idx] < 0.3:  # and score_max < 20.0:
            ## scores[0, max_idx] < 0.3 ---> a premise

            search_gt = (np.array(self.last_gt)).copy() ## ymin xmin ymax xmax
            # search_gt = last_gt.copy()
            search_gt[0] = cur_ori_img.height / 2.0 - (self.last_gt[2] - self.last_gt[0]) / 2.0
            search_gt[2] = cur_ori_img.height / 2.0 + (self.last_gt[2] - self.last_gt[0]) / 2.0
            search_gt[1] = cur_ori_img.width / 2.0 - (self.last_gt[3] - self.last_gt[1]) / 2.0
            search_gt[3] = cur_ori_img.width / 2.0 + (self.last_gt[3] - self.last_gt[1]) / 2.0 ## in the center of img

            cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                               mean_rgb=128)
            cur_img_array = np.array(cropped_img1)
            detection_box_ori1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                   feed_dict={self.input_cur_image: cur_img_array,
                                                              self.initConstantOp: self.init_feature_maps})

            ## now we get 100 proposal boxes  again

            ## ====================================================
            if scores1[0, 0] > 0.8:
                detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
                detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
                detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
                detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
                detection_box_ori = detection_box_ori1.copy()
                # max_idx = 0
                search_box1 = detection_box_ori[0]

                search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
                search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
                search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
                search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
                if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                    score_max = -1
                else:
                    search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                                   search_box1[2] - search_box1[0]]
                    search_box1 = np.reshape(search_box1, (1, 4))

                    search_regions = extract_regions(cur_ori_img_array, search_box1)
                    search_regions = search_regions[:, :, :, ::-1]
                    score_max = self.sess.run(self.outputsSingleOp, feed_dict={self.imageSingleOp: search_regions})
                    score_max = score_max[0, 1]

                # search_box1 = [search_box1[1],search_box1[0],search_box1[3]-search_box1[1],search_box1[2]-search_box1[0]]
                # search_box1 = np.reshape(search_box1, (1, 4))
                # search_regions = extract_regions(cur_ori_img_array, search_box1)
                # score_max = sess.run(outputsSingleOp, feed_dict={imageSingleOp: search_regions})
                if score_max > 0:
                    max_idx = 0
                    scores = scores1.copy()

                    detection_box = detection_box_ori[max_idx]

                if score_max < 0:
                    search_box1 = detection_box_ori[:20]
                    search_box = np.zeros_like(search_box1)
                    search_box[:, 1] = search_box1[:, 0]
                    search_box[:, 0] = search_box1[:, 1]
                    search_box[:, 2] = search_box1[:, 3]
                    search_box[:, 3] = search_box1[:, 2]
                    haha = np.ones_like(search_box[:, 2]) * 3
                    search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
                    search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
                    search_box[:, 2] = np.maximum(search_box[:, 2], haha)
                    search_box[:, 3] = np.maximum(search_box[:, 3], haha)
                    haha2 = np.zeros_like(search_box[:, 0])
                    search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
                    search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
                    haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
                    search_box[:, 0] = np.minimum(search_box[:, 0], haha)
                    haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
                    search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

                    search_regions = extract_regions(cur_ori_img_array, search_box)
                    search_regions = search_regions[:, :, :, ::-1]
                    mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
                    mdnet_scores = mdnet_scores[:, 1]
                    max_idx1 = np.argmax(mdnet_scores)
                    if mdnet_scores[max_idx1] > 0 and scores1[0,max_idx1] > 0.3:
                        score_max = mdnet_scores[max_idx1]
                        max_idx = max_idx1
                        scores = scores1.copy()

                        detection_box = detection_box_ori[max_idx]



            ## ====================================================
            if score_max < 0:
                search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0

                cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                                   mean_rgb=128)
                cur_img_array = np.array(cropped_img1)
                detection_box_ori1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                       feed_dict={self.input_cur_image: cur_img_array,
                                                                  self.initConstantOp: self.init_feature_maps})
                if scores1[0, 0] > 0.8:
                    detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
                    detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
                    detection_box_ori = detection_box_ori1.copy()
                    # max_idx = 0
                    search_box1 = detection_box_ori[0]

                    search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
                    search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
                    search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
                    search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
                    if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                        score_max = -1
                    else:
                        search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                                       search_box1[2] - search_box1[0]]
                        search_box1 = np.reshape(search_box1, (1, 4))
                        search_regions = extract_regions(cur_ori_img_array, search_box1)
                        search_regions = search_regions[:, :, :, ::-1]
                        score_max = self.sess.run(self.outputsSingleOp, feed_dict={self.imageSingleOp: search_regions})
                        score_max = score_max[0, 1]

                    # search_box1 = [search_box1[1],search_box1[0],search_box1[3]-search_box1[1],search_box1[2]-search_box1[0]]
                    # search_box1 = np.reshape(search_box1, (1, 4))
                    # search_regions = extract_regions(cur_ori_img_array, search_box1)
                    # score_max = sess.run(outputsSingleOp, feed_dict={imageSingleOp: search_regions})
                    if score_max > 0:
                        scores = scores1.copy()
                        max_idx = 0
                        detection_box = detection_box_ori[max_idx]

                    if score_max < 0:
                        search_box1 = detection_box_ori[:20]
                        search_box = np.zeros_like(search_box1)
                        search_box[:, 1] = search_box1[:, 0]
                        search_box[:, 0] = search_box1[:, 1]
                        search_box[:, 2] = search_box1[:, 3]
                        search_box[:, 3] = search_box1[:, 2]
                        haha = np.ones_like(search_box[:, 2]) * 3
                        search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
                        search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
                        search_box[:, 2] = np.maximum(search_box[:, 2], haha)
                        search_box[:, 3] = np.maximum(search_box[:, 3], haha)
                        haha2 = np.zeros_like(search_box[:, 0])
                        search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
                        search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
                        haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
                        search_box[:, 0] = np.minimum(search_box[:, 0], haha)
                        haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
                        search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

                        search_regions = extract_regions(cur_ori_img_array, search_box)
                        search_regions = search_regions[:, :, :, ::-1]
                        mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
                        mdnet_scores = mdnet_scores[:, 1]
                        max_idx1 = np.argmax(mdnet_scores)
                        if mdnet_scores[max_idx1] > 0 and scores1[0,max_idx1] > 0.3:  ## judgement
                            score_max = mdnet_scores[max_idx1]
                            max_idx = max_idx1
                            scores = scores1.copy()

                            detection_box = detection_box_ori[max_idx]

            if score_max < 0:
                search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0 / 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0 / 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0 / 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0 / 2.0  ## search_gt size reduce ???

                cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                                   mean_rgb=128)
                cur_img_array = np.array(cropped_img1)
                detection_box_ori1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                       feed_dict={self.input_cur_image: cur_img_array,
                                                                  self.initConstantOp: self.init_feature_maps})
                if scores1[0, 0] > 0.8:
                    detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
                    detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
                    detection_box_ori = detection_box_ori1.copy()
                    # max_idx = 0
                    search_box1 = detection_box_ori[0]

                    search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
                    search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
                    search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
                    search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
                    if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                        score_max = -1
                    else:
                        search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                                       search_box1[2] - search_box1[0]]
                        search_box1 = np.reshape(search_box1, (1, 4))

                        search_regions = extract_regions(cur_ori_img_array, search_box1)
                        search_regions = search_regions[:, :, :, ::-1]

                        score_max = self.sess.run(self.outputsSingleOp, feed_dict={self.imageSingleOp: search_regions})
                        score_max = score_max[0, 1]

                    # search_box1 = [search_box1[1],search_box1[0],search_box1[3]-search_box1[1],search_box1[2]-search_box1[0]]
                    # search_box1 = np.reshape(search_box1, (1, 4))
                    # search_regions = extract_regions(cur_ori_img_array, search_box1)
                    # score_max = sess.run(outputsSingleOp, feed_dict={imageSingleOp: search_regions})
                    if score_max > 0:
                        scores = scores1.copy()

                        max_idx = 0
                        detection_box = detection_box_ori[max_idx]

                    if score_max < 0:
                        search_box1 = detection_box_ori[:20]
                        search_box = np.zeros_like(search_box1)
                        search_box[:, 1] = search_box1[:, 0]
                        search_box[:, 0] = search_box1[:, 1]
                        search_box[:, 2] = search_box1[:, 3]
                        search_box[:, 3] = search_box1[:, 2]
                        haha = np.ones_like(search_box[:, 2]) * 3
                        search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
                        search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
                        search_box[:, 2] = np.maximum(search_box[:, 2], haha)
                        search_box[:, 3] = np.maximum(search_box[:, 3], haha)
                        haha2 = np.zeros_like(search_box[:, 0])
                        search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
                        search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
                        haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
                        search_box[:, 0] = np.minimum(search_box[:, 0], haha)
                        haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
                        search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

                        search_regions = extract_regions(cur_ori_img_array, search_box)
                        search_regions = search_regions[:, :, :, ::-1]
                        mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
                        mdnet_scores = mdnet_scores[:, 1]
                        max_idx1 = np.argmax(mdnet_scores)
                        if mdnet_scores[max_idx1] > 0 and scores1[0,max_idx1] > 0.3:
                            score_max = mdnet_scores[max_idx1]
                            max_idx = max_idx1
                            scores = scores1.copy()

                            detection_box = detection_box_ori[max_idx]

            if score_max < 0:
                search_gt[0] = cur_ori_img.height / 2.0 - self.first_h / 2.0 * 2.0
                search_gt[2] = cur_ori_img.height / 2.0 + self.first_h / 2.0 * 2.0
                search_gt[1] = cur_ori_img.width / 2.0 - self.first_w / 2.0 * 2.0
                search_gt[3] = cur_ori_img.width / 2.0 + self.first_w / 2.0 * 2.0

                cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                                   mean_rgb=128)
                cur_img_array = np.array(cropped_img1)
                detection_box_ori1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                       feed_dict={self.input_cur_image: cur_img_array,
                                                                  self.initConstantOp: self.init_feature_maps})
                if scores1[0, 0] > 0.8:
                    detection_box_ori1[:, 0] = detection_box_ori1[:, 0] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 1] = detection_box_ori1[:, 1] * scale1[1] + win_loc1[1]
                    detection_box_ori1[:, 2] = detection_box_ori1[:, 2] * scale1[0] + win_loc1[0]
                    detection_box_ori1[:, 3] = detection_box_ori1[:, 3] * scale1[1] + win_loc1[1]
                    detection_box_ori = detection_box_ori1.copy()
                    # max_idx = 0
                    search_box1 = detection_box_ori[0]
                    search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
                    search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
                    search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
                    search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)

                    if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                        score_max = -1
                    else:
                        search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                                       search_box1[2] - search_box1[0]]
                        search_box1 = np.reshape(search_box1, (1, 4))

                        search_regions = extract_regions(cur_ori_img_array, search_box1)
                        search_regions = search_regions[:, :, :, ::-1]

                        score_max = self.sess.run(self.outputsSingleOp, feed_dict={self.imageSingleOp: search_regions})
                        score_max = score_max[0, 1]

                    # search_box1 = [search_box1[1],search_box1[0],search_box1[3]-search_box1[1],search_box1[2]-search_box1[0]]
                    # search_box1 = np.reshape(search_box1, (1, 4))
                    # search_regions = extract_regions(cur_ori_img_array, search_box1)
                    # score_max = sess.run(outputsSingleOp, feed_dict={imageSingleOp: search_regions})
                    if score_max > 0:
                        max_idx = 0
                        scores = scores1.copy()

                        detection_box = detection_box_ori[max_idx]

                    if score_max < 0:
                        search_box1 = detection_box_ori[:20]
                        search_box = np.zeros_like(search_box1)
                        search_box[:, 1] = search_box1[:, 0]
                        search_box[:, 0] = search_box1[:, 1]
                        search_box[:, 2] = search_box1[:, 3]
                        search_box[:, 3] = search_box1[:, 2]
                        haha = np.ones_like(search_box[:, 2]) * 3
                        search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
                        search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
                        search_box[:, 2] = np.maximum(search_box[:, 2], haha)
                        search_box[:, 3] = np.maximum(search_box[:, 3], haha)
                        haha2 = np.zeros_like(search_box[:, 0])
                        search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
                        search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
                        haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
                        search_box[:, 0] = np.minimum(search_box[:, 0], haha)
                        haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
                        search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

                        search_regions = extract_regions(cur_ori_img_array, search_box)
                        search_regions = search_regions[:, :, :, ::-1]
                        mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
                        mdnet_scores = mdnet_scores[:, 1]
                        max_idx1 = np.argmax(mdnet_scores)
                        if mdnet_scores[max_idx1] > 0 and scores1[0,max_idx1] > 0.3:
                            score_max = mdnet_scores[max_idx1]
                            max_idx = max_idx1
                            scores = scores1.copy()

                            detection_box = detection_box_ori[max_idx]

        ##--------------------------------------------------------------------------------------





        #print scores[0,max_idx]
        if scores[0, max_idx] < 0.3:
            last_reliable_w = self.first_w
            last_reliable_h = self.first_h
            count_research = 0
            isfind = 0
            #print cur_ori_img.width / 2.0 / last_reliable_w, cur_ori_img.height/2.0/last_reliable_h
            while count_research < 500 and (self.startx < cur_ori_img.width + 2 * last_reliable_w - 1) and self.starty < cur_ori_img.height + 2 * last_reliable_h - 1:
                # startx = 4*last_reliable_w + startx
                count_research += 1

                ## MAYBE the global-wide search, noting !!!!
                search_gt = np.int32(
                    [self.starty - last_reliable_h / 2.0, self.startx - last_reliable_w / 2.0, self.starty + last_reliable_h / 2.0,
                     self.startx + last_reliable_w / 2.0])
                cropped_img1, last_gt_norm1, win_loc1, scale1 = crop_search_region(cur_ori_img, search_gt, 300,
                                                                                   mean_rgb=128)
                cur_img_array1 = np.array(cropped_img1)
                detection_box1, scores1 = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                        feed_dict={self.input_cur_image: cur_img_array1,
                                                                   self.initConstantOp: self.init_feature_maps})
                #print scores1[0,0]

                if scores1[0, 0] > 0.5:
                    detection_box1[:, 0] = detection_box1[:, 0] * scale1[0] + win_loc1[0]
                    detection_box1[:, 1] = detection_box1[:, 1] * scale1[1] + win_loc1[1]
                    detection_box1[:, 2] = detection_box1[:, 2] * scale1[0] + win_loc1[0]
                    detection_box1[:, 3] = detection_box1[:, 3] * scale1[1] + win_loc1[1]
                    detection_box_ori = detection_box1.copy()
                    # max_idx = 0
                    search_box1 = detection_box_ori[0]
                    search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
                    search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
                    search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
                    search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)
                    if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]):
                        score_max = -1
                    else:
                        search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
                                       search_box1[2] - search_box1[0]]
                        search_box1 = np.reshape(search_box1, (1, 4))

                        search_regions = extract_regions(cur_ori_img_array, search_box1)
                        search_regions = search_regions[:, :, :, ::-1]

                        score_max = self.sess.run(self.outputsSingleOp,
                                                  feed_dict={self.imageSingleOp: search_regions})
                        score_max = score_max[0, 1]
                    if score_max > 0:
                        scores = scores1.copy()
                        max_idx = 0
                        self.startx = 0
                        self.starty = 0

                        detection_box = detection_box_ori[max_idx]
                        break

                    if score_max < 0:
                        search_box1 = detection_box_ori[:20]
                        search_box = np.zeros_like(search_box1)
                        search_box[:, 1] = search_box1[:, 0]
                        search_box[:, 0] = search_box1[:, 1]
                        search_box[:, 2] = search_box1[:, 3]
                        search_box[:, 3] = search_box1[:, 2]
                        haha = np.ones_like(search_box[:, 2]) * 3
                        search_box[:, 2] = search_box[:, 2] - search_box[:, 0]
                        search_box[:, 3] = search_box[:, 3] - search_box[:, 1]
                        search_box[:, 2] = np.maximum(search_box[:, 2], haha)
                        search_box[:, 3] = np.maximum(search_box[:, 3], haha)
                        haha2 = np.zeros_like(search_box[:, 0])
                        search_box[:, 0] = np.maximum(search_box[:, 0], haha2)
                        search_box[:, 1] = np.maximum(search_box[:, 1], haha2)
                        haha = np.ones_like(search_box[:, 2]) * cur_ori_img.width - 1 - search_box[:, 2]
                        search_box[:, 0] = np.minimum(search_box[:, 0], haha)
                        haha2 = np.ones_like(search_box[:, 3]) * cur_ori_img.height - 1 - search_box[:, 3]
                        search_box[:, 1] = np.minimum(search_box[:, 1], haha2)

                        search_regions = extract_regions(cur_ori_img_array, search_box)
                        search_regions = search_regions[:, :, :, ::-1]
                        mdnet_scores = self.sess.run(self.outputsOp, feed_dict={self.imageOp: search_regions})
                        mdnet_scores = mdnet_scores[:, 1]
                        max_idx1 = np.argmax(mdnet_scores)
                        score_max = mdnet_scores[max_idx1]
                        if mdnet_scores[max_idx1] > 0 and scores1[0, max_idx1] > 0.5:
                            scores = scores1.copy()
                            max_idx = max_idx1
                            detection_box = detection_box_ori[max_idx]
                            self.startx = 0
                            self.starty = 0
                            break

                ## here we update the self.starty
                self.starty = 2 * last_reliable_h + self.starty
                if self.starty >= cur_ori_img.height + 2 * last_reliable_h - 1 and self.startx < cur_ori_img.width + 2* last_reliable_w-1:
                    self.starty = 0
                    ## here we update the self.startx
                    self.startx = 2 * last_reliable_w + self.startx

            if self.startx >= cur_ori_img.width + 2* last_reliable_w-1:
                self.startx = 0
                self.starty = 0

            ## now we finish the global-image search


        ##--------------------------------------------------------------------------------------

        ## start to store the postive sample(whose score_max > self.first_score/2.0) ---> for online training
        if scores[0, max_idx] > 0.5 and score_max > self.first_score/2.0:
            gt_tmp = np.array([detection_box[1], detection_box[0], detection_box[3] - detection_box[1],
                               detection_box[2] - detection_box[0]])
            pos_examples1 = gen_samples(SampleGenerator('gaussian', cur_ori_img.size, 0.1, 1.2), gt_tmp, 50, [0.7, 1])
            pos_regions1 = extract_regions(cur_ori_img_array, pos_examples1)
            pos_regions1 = pos_regions1[:, :, :, ::-1]
            # neg_examples2 = np.zeros((50,4))
            # count = 0
            # t = 0
            # while count < 50 and t < 100:
            #     x1 = max(detection_box[1], detection_box_ori[t,1])
            #     y1 = max(detection_box[0],detection_box_ori[t,0])
            #     x2 = min(detection_box[3],detection_box_ori[t,3])
            #     y2 = min(detection_box[2],detection_box_ori[t,2])
            #     tmp1 = (x2-x1)*(y2-y1)
            #     tmp = tmp1 / float((detection_box[2]-detection_box[0])*(detection_box[3]-detection_box[1]) + (detection_box_ori[t,2]-detection_box_ori[t,0]) * (detection_box_ori[t,3]-detection_box_ori[t,1]) - tmp1)
            #     if tmp < 0.5 and (detection_box_ori[t,3]-detection_box_ori[t,1]) > 0 and (detection_box_ori[t,2] - detection_box_ori[t,0]) > 0:
            #         neg_examples2[count,0] = detection_box_ori[t,1]
            #         neg_examples2[count,1] = detection_box_ori[t,0]
            #         neg_examples2[count,2] = detection_box_ori[t,3] - detection_box_ori[t,1]
            #         neg_examples2[count,3] = detection_box_ori[t,2] - detection_box_ori[t,0]
            #         if neg_examples2[count,0] < 0:
            #             neg_examples2[count,0] = 0
            #         if neg_examples2[count,1] < 0:
            #             neg_examples2[count,1] = 0
            #         if neg_examples2[count,2] < 1:
            #             neg_examples2[count,2] = 1
            #         if neg_examples2[count,3] < 1:
            #             neg_examples2[count,3] = 1
            #         if neg_examples2[count,0] > cur_ori_img.width-1-neg_examples2[count,2]:
            #             neg_examples2[count,0] = cur_ori_img.width-1-neg_examples2[count,2]
            #         if neg_examples2[count,1] > cur_ori_img.height-1-neg_examples2[count,3]:
            #             neg_examples2[count,1] = cur_ori_img.height-1-neg_examples2[count,3]
            #         count += 1
            #
            #     t+=1
            #
            # if count < 50:
            neg_examples2 = gen_samples(SampleGenerator('uniform', cur_ori_img.size, 1.5, 1.2), gt_tmp, 200, [0, 0.5])
            # neg_examples2 = np.concatenate((neg_examples1,neg_examples2), axis=0)
            neg_regions1 = extract_regions(cur_ori_img_array, neg_examples2)
            neg_regions1 = neg_regions1[:, :, :, ::-1]

            ## sampling the pos/neg samples around the [detection_box]

            tmp_regions = np.concatenate((pos_regions1, neg_regions1, neg_regions1[:6]), axis=0)
            # pdb.set_trace()
            feat1 = self.sess.run(self.featOp, feed_dict={self.imageOp1: tmp_regions})
            pos_feat1 = feat1[:50, :, :, :]
            neg_feat1 = feat1[50:250, :, :, :]
            self.pos_feat_record = np.concatenate((self.pos_feat_record, pos_feat1), axis=0)
            self.neg_feat_record = np.concatenate((self.neg_feat_record, neg_feat1), axis=0) ## store the feature
            if self.pos_feat_record.shape[0] > 250 + 1:
                self.pos_feat_record = self.pos_feat_record[50:, :, :, :]
                self.neg_feat_record = self.neg_feat_record[200:, :, :, :]
                ## here just like:
                ## keeping the num of self.pos_feat_record --> 250, num of self.neg_feat_record --> (250/50*400)
                ## when reach the up bound, then throw the earlest 50 pos samples / 200 neg samples

        neg_feat_last = []
        hard_pos_last = []
        if np.mod(self.i, 10) == 0:  ## each i step, just run online-update once
            for iter in range(15):

                pos_feat = np.random.randint(0, self.pos_feat_record.shape[0], 50)
                pos_feat = self.pos_feat_record[pos_feat]
                if len(neg_feat_last) > 0:
                    ## the hardest negative samples
                    neg_feat = np.random.randint(0, self.neg_feat_record.shape[0], 200 - neg_feat_last.shape[0])
                    neg_feat = self.neg_feat_record[neg_feat]
                    neg_feat = np.concatenate((neg_feat_last, neg_feat), axis=0)
                else:
                    neg_feat = np.random.randint(0, self.neg_feat_record.shape[0], 200)
                    neg_feat = self.neg_feat_record[neg_feat]

                featInputs = np.concatenate((pos_feat, neg_feat), axis=0)
                ## online-training
                _, loss1, logits1 = self.sess.run([self.trainVGGMOp, self.lossOp, self.logitsOp],
                                             feed_dict={self.featInputOp: featInputs, self.labelOp: self.labels, self.lrOp: 0.0002})
                logits2 = self.sess.run(self.logitsOp, feed_dict={self.featInputOp: featInputs})

                hard_neg = np.argsort(-logits2[50:, 1])
                neg_feat_last = featInputs[50:, :, :, :]
                neg_feat_last = neg_feat_last[hard_neg[:30], :, :, :]## get the hardest 30 neg samples



        ##--------------------------------------------------------------------------------------



        if scores[0, max_idx] < 0.3:
            x_c = (detection_box[3] + detection_box[1]) / 2.0
            y_c = (detection_box[0] + detection_box[2]) / 2.0
            w1 = self.last_gt[3] - self.last_gt[1]
            h1 = self.last_gt[2] - self.last_gt[0]
            x1 = x_c - w1 / 2.0
            y1 = y_c - h1 / 2.0
            x2 = x_c + w1 / 2.0
            y2 = y_c + h1 / 2.0
            self.last_gt = np.float32([y1, x1, y2, x2])
            ## keep the center, and use w,h of last_gt  (cannot understand)
        else:
            self.last_gt = detection_box
            self.target_w = detection_box[3] - detection_box[1]
            self.target_h = detection_box[2] - detection_box[0]

        ## y_min x_min y_max x_max
        if self.last_gt[0] < 0:
            self.last_gt[0] = 0
            self.last_gt[2] = self.target_h
        if self.last_gt[1] < 0:
            self.last_gt[1] = 0
            self.last_gt[3] = self.target_w
        if self.last_gt[2] > cur_ori_img.height:
            self.last_gt[2] = cur_ori_img.height - 1
            self.last_gt[0] = cur_ori_img.height - 1 - self.target_h
        if self.last_gt[3] > cur_ori_img.width:
            self.last_gt[3] = cur_ori_img.width - 1
            self.last_gt[1] = cur_ori_img.width - 1 - self.target_w


        self.target_w = (self.last_gt[3] - self.last_gt[1])
        self.target_h = (self.last_gt[2] - self.last_gt[0])


        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        ## show the detection result
        show_res(image, np.array(self.last_gt, dtype=np.int32), 'MBMD Tracker', score=scores[0,max_idx],score_max=score_max)

        ## fuck ???
        if scores[0,max_idx] > 0.5 and score_max > 0:
            confidence_score = 0.99
        elif scores[0,max_idx] < 0.3 and score_max < 0:
            confidence_score = np.nan
        elif score_max > 20.0:
            confidence_score = 0.99
        else:
            confidence_score = scores[0,max_idx]   ## <-- mdzz cnm

        return vot.Rectangle(float(self.last_gt[1]), float(self.last_gt[0]), float(width), float(height)),confidence_score#scores[0,max_idx]

import time


handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
print len(handle)
print image.shape
print "init img: {}".format(imagefile)
t1 = time.time()
tracker = MobileTracker(image,selection)
t2 = time.time()
print "\ntracker initial finish, time cost {}s\n".format(t2-t1)

t3=time.time()
while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    print (imagefile)
    image = cv2.imread(imagefile)
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
t4=time.time()
print "finish, time cost {}s per img".format((t4-t3)/len(handle))
