from __future__ import print_function,division

import os
import cv2
import glob
import numpy as np
from PIL import Image
from data_utils import crop_search_region,draw_box,transform

import torch
import torch.utils.data as data

def convert_to_one_hot(y, C):
    y = np.array(y)
    return np.eye(C)[y.reshape(-1)]


## otb raw gt : [x y w h] & (x,y) in low corner
class otb_dataset(data.Dataset):
    def __init__(self,data_rootpath, num_seq=1, frame_interval=10, template_size=128, img_size=300):
        self.data_rootpath = data_rootpath
        self.num_seq = num_seq
        self.template_size = template_size
        self.img_size = img_size
        self.frame_interval = frame_interval
        ## two choice: num_seq=1, frame_interval=10 or num_seq=10, frame_interval=1

        self.seq_list = sorted(glob.glob(data_rootpath + '/*'))
        for i, seq in enumerate(self.seq_list):
            self.seq_list[i] = os.path.basename(seq)

        # print seq_list
        self.seq_img_dict = dict()
        self.gt_dict = dict()
        for seq in self.seq_list:
            #print seq
            img_list = sorted(glob.glob(os.path.join(self.data_rootpath, seq) + '/img/*.jpg'))
            self.seq_img_dict[seq] = img_list

            with open(os.path.join(self.data_rootpath, seq, 'groundtruth_rect.txt'), 'r') as fl:
                lines = fl.readlines()

                if len(lines[0].rstrip().split(',')) > 1:
                    for i, line in enumerate(lines):
                        lines[i] = line.rstrip().split(',')
                else:
                    for i, line in enumerate(lines):
                        lines[i] = line.rstrip().split()

                lines = np.array(lines, dtype=np.float32).astype(np.int)
                lines[:,2] = lines[:,0] + lines[:,2]
                lines[:,3] = lines[:,1] + lines[:,3]


                self.gt_dict[seq] = lines

    def draw_gtbox(self,seq,idx):
        img = Image.open(self.seq_img_dict[seq][idx])
        gt_xywh = self.gt_dict[seq][idx]
        x1, y1, x2, y2 = gt_xywh[0] , \
                         gt_xywh[1] , \
                         gt_xywh[2] , \
                         gt_xywh[3]
        gt = [x1,y1,x2,y2]
        draw_box(img,gt,img_path='/home/yuzhe/tmp/4/{}_{}.jpg'.format(seq,idx))

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        #print (seq)
        seq_imgs = self.seq_img_dict[seq]
        seq_gtboxs = self.gt_dict[seq]
        num_frames = len(seq_imgs)
        assert len(seq_imgs) == len(seq_gtboxs)

        if num_frames >= self.num_seq:
            start_id = np.random.randint(0,num_frames - (self.num_seq)*self.frame_interval)
            frame_ids = np.linspace(start_id, start_id+(self.num_seq)*self.frame_interval, self.num_seq + 1).astype(np.int)
        else:
            frame_ids = np.random.randint(0, num_frames, self.num_seq).astype(np.int)

        first_gt = seq_gtboxs[frame_ids[0]]
        img_list, gt_list = [], []

        for i,id in enumerate(frame_ids):
            img = Image.open(seq_imgs[id])
            #print (img.size)
            #print (np.array(img).shape)
            if not (np.array(img).ndim)==3:
                img_array = np.expand_dims(np.array(img),axis=2)
                img_array = np.tile(img_array,(1,1,3))
                img = Image.fromarray(img_array)
            gt = seq_gtboxs[id]

            if i == 0:
                template = img.crop(first_gt)
                template = template.resize([self.template_size,self.template_size])
            else:
                last_gt = seq_gtboxs[id - 1]
                search_region, win_loc, scaled = crop_search_region(img, last_gt, self.img_size)
                img_list.append(np.array(search_region))
                gt_search_region = [(gt[0] - win_loc[0]) / scaled[0],
                                    (gt[1] - win_loc[1]) / scaled[1],
                                    (gt[2] - win_loc[0]) / scaled[0],
                                    (gt[3] - win_loc[1]) / scaled[1],]
                gt_list.append(np.array(gt_search_region))

        ## Test: the gt box in search region
        if None:
            n = len(gt_list)
            for i in range(n):
                img1 = Image.fromarray(img_list[i])
                gt1 = gt_list[i]
                print (gt1)
                print (np.array(gt1).astype(np.int))
                draw_box(img1,np.array(gt1).astype(np.int),img_path='/home/yuzhe/tmp/{}.jpg'.format(i))

        ## here transform and normalize
        template = np.array(template).transpose((1,0,2)) ##transpose to (W,H,3)
        template = transform(template)
        for i,im in enumerate(img_list):
            img_list[i] = transform(im)
        seq_input_img = np.stack(img_list,axis=0).transpose((0,2,1,3)) ##transpose to (num_seq,W,H,3)
        seq_input_gt = np.stack(gt_list,axis=0)

        return template,seq_input_gt, seq_input_img


def otb_collate(batch):
    img_batch = []
    gt_batch = []
    template_batch = []

    for sample in batch:
        template_batch.append(sample[0])
        img_batch.append(sample[2])
        gt_batch.append(sample[1])

    img_batch = np.stack(img_batch, axis=0)
    gt_batch = np.stack(gt_batch, axis=0)
    template_batch = np.expand_dims(np.stack(template_batch,axis=0),axis=1)
    batchsize,num_seq,_ = gt_batch.shape
    label_batch = np.ones((batchsize,num_seq,1),dtype=np.int)

    return template_batch,img_batch,gt_batch,label_batch

########################################################################



if __name__ == '__main__':
    test_loader = otb_dataset('/home/yuzhe/Downloads/part_vot_seq/')
    data_loader = torch.utils.data.DataLoader(test_loader,
                                              batch_size=4, shuffle=True, num_workers=1, collate_fn=otb_collate)
    # a = test_loader[7]
    # print (len(test_loader))

    for idx, (templates, imgs, gts, labels) in enumerate(data_loader):
        print(templates.shape)
        print (imgs.shape)
        print(gts.shape)
        print(labels.shape)
        # print(imgs[3][0].dtype)
        # img1 = Image.fromarray(imgs[3][0].transpose((1,0,2)))
        # gt1 = gts[3][0]
        # draw_box(img1, gt1, img_path='/home/yuzhe/tmp/{}.jpg'.format(idx))


        # seq = test_loader.seq_list[10]
        # num_seq = len(test_loader.label_dict['Biker'])
        # for i in range(num_seq):
        #    test_loader.draw_gtbox('Biker', i)



