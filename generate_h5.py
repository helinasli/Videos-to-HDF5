"""
    Generate Dataset
    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )
"""
import os, sys
import argparse
from PIL import Image

from torchvision import transforms
import torch
sys.path.append('../')

import networks
from networks.CNN import ResNet
from KTS1.cpd_auto import cpd_auto
from tqdm import tqdm
#from googlenet_pytorch import GoogLeNet
import math
import cv2
import torchvision
#import hub
import numpy as np
import h5py


class Generate_Dataset:
    def __init__(self,model_name, video_path, save_path):
        #self.resnet = ResNet()
        self.model_name=model_name
        if self.model_name=="resnet":
            self.model=ResNet()
        else :
            #self.model=GoogLeNet()
            pass
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.frame_root_path = 'frame1'
        self.h5_file = h5py.File(save_path, 'w')

        self._set_video_list(video_path)

    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
            self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))


    def _extract_feature(self, frame):

        if self.model_name=="resnet":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            res_pool5 = self.model(frame)
            frame_feat = res_pool5.cpu().data.numpy().flatten()
        else:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            frame = Image.fromarray(frame)
            input_tensor = preprocess(frame)
            input_batch = input_tensor.unsqueeze(0)
            #print("\n\n\n")
            #print(input_batch.shape)
            #print("\n\n\n")
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                self.model.to('cuda')
             

            with torch.no_grad():
                frame_feat = self.model(input_batch)
                frame_feat=frame_feat[0]
        return frame_feat


        

    def _get_change_points(self, video_feat, n_frame, fps):
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0], change_points, [n_frame-1]))

        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]

            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))

        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0]
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return change_points, n_frame_per_seg

    # TODO : save dataset
    def _save_dataset(self):
        pass

    def generate_dataset(self):
        for video_idx, video_filename in enumerate(tqdm(self.video_list)):
            video_path = video_filename
            print(video_filename)
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_basename = os.path.basename(video_path).split('.')[0]

            if not os.path.exists(os.path.join(self.frame_root_path, video_basename)):
                os.makedirs(os.path.join(self.frame_root_path, video_basename))

            video_capture = cv2.VideoCapture(video_path)
            video_capture.set(cv2.CAP_PROP_FPS, 25)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_list = []
            picks = []
            video_feat = None
            video_feat_for_train = None
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                if success:
                    frame_feat = self._extract_feature(frame)
                    #print(frame_feat.shape)
                    if frame_idx % 15 == 0:
                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))

                    if video_feat is None:
                        video_feat = frame_feat
                    else:
                        video_feat = np.vstack((video_feat, frame_feat))

                    img_filename = "{}.jpg".format(str(frame_idx).zfill(5))
                    cv2.imwrite(os.path.join(self.frame_root_path, video_basename, img_filename), frame)

                else:
                    break

            video_capture.release()

            change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps)

            # self.dataset['video_{}'.format(video_idx+1)]['frames'] = list(frame_list)
            # self.dataset['video_{}'.format(video_idx+1)]['features'] = list(video_feat)
            # self.dataset['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            # self.dataset['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            # self.dataset['video_{}'.format(video_idx+1)]['fps'] = fps
            # self.dataset['video_{}'.format(video_idx+1)]['change_points'] = change_points
            # self.dataset['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg
            
            
            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, help="model to use")
    parser.add_argument("--data", type=str, required=True, help="the path of the folder video data")
    parser.add_argument("--out", type=str, required=True, help="the output file .h5")
    args = parser.parse_args()
    gen = Generate_Dataset(args.model_name,args.data, args.out)

    gen.generate_dataset()
    gen.h5_file.close()

