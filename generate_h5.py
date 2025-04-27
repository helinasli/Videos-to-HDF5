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
        # Preprocessing
        frame = cv2.resize(frame, (224, 224))
        if not hasattr(self, 'prev_features'):
            self.prev_features = None
            
        if self.model_name=="resnet":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_pool5 = self.model(frame)
            frame_feat = res_pool5.cpu().data.numpy().flatten()
            
            # Lighter temporal smoothing (0.9 instead of 0.7)
            if self.prev_features is not None:
                frame_feat = 0.9 * frame_feat + 0.1 * self.prev_features
            self.prev_features = frame_feat
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
            frame_feat = frame_feat.cpu().data.numpy().flatten()
        #print('frame feat', frame_feat.shape)
        return frame_feat

    # CHANGE POINTS HERE -> cpd_auto don't work
    
    def _get_change_points(self, video_feat, n_frame, fps):
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        
        # Stage 1: Get initial change points with moderate parameters
        K = np.dot(video_feat, video_feat.T)
        lmin = int(fps * 2)  # 2 seconds minimum
        vmax = 0.6  # Moderate penalty
        
        change_points, scores = cpd_auto(K, m, vmax, lmin)
        
        # Stage 2: Filter change points based on their significance
        if len(change_points) > 0:
            # Calculate segment differences
            diffs = []
            change_points = np.concatenate(([0], change_points, [n_frame-1]))
            
            for i in range(len(change_points)-1):
                start = change_points[i]
                end = change_points[i+1]
                
                # Calculate feature difference between segments
                if i < len(change_points)-2:
                    next_end = change_points[i+2]
                    curr_feat = np.mean(video_feat[start:end], axis=0)
                    next_feat = np.mean(video_feat[end:next_end], axis=0)
                    diff = np.linalg.norm(curr_feat - next_feat)
                    diffs.append(diff)
            
            if diffs:
                # Keep only significant changes (above mean difference)
                threshold = np.mean(diffs) + 0.5 * np.std(diffs)
                significant_changes = [0]  # Always keep first point
                
                for i, diff in enumerate(diffs):
                    if diff > threshold:
                        significant_changes.append(change_points[i+1])
                
                significant_changes.append(n_frame-1)  # Always keep last point
                change_points = np.array(significant_changes)
        
        # Format output
        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]
            temp_change_points.append(segment)
        
        return np.array(temp_change_points)


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

            change_points = self._get_change_points(video_feat, n_frames, fps)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, help="model to use")
    parser.add_argument("--data", type=str, required=True, help="the path of the folder video data")
    parser.add_argument("--out", type=str, required=True, help="the output file .h5")
    args = parser.parse_args()
    gen = Generate_Dataset(args.model_name,args.data, args.out)

    gen.generate_dataset()
    gen.h5_file.close()

