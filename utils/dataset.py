#!/usr/bin/python3.7
#References:
# https://github.com/JunLi-Galios/CDFL
#https://github.com/alexanderrichard/NeuralNetwork-Viterbi
import numpy as np
import random

random.seed(0)
np.random.seed(0)


class Dataset(object):

    def __init__(self, base_path, video_list, label2index, shuffle = False,multiview=False):
        self.multiview=multiview
        self.features = dict()
        self.transcript = dict()
        self.gt=dict()
        self.transcript2duration=dict()
        self.shuffle = shuffle
        self.idx = 0
        self.vid2view=self.organize_views(video_list)
        # read features for each video
        for video in video_list:
            self.features[video] = np.load(base_path + '/features/' + video + '.npy')
            # transcript
            with open(base_path + 'transcripts/' + video + '.txt') as f:
                self.transcript[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
            with open(base_path + 'groundTruth/' + video + '.txt') as f:
                self.gt[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]



        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        if self.shuffle:
            random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)




    def videos(self):
        return list(self.features.keys())

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            video2=self.get_otherview(video)
            if self.multiview and video2!=None:
                features1,features2,transcript,gt=self.synch_views(video,video2)
                return features1,features2,transcript,gt,video,video2
            else:
                return self.features[video], None, self.transcript[video], self.gt[video], video,video

    def get(self):
        try:
            return next(self)
        except StopIteration:
            return self.get()

    def organize_views(self,all_vids):
        all_vids.sort()
        vid2views = {}
        for video in all_vids:
            P = video.split('_')[0]
            D= video.split('_')[-1]
            if P not in vid2views:
                vid2views[P] = {}
            if D not in vid2views[P]:
                vid2views[P][D]=[]
            vid2views[P][D].append(video)

        return vid2views

    def get_otherview(self,query):
        video=query
        person = query.split('_')[0];
        dish = query.split('_')[-1]
        if len(self.vid2view[person][dish])>1:
            while query==video:
                random.shuffle(self.vid2view[person][dish])
                video=self.vid2view[person][dish][0]
            return video
        else:
            return None

    def synch_views(self,video1,video2):
        l1=min(len(self.gt[video1]),self.features[video1].shape[1])
        l2 = min(len(self.gt[video2]),self.features[video2].shape[1])
        l=min(l1,l2)
        features1=self.features[video1][:,:l]
        features2 = self.features[video2][:, :l]
        gt=self.gt[video1][:l]
        transcript=[gt[0]]
        for t in range(1,l):
            if gt[t] != gt[t - 1]:
                transcript.append(gt[t])

        if len(transcript)==len(self.transcript[video1]):assert np.sum(np.asarray(transcript)!=np.asarray(self.transcript[video1]))==0

        return features1,features2,transcript,gt


