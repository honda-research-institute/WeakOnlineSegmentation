#!/usr/bin/python

import argparse
import glob
import os
import re
import numpy as np

class Evaluation:

    def segment_intervals(self,Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        return intervals

    def segment_labels(self,Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
        return Yi_split

    def read_predictions(self,file, dataset="Breakfast"):
        with open(file, 'r') as file1:
            lines = file1.read().split('\n')[0:-1]
        pred = dict()
        for line in lines:
            if dataset == "Breakfast":
                video = line.split(',')[0]
                start = int(line.split(',')[1])
                end = int(line.split(',')[2])
                action = int(line.split(',')[3])
            if dataset == "IKEA":
                task = line.split(',')[0]
                cam = line.split(',')[1]
                session = line.split(',')[2]
                start = int(line.split(',')[3])
                end = int(line.split(',')[4])
                action = int(line.split(',')[5])
                video = ','.join([task, cam, session])
            b = np.ones([1, end - start + 1], dtype=int) * action
            if video not in pred:
                pred[video] = b
            else:
                pred[video] = np.concatenate((pred[video], b), axis=1)

        return pred

    def read_groundtruth(self,gt_path, pred, label2index):
        gt = dict()
        for video in pred:
            with open(gt_path + 'groundTruth/' + video + '.txt') as f:
                gt[video] = [label2index[line] for line in f.read().split('\n')[0:-1]]
        return gt

    def post_process(self,pred, y):
        P = []
        Y = []
        V = []
        for video in pred:
            L1 = len(pred[video][0])
            L2 = len(y[video])
            l = min(L1, L2)
            P.append(pred[video][0, :l])
            V.append(video)
            # yy=np.asarray(y[video][::2])
            # Y.append(yy[:l])
            Y.append(np.asarray(y[video][:l]))
        return P, Y, V

    def accuracy(self,P, Y):
        def acc_(p, y):
            res = np.mean(p == y) * 100
            return res

        if type(P) == list:
            return np.mean([np.mean(P[i] == Y[i]) for i in range(len(P))]) * 100
        else:
            return acc_(P, Y)

    def accuracy_allframes(self,P, Y):
        def acc_(p, y):
            e = np.sum(p != y)
            return e, len(p)

        error = 0
        total = 0
        for i in range(len(P)):
            e, t = acc_(P[i], Y[i])
            error = e + error
            total = total + t
        return (1 - error / total) * 100

    def get_temporal_acc(self,P, Y, bg_class=0):
        def acc_(p, y):
            return np.mean(p == y) * 100

        # [15.587287304619869, 28.391340617206154, 35.118493951673436, 38.363352097568288, 32.744900973126974]
        # [14.947818145215898, 27.286451830579992, 34.675199619186785, 37.033558667051018, 32.579645497406496]
        def acc_w(p, y, bg_class=None):
            ind = y != bg_class
            res = np.mean(p[ind] == y[ind]) * 100
            return res

        histogram = [[] for k in range(5)]
        for i in range(len(P)):
            L = len(P[i])
            cut = np.linspace(0, L, 6)[1:];
            delta = int(cut[1] - cut[0])

            for j in range(len(cut)):
                bound = int(cut[j])
                if np.sum(Y[i][max(0, 0):bound] != bg_class) == 0:
                    print
                    continue
                acc = acc_w(P[i][max(0, 0):bound], Y[i][max(0, 0):bound], bg_class)
                # acc = acc_(P[i][max(0, bound - delta):bound], Y[i][max(0, bound - delta):bound])
                histogram[j].append(acc)
        for k in range(len(cut)):
            histogram[k] = np.mean(histogram[k])
        print(histogram)

    def accuracy_wo_bg(self,P, Y, bg_class=0):
        def acc_w(p, y, bg_class=None):
            ind = y != bg_class
            res = np.mean(p[ind] == y[ind]) * 100
            return res

        if type(P) == list:
            res = [acc_w(P[i], Y[i], bg_class) for i in range(len(P))]
            return np.mean(res)
        else:
            return acc_w(P, Y)

    def IoU(self,P, Y, bg_class=0):
        # From ICRA paper:
        # Learning Convolutional Action Primitives for Fine-grained Action Recognition
        # Colin Lea, Rene Vidal, Greg Hager
        # ICRA 2016

        def overlap_(p, y, bg_class):
            true_intervals = np.array(self.segment_intervals(y))
            true_labels = self.segment_labels(y)
            pred_intervals = np.array(self.segment_intervals(p))
            pred_labels = self.segment_labels(p)

            if bg_class is not None:
                true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
                true_labels = np.array([l for l in true_labels if l != bg_class])
                pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
                pred_labels = np.array([l for l in pred_labels if l != bg_class])

            n_true_segs = true_labels.shape[0]
            n_pred_segs = pred_labels.shape[0]
            seg_scores = np.zeros(n_true_segs, np.float)

            for i in range(n_true_segs):
                for j in range(n_pred_segs):
                    if true_labels[i] == pred_labels[j]:
                        intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                             true_intervals[i][0])
                        union = max(pred_intervals[j][1], true_intervals[i][1]) - min(pred_intervals[j][0],
                                                                                      true_intervals[i][0])
                        if union == 0: print("union is zero\n")
                        score_ = float(intersection) / union
                        seg_scores[i] = max(seg_scores[i], score_)

            return seg_scores.mean() * 100

        if type(P) == list:
            return np.mean([overlap_(P[i], Y[i], bg_class) for i in range(len(P))])
        else:
            return overlap_(P, Y, bg_class)

    def IoD(self,P, Y, bg_class=0):
        # From ICRA paper:
        # Learning Convolutional Action Primitives for Fine-grained Action Recognition
        # Colin Lea, Rene Vidal, Greg Hager
        # ICRA 2016

        def overlap_d(p, y, bg_class):
            true_intervals = np.array(self.segment_intervals(y))
            true_labels = self.segment_labels(y)
            pred_intervals = np.array(self.segment_intervals(p))
            pred_labels = self.segment_labels(p)

            if bg_class is not None:
                true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
                true_labels = np.array([l for l in true_labels if l != bg_class])
                pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
                pred_labels = np.array([l for l in pred_labels if l != bg_class])

            n_true_segs = true_labels.shape[0]
            n_pred_segs = pred_labels.shape[0]
            seg_scores = np.zeros(n_true_segs, np.float)

            for i in range(n_true_segs):
                for j in range(n_pred_segs):
                    if true_labels[i] == pred_labels[j]:
                        intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                             true_intervals[i][0])
                        union = pred_intervals[j][1] - pred_intervals[j][0]
                        score_ = float(intersection) / union
                        seg_scores[i] = max(seg_scores[i], score_)

            return seg_scores.mean() * 100

        if type(P) == list:
            return np.mean([overlap_d(P[i], Y[i], bg_class) for i in range(len(P))])
        else:
            return overlap_d(P, Y, bg_class)


    def recog_file(self,filename, ground_truth_path,label2index,file1):

        # read ground truth
        gt_file = ground_truth_path + re.sub('.*/','/',filename) + '.txt'
        with open(gt_file, 'r') as f:
            ground_truth = f.read().split('\n')[0:-1]
            f.close()
        # read recognized sequence
        with open(filename, 'r') as f:
            recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
            f.close()
################

        for i in range(len(recognized)):
            if i !=0:
                if recognized[i]!=recognized[i-1]:
                    end = i - 1
                    file1.write("{},{},{},{}".format(filename.split('/')[-1],start,end,label2index[recognized[i-1]]))
                    file1.write("\n")
                    start=i
            else:
                start=0

        end = i
        file1.write("{},{},{},{}".format(filename.split('/')[-1],start,end,label2index[recognized[i]]))
        file1.write("\n")
#################

        n_frame_errors = 0
        for i in range(min(len(recognized),len(ground_truth))):

            if not recognized[i] == ground_truth[i]:
                n_frame_errors += 1

        return n_frame_errors, len(recognized)

    def main(self,path,split,iteration):
        ### MAIN #######################################################################

        ### arguments ###
        ### --recog_dir: the directory where the recognition files from inferency.py are placed
        ### --ground_truth_dir: the directory where the framelevel ground truth can be found
        label2index=dict()
        index2label = dict()
        with open(path+'mapping.txt', 'r') as f:
            content = f.read().split('\n')[0:-1]
            for line in content:
                label2index[line.split()[1]] = int(line.split()[0])
                index2label[int(line.split()[0])] = line.split()[1]


        parser = argparse.ArgumentParser()
        parser.add_argument('--recog_dir', default='results')
        parser.add_argument('--ground_truth_dir', default=path+'groundTruth')
        args = parser.parse_args()

        filelist = glob.glob('{}/results/Predictions/P*'.format(split))
        filelist.sort()
        print('Evaluate %d video files...' % len(filelist))

        n_frames = 0
        n_errors = 0
        # loop over all recognition files and evaluate the frame error
        with open("{}/results/predictions_val_{}.txt".format(split,iteration), 'w') as file1:
            for filename in filelist:
                errors, frames = self.recog_file(filename, args.ground_truth_dir,label2index,file1)
                n_errors += errors
                n_frames += frames

        # print frame accuracy (1.0 - frame error rate)

        pred_file="{}/results/predictions_val_{}.txt".format(split,iteration)
        gt_path = "./data/"
        pred = self.read_predictions(pred_file, "Breakfast")
        y = self.read_groundtruth(gt_path, pred, label2index)
        P, Y, V = self.post_process(pred, y)
        acc = self.accuracy(P, Y)
        acc_bg = self.accuracy_wo_bg(P, Y, bg_class=0)
        iou = self.IoU(P, Y, bg_class=0)
        iod = self.IoD(P, Y, bg_class=0)
        print("Background label is 0")
        print("acc: {}".format(acc))
        print("acc_bg: {}".format(acc_bg))
        print("IoU: {}".format(iou))
        print("IoD: {}".format(iod))



