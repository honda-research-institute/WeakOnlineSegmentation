#!/usr/bin/python3.7
#References:
# https://github.com/JunLi-Galios/CDFL
#https://github.com/alexanderrichard/NeuralNetwork-Viterbi
import numpy as np
from utils.dataset import Dataset
from utils.network import Trainer, Forwarder
from utils.viterbi import Viterbi
from utils.viterbi import Viterbi
from inference_online import Inference
from eval import Evaluation
import utils.options as options
import os
import argparse



args = options.parser.parse_args()
path=args.dataset_path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open(path+'mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]
for split in [args.split]:
    ### read training data #########################################################
    with open(path+'split{}.train'.format(split), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset(path, video_list, label2index, shuffle = True,multiview=False)
    ### generate path grammar for inference ########################################
    paths = set()
    for _, _,transcript,_,_,_ in dataset:
        paths.add( ' '.join([index2label[index] for index in transcript]) )
    with open('{}/results/grammar.txt'.format(split), 'w') as f:
        f.write('\n'.join(paths) + '\n')
    dataset.multiview=args.multi_view
    WPI_net=args.WPI_net
    MVI=args.MVI
    print("multiview is {} and split is {} with {} vids training for {} iterations".format(dataset.multiview,split,len(dataset),args.n_iterations))
    if dataset.multiview:
        print("The chosen Multi-View Inference Technique is {}".format(MVI))
        if MVI=="WPI":print("WPI network is {}".format(WPI_net))
    else: print("single-view training")
    decoder = Viterbi(None, None, frame_sampling=30)
    trainer = Trainer(decoder, dataset.input_dimension, dataset.n_classes,MVI,WPI_net, buffer_size = len(dataset), buffered_frame_ratio = 25)
    learning_rate = 0.01
    window = 10
    step = 5
    I = Inference()
    E = Evaluation()
    for i in range(100010):
        sequence,sequence2, transcript,gt,video,video2 = dataset.get()
        loss1, loss2,loss3 = trainer.train(sequence,sequence2, transcript,gt,video,video2,label2index, split, learning_rate=learning_rate, window=window, step=step)
        # print some progress information
        if (i+1) % 100 == 0:
            print('Iteration %d, loss1: %f, loss2: %f, loss3: %f, loss: %f' % (i+1, loss1, loss2,loss3, loss1 - loss2+loss3))
        # save model every 1000 iterations
        if (i+1) % args.n_iterations == 0 :
            print("saving at iteration {}".format(i + 1))
            network_file = '{}/results/network.iter-'.format(split) + str(i+1) + '.net'
            length_file_TAB = '{}/results/lengths.iter-'.format(split) + str(i + 1) + '.txt'
            prior_file_TAB = '{}/results/prior.iter-'.format(split) + str(i + 1) + '.txt'
            trainer.save_model(network_file,length_file_TAB,prior_file_TAB)
        # adjust learning rate after 60000 iterations
        if (i + 1) == 60000:
            learning_rate = learning_rate * 0.1

        if (i+1) % args.n_iterations==0 :
            I.main(path,split,i+1)
            E.main(path,split,i+1)
            break






