#!/usr/bin/python3.7

import numpy as np
import multiprocessing as mp
import queue
import glob,os
from utils.dataset import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi


class Inference:
    ### helper function for parallelized Viterbi decoding ##########################

    def online_decode(self,queue, log_probs, decoder, index2label):
        while not queue.empty():
            try:
                video = queue.get(timeout = 3)
                stn_score, stn_labels, stn_segments = decoder.decode_online(log_probs[video])
                with open('{}/results/Predictions/'.format(self.split) + video, 'w') as f:
                    f.write( '### Recognized sequence: ###\n' )
                    f.write( ' '.join( [index2label[s.label] for s in stn_segments] ) + '\n' )
                    f.write( '### Score: ###\n' + str(stn_score) + '\n')
                    f.write( '### Frame level recognition: ###\n')
                    f.write( ' '.join( [index2label[l] for l in stn_labels] ) + '\n' )
            except queue.Empty:
                pass


    def main(self,path,split,load_iteration):
        self.split=split
        print("doing inference at iteration {}".format(load_iteration))
        ### read label2index mapping and index2label mapping ###########################
        label2index = dict()
        index2label = dict()
        with open(path+'mapping.txt', 'r') as f:
            content = f.read().split('\n')[0:-1]
            for line in content:
                label2index[line.split()[1]] = int(line.split()[0])
                index2label[int(line.split()[0])] = line.split()[1]

        ### read test data #############################################################
        with open(path+'split{}.test'.format(split), 'r') as f:
            video_list = f.read().split('\n')[0:-1]
        dataset = Dataset(path, video_list, label2index, shuffle = False,multiview=False)


        log_prior=np.log( np.loadtxt('{}/results/prior.iter-'.format(split) + str(load_iteration) + '.txt') )
        grammar = PathGrammar('{}/results/grammar.txt'.format(split), label2index)
        length_model = PoissonModel('{}/results/lengths.iter-'.format(split) + str(load_iteration) + '.txt', max_length=2000)
        forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
        forwarder.load_model('{}/results/network.iter-'.format(split) + str(load_iteration) + '.net')

        # parallelization
        n_threads = 4
        viterbi_decoder = Viterbi(grammar, length_model, frame_sampling=30)
        #get intermediate grammar

        # forward each video
        log_probs = dict()
        queue = mp.Queue()
        for i, data in enumerate(dataset):
            sequence,_, _ ,_,_,_= data
            video = list(dataset.features.keys())[i]
            queue.put(video)
            outputGRU, log_probs_GRU = forwarder.forward(sequence, sequence)
            log_probs[video] =log_probs_GRU.data.cpu().numpy()- log_prior
            log_probs[video] = log_probs[video] - np.max(log_probs[video])

        # Viterbi decoding
        procs = []


        filelist = glob.glob('{}/results/Predictions'.format(split) + '/P*')
        filelist.sort()
        if len(filelist)!=0:
            print("DELETING {} PREVIOUS FILES".format(len(filelist)))
            for file in filelist:
                os.remove(file)
        filelist = glob.glob('{}/results/Predictions'.format(split) + '/P*')
        assert len(filelist)==0
        for i in range(n_threads):
            p = mp.Process(target=self.online_decode, args=(queue, log_probs, viterbi_decoder, index2label))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

