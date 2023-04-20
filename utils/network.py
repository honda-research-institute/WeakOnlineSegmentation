#!/usr/bin/python3.7
#References:
# https://github.com/JunLi-Galios/CDFL
#https://github.com/alexanderrichard/NeuralNetwork-Viterbi
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from .grammar import MultiTranscriptGrammar
from .length_model import PoissonModel


torch.manual_seed(0)
random.seed(0)


class Buffer(object):

    def __init__(self, buffer_size, n_classes):
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, features, transcript, framelabels):
        if len(self.features) < self.buffer_size:
            # sequence data 
            self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            # statistics for prior and mean lengths
            self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            # statistics for prior and mean lengths
            self.instance_counts[self.next_position] = np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size
        # update frame selectors
        self.frame_selectors = []
        for seq_idx in range(len(self.features)):
            self.frame_selectors += [ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ]


    def random(self):
        return random.choice(self.frame_selectors) # return sequence_idx and frame_idx within the sequence

    def n_frames(self):
        return len(self.frame_selectors)


# wrapper class to provide torch tensors for the network
class DataWrapper(torch.utils.data.Dataset):

    # for each frame in the sequence, create a subsequence of length window_size
    def __init__(self, sequence, window_size = 21):
        self.features = []
        self.labels = []
        # ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        # extract temporal window around each frame of the sequence
        for frame in range(sequence.shape[1]):
            left, right = max(0, frame - window_size // 2), min(sequence.shape[1], frame + 1 + window_size // 2)
            tmp = np.zeros((sequence.shape[0], window_size), dtype=np.float32 )
            tmp[:, window_size // 2 - (frame - left) : window_size // 2 + (right - frame)] = sequence[:, left : right]
            self.features.append(np.transpose(tmp))
            self.labels.append(-1) # dummy label, will be updated after Viterbi decoding

    # add a sampled (windowed frame, label) pair to the data wrapper (include buffered data during training)
    # @sequence the sequence from which the frame is sampled
    # @label the Viterbi decoding label for the frame at frame_idx
    # @frame_idx the index of the frame to sample
    def add_buffered_frame(self, sequence, label, frame_idx):
        left, right = max(0, frame_idx - self.window_size // 2), min(sequence.shape[1], frame_idx + 1 + self.window_size // 2)
        tmp = np.zeros((sequence.shape[0], self.window_size), dtype=np.float32 )
        tmp[:, self.window_size // 2 - (frame_idx - left) : self.window_size // 2 + (right - frame_idx)] = sequence[:, left : right]
        self.features.append(np.transpose(tmp))
        self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        assert idx < len(self)
        features = torch.from_numpy( self.features[idx] )
        labels = torch.from_numpy( np.array([self.labels[idx]], dtype=np.int64) )
        return features, labels


class DataWrapper_dual(torch.utils.data.Dataset):

    # for each frame in the sequence, create a subsequence of length window_size
    def __init__(self,sequence, sequence2, window_size = 21):
        self.features = [];self.features2=[]
        self.labels = [];self.full_labels=[]
        # ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        # extract temporal window around each frame of the sequence
        for frame in range(sequence.shape[1]):
            left, right = max(0, frame - window_size // 2), min(sequence.shape[1], frame + 1 + window_size // 2)
            tmp = np.zeros((sequence.shape[0], window_size), dtype=np.float32 )
            tmp[:, window_size // 2 - (frame - left) : window_size // 2 + (right - frame)] = sequence[:, left : right]
            self.features.append(np.transpose(tmp))
            self.labels.append(-1) # dummy label, will be updated after Viterbi decoding
            self.full_labels.append(-1)
            tmp = np.zeros((sequence2.shape[0], window_size), dtype=np.float32 )
            tmp[:, window_size // 2 - (frame - left) : window_size // 2 + (right - frame)] = sequence2[:, left : right]
            self.features2.append(np.transpose(tmp))


    def add_buffered_frame(self, sequence, label, frame_idx):
        left, right = max(0, frame_idx - self.window_size // 2), min(sequence.shape[1], frame_idx + 1 + self.window_size // 2)
        tmp = np.zeros((sequence.shape[0], self.window_size), dtype=np.float32 )
        tmp[:, self.window_size // 2 - (frame_idx - left) : self.window_size // 2 + (right - frame_idx)] = sequence[:, left : right]
        self.features.append(np.transpose(tmp))
        self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        assert idx < len(self)
        features = torch.from_numpy( self.features[idx] )
        features2 = torch.from_numpy(self.features2[idx])
        labels = torch.from_numpy(np.array([self.labels[idx]], dtype=np.float32))
        return features, features2,labels


class Net(nn.Module):

    def __init__(self, input_dim, hidden_size, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.gru = nn.GRU(input_dim, hidden_size, 1, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)


    def forward(self, x):
        dummy, output = self.gru(x)
        output = self.fc(output)
        return output

class Net_vc(nn.Module):

    def __init__(self, input_dim, hidden_size, n_classes):
        super(Net_vc, self).__init__()
        self.n_classes = n_classes
        self.gru = nn.GRU(input_dim, hidden_size, 1, bidirectional=False, batch_first=True)
        self.conv=nn.Conv1d(in_channels=input_dim, out_channels=hidden_size, kernel_size=10, padding=0,bias=True)

        self.fc_cmp = nn.Linear(2*hidden_size, hidden_size)
        self.fc = nn.Linear( hidden_size, n_classes)

    def forward(self, x1, x2, mode):
        if mode == "GRU":
            dummy1, output1 = self.gru(x1)
            dummy2, output2 = self.gru(x2)
        elif mode == "TC":
            x1 = x1.permute(0, 2, 1);
            x2 = x2.permute(0, 2, 1)
            output1 = nn.functional.relu(self.conv(x1))
            output2 = nn.functional.relu(self.conv(x2))
            output1 = output1.permute(0, 2, 1);
            output2 = output2.permute(0, 2, 1)
            tmp, _ = torch.topk(output1, k=1, dim=1)
            output1 = torch.mean(tmp, 1, keepdim=True)
            tmp, _ = torch.topk(output2, k=1, dim=1)
            output2 = torch.mean(tmp, 1, keepdim=True)
            output1 = output1.permute(1, 0, 2);
            output2 = output2.permute(1, 0, 2)

        two_views1 = torch.cat((output1, output2), dim=-1)
        output1 = nn.functional.relu(self.fc_cmp(two_views1))
        output1 = self.fc(output1)
        return output1

class Forwarder(object):

    def __init__(self, input_dimension, n_classes):
        self.n_classes = n_classes
        hidden_size = 64
        self.net = Net(input_dimension, hidden_size, n_classes)
        self.net.cuda()

        self.net_vc = Net_vc(input_dimension, 32, 2)
        self.net_vc.cuda()



    def _forward(self, data_wrapper, batch_size = 512):
        dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size = batch_size, shuffle = False)
        # output probability container
        output_list = []
#         offset = 0
        # forward all frames
        for data in dataloader:
            input, _ = data
            input = input.cuda()
            output = self.net(input)[0,:,:]
            output_list.append(output)
        output_GRU = torch.cat(output_list, dim=0)
        log_probs_output_GRU = nn.functional.log_softmax(output_GRU, dim=1)  # tensor is of shape (batch_size, 1, features)
        return nn.functional.relu(output_GRU),log_probs_output_GRU

    def forward(self, sequence,in_feat, batch_size = 512):
        data_wrapper = DataWrapper(sequence, window_size = 21)
        return self._forward(data_wrapper)

    def _forward_dual(self, data_wrapper,mode="GRU", batch_size=512):
        dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size=batch_size, shuffle=False)
        # output probability container
        output_list = []
        # forward all frames
        for data in dataloader:
            input,input2, _ = data
            input = input.cuda();input2 = input2.cuda()
            output = self.net_vc(input,input2,mode)
            output=output[0, :, :]
            output_list.append(output)
        output_GRU = torch.cat(output_list, dim=0)
        probs_output_GRU =nn.functional.softmax(output_GRU,dim=1)
        return probs_output_GRU

    def load_model(self, model_file):
        self.net.cpu()
        self.net.load_state_dict( torch.load(model_file) )
        self.net.cuda()



class Trainer(Forwarder):

    def __init__(self, decoder,input_dimension, n_classes, MVI="PI",WPI_net="TC", buffer_size=0,buffered_frame_ratio = 25):
        super(Trainer, self).__init__(input_dimension, n_classes)
        self.buffer = Buffer(buffer_size, n_classes)
        self.decoder = decoder
        self.buffered_frame_ratio = buffered_frame_ratio
        self.prior = np.ones((self.n_classes), dtype=np.float32) / self.n_classes
        self.mean_lengths = np.ones((self.n_classes), dtype=np.float32)
        self.WPI_net=WPI_net
        self.MVI = MVI

    def update_mean_lengths(self):
        self.mean_lengths = np.zeros( (self.n_classes), dtype=np.float32 )
        for label_count in self.buffer.label_counts:
            self.mean_lengths += label_count
        instances = np.zeros((self.n_classes), dtype=np.float32)
        for instance_count in self.buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths= np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 else sum(self.mean_lengths) / sum(instances) for i in range(self.n_classes) ] )



    def update_prior(self):
        # count labels
        self.prior = np.zeros((self.n_classes), dtype=np.float32)
        for label_count in self.buffer.label_counts:
            self.prior += label_count
        self.prior = self.prior / np.sum(self.prior)
        # backup to uniform probability for unseen classes
        n_unseen = sum(self.prior == 0)
        self.prior = self.prior * (1.0 - float(n_unseen) / self.n_classes)
        self.prior = np.array( [ self.prior[i] if self.prior[i] > 0 else 1.0 / self.n_classes for i in range(self.n_classes) ] )


    def train(self, sequence,sequence2, transcript,gt,video,video2,label2index, split, learning_rate = 0.1, window = 20, step = 5):
        #print('--------------------new video-----------------')
        data_wrapper = DataWrapper(sequence, window_size = 21)
        # forwarding and Viterbi decoding
        output_origin,log_probs_origin = self._forward(data_wrapper)
        if sequence2 is not None:
            data_wrapper_gru2 = DataWrapper(sequence2, window_size = 21)
            _,log_probs_origin2=self._forward(data_wrapper_gru2)

            data_wrapper_gru_dual = DataWrapper_dual(sequence,sequence2, window_size=21)
            probs_view_confidence= self._forward_dual(data_wrapper_gru_dual,mode=self.WPI_net)
            w1 = probs_view_confidence[:, 0:1]
            w2 = probs_view_confidence[:, 1:2]

        log_probs = log_probs_origin.data.cpu().numpy() - np.log(self.prior)
        log_probs = log_probs - np.max(log_probs)


        optimizer = optim.SGD(list(self.net.parameters())+list(self.net_vc.parameters()), lr = learning_rate / 512)
        optimizer.zero_grad()
        ################################
        if sequence2 is not None:
            log_probs2 = log_probs_origin2.data.cpu().numpy() - np.log(self.prior)
            log_probs2 = log_probs2- np.max(log_probs2)

        self.decoder.grammar = MultiTranscriptGrammar('{}/results/grammar.txt'.format(split), label2index, transcript, 0)
        self.decoder.length_model = PoissonModel(self.mean_lengths,max_length = 2000)

        if sequence2 is not None:
            _, labels1, _,  all_segments_online = self.decoder.decode(log_probs)
            if self.MVI == "WPI":
                aggr_log_probs = w1.data.cpu().numpy() * log_probs + w2.data.cpu().numpy() * log_probs2
                score, labels, segments, _ = self.decoder.decode(aggr_log_probs)
            elif self.MVI == "PI":
                aggr_log_probs = 0.5 * log_probs + 0.5 * log_probs2
                score, labels, segments, _ = self.decoder.decode(aggr_log_probs)
            elif self.MVI == "SV":
                score, labels, segments, _ = self.decoder.decode(log_probs + log_probs2,dual=True)
            else:
                assert 1==0,"wrong Multi-View Inference (MVI) chosen"

            penalty = -(w1 * log_probs_origin.detach() + w2 * log_probs_origin2.detach())
            loss_vc = self.decoder.forward_score(penalty, segments , transcript, window, step)
        else:
            score, labels, segments, all_segments_online = self.decoder.decode(log_probs)
            loss_vc=0
        ################################
        penalty = -log_probs_origin
        loss1 = self.decoder.forward_score(penalty, segments, transcript, window, step)
        loss2 = self.decoder.incremental_forward_score(penalty, segments, transcript, window, step)
        loss3 = self.decoder.OODL(all_segments_online, segments, -penalty)
        alpha = 1
        loss = loss1 - loss2 + alpha * loss3
        if sequence2 is not None and self.MVI=="WPI":
            loss=loss+loss_vc
        loss.backward()
        optimizer.step()

        # add sequence to buffer
        self.buffer.add_sequence(output_origin, transcript, labels)
        self.update_prior()
        self.update_mean_lengths()

        return loss1, loss2, loss3


    def save_model(self, network_file,length_file, prior_file):
        self.net.cpu()
        torch.save(self.net.state_dict(), network_file)
        self.net.cuda()
        np.savetxt(length_file, self.mean_lengths)
        np.savetxt(prior_file, self.prior)




