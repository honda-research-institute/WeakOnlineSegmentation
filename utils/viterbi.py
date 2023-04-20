#!/usr/bin/python3.7
#References:
#https://github.com/JunLi-Galios/CDFL
#https://github.com/alexanderrichard/NeuralNetwork-Viterbi
import numpy as np
from .grammar import PathGrammar
from .length_model import PoissonModel
import glob
import re
import torch
torch.manual_seed(0)
np.random.seed(0)
# Viterbi decoding
class Viterbi(object):

    ### helper structure ###
    class TracebackNode(object):
        def __init__(self, label, predecessor, length = 0, boundary = False):
            self.label = label
            self.length = length
            self.predecessor = predecessor
            self.boundary = boundary

    ### helper structure ###
    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, traceback):
                self.score = score
                self.traceback = traceback

        def update(self, key, score, traceback, logadd=False):
            if (not key in self):
                self[key] = self.Hypothesis(score, traceback)
            else:
                if logadd:
                    self[key].score = self.logadd(score, self[key].score)
                elif self[key].score <= score:
                    self[key] = self.Hypothesis(score, traceback)

        def logadd(self, a, b):
#             print('a', a)
#             print('b', b)
            if a <= b:
                result = - torch.log(1 + torch.exp(a - b)) + a
            else:
                result = - torch.log(1 + torch.exp(b - a)) + b
            return result

    # @grammar: the grammar to use, must inherit from class Grammar
    # @length_model: the length model to use, must inherit from class LengthModel
    # @frame_sampling: generate hypotheses every frame_sampling frames
    # @max_hypotheses: maximal number of hypotheses. Smaller values result in stronger pruning
    def __init__(self, grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf):
        self.grammar = grammar
        self.length_model = length_model
        self.frame_sampling = frame_sampling
        self.max_hypotheses = max_hypotheses




    def get_sequence_probability_difference(self,frame_scores, segments, segments_online, t):
        total_n=0
        online_prob=torch.tensor(0).cuda().type(torch.cuda.FloatTensor);offline_prob=torch.tensor(0).cuda().type(torch.cuda.FloatTensor)

        i_offline=0;i_online=0;
        offline_start=0;offline_end=min(segments[0].length,t)
        online_start=0;online_end=segments_online[0].length
        assert np.sum(np.asarray([segments_online[i].length for i in range(len(segments_online))]))==t

        while(1):
            if online_end == offline_end and online_end==t: #the final segment
                end = min(offline_end, online_end)
                start = max(online_start, offline_start)
                if segments_online[i_online].label != segments[i_offline].label and start != end:
                    online_prob = online_prob +self.frame_score2(frame_scores,end-1,start-1,segments_online[i_online].label)
                    offline_prob = offline_prob + self.frame_score2(frame_scores,end-1,start-1,segments[i_offline].label)
                    total_n = total_n + end - start
                break

            end=min(offline_end, online_end)
            start=max(online_start,offline_start)
            if segments_online[i_online].label==segments[i_offline].label or start==end: # if segmetn labels are the same
                if online_end < offline_end:
                    i_online =i_online + 1
                    assert i_online < len(segments_online)
                    online_start = online_end
                    online_end = online_start + segments_online[i_online].length
                else:
                    i_offline = i_offline + 1
                    assert i_offline < len(segments)
                    offline_start = offline_end
                    offline_end = min(offline_start + segments[i_offline].length,t)
                continue

            assert start<end
            online_prob = online_prob + self.frame_score2(frame_scores, end - 1, start - 1,segments_online[i_online].label)
            offline_prob = offline_prob + self.frame_score2(frame_scores, end - 1, start - 1, segments[i_offline].label)
            total_n=total_n+end-start

            if online_end<offline_end:
                i_online=i_online+1
                assert i_online<len(segments_online)
                online_start=online_end
                online_end=online_start+segments_online[i_online].length
            else:
                i_offline=i_offline+1
                assert i_offline<len(segments)
                offline_start = offline_end
                offline_end = min(offline_start + segments[i_offline].length,t)

        return offline_prob,online_prob,total_n
    #


    def OODL(self,all_segments_online, segments, log_frame_probs): #online-offline discrepency loss
        frame_scores = torch.cumsum(log_frame_probs, axis=0)
        loss=0
        i=0
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            if t+self.frame_sampling<frame_scores.shape[0]: # partial loss for t=T would be 0 as online==offline
                offline_probability,online_probability,total_n=self.get_sequence_probability_difference(frame_scores,segments,all_segments_online[i],t+1)
                loss=loss+torch.maximum(torch.tensor(0).cuda(),online_probability-offline_probability)/(total_n+1e-16) #for energy
                #loss = loss + online_probability / (total_n + 1e-16)
                i=i+1

        return loss


    def cum_OODL(self,all_segments_online, segments, log_frame_probs): #cumulative online-offline discrepency loss
        frame_scores = torch.cumsum(log_frame_probs, axis=0)
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        loss=torch.tensor(0).cuda().type(torch.cuda.FloatTensor)
        i = 0
        # decode each following time step
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            l=0;k=0
            while l<t:
                l=l+segments[k].length
                k=k+1
            offline_label=segments[k-1].label
            online_label=all_segments_online[i][-1].label
            hyps, offline_probability,online_probability = self.get_cum_probability_difference(t, hyps, frame_scores,online_label,offline_label)
            if i!=len(all_segments_online)-1 and offline_label!=online_label: #The last segment will be counted below in the finalize_decoding_torch
                loss=loss+(offline_probability-online_probability)/(t+1e-16)
            i = i + 1
        assert i==len(all_segments_online)
        offline_label = segments[- 1].label
        online_label = all_segments_online[-1][-1].label
        if offline_label != online_label:
            offline_probability,online_probability = self.finalize_decoding_torch(hyps,online_label,offline_label)
            loss = loss +(offline_probability-online_probability)/(frame_scores.shape[0])
        return loss

    def get_cum_probability_difference(self,t, old_hyp, frame_scores, online_label, offline_label):
        def logadd(a, b):
            if a == np.inf and b == np.inf:
                return np.inf
            if a <= b:
                result = - torch.log(1 + torch.exp(a - b)) + a
            else:
                result = - torch.log(1 + torch.exp(b - a)) + b
            return result

        new_hyp = self.HypDict()
        L_c = dict()
        assert len(old_hyp.items())!=0
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            new_key = context + (label, min(length + self.frame_sampling, self.length_model.max_length()))
            score_temp = hyp.score + self.frame_score(frame_scores, t, label) #+ self.grammar.time_score(t, context[1:], label)
            if label == online_label or label == offline_label:
                if label not in L_c:
                    L_c[label] = np.inf
                L_c[label] = logadd(L_c[label], -1 * score_temp)
            score = hyp.score + self.frame_score(frame_scores, t, label)
            new_hyp.update(new_key, score, self.TracebackNode(label, hyp.traceback, boundary=False))
            # ... or go to the next label
            context = context + (label,)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol() :
                    continue
                new_key = context + (new_label, self.frame_sampling)
                score_temp = hyp.score + self.frame_score(frame_scores, t, new_label) + self.length_model.score(length,label) + self.grammar.score(context, new_label) #+ self.grammar.time_score(t, context[1:], new_label)
                if new_label==online_label or new_label==offline_label:
                    if new_label not in L_c:
                        L_c[new_label] = np.inf
                    L_c[new_label] = logadd(L_c[new_label], -1 * score_temp)
                score = hyp.score + self.frame_score(frame_scores, t, new_label) + self.length_model.score(length,label) + self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary=True))
        # return new hypotheses
        return new_hyp,L_c[offline_label],L_c[online_label]


    def decode(self, log_frame_probs,dual=False):
        all_segments_online= []
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores allow for quick lookup if frame_sampling > 1
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        # decode each following time step
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            hyps,online_hyp = self.decode_frame(t, hyps, frame_scores,dual)
            self.prune(hyps)
            _,segments_online = self.traceback(online_hyp, t+1)
            all_segments_online.append(segments_online)
        # transition to end symbol
        final_hyp = self.finalize_decoding(hyps,dual)
        labels, segments = self.traceback(final_hyp, frame_scores.shape[0])

        online_hyp = self.finalize_decoding_online2(hyps,dual)
        _,all_segments_online[-1]=self.traceback(online_hyp, frame_scores.shape[0])

        return final_hyp.score, labels, segments,all_segments_online


    def decode_online(self, log_frame_probs):
        segments=[]
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores allow for quick lookup if frame_sampling > 1
        # create initial hypotheses
        hyps,segment = self.init_decoding_online(frame_scores)
        segments.append(segment)
        # decode each following time step
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            hyps,segment = self.decode_frame_online(t, hyps, frame_scores)
            self.prune(hyps)
            segments.append(segment)
        # transition to end symbol
        segments[-1] = self.finalize_decoding_online(hyps,frame_scores.shape[0])
        labels,segments=self.patch_segments(segments,frame_scores.shape[0])
        return 0, labels, segments




    ### helper functions ###
    def frame_score(self, frame_scores, t, label):
        if t >= self.frame_sampling:
            return frame_scores[t, label] - frame_scores[t - self.frame_sampling, label]
        else:
            return frame_scores[t, label]

    def frame_score2(self, frame_scores, end,start, label):
        if start >= 0:
            return frame_scores[end, label] - frame_scores[start, label]
        else:
            return frame_scores[end, label]

    def prune(self, hyps):
        if len(hyps) > self.max_hypotheses:
            tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )
            del_keys = [ x[1] for x in tmp[0 : -self.max_hypotheses] ]
            for key in del_keys:
                del hyps[key]

    def init_decoding(self, frame_scores):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling)
            score = self.grammar.score(context, label) + self.frame_score(frame_scores, self.frame_sampling - 1, label)
            hyps.update(key, score, self.TracebackNode(label, None, boundary = True))
        return hyps



    def init_decoding_online(self, frame_scores):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling)
            score = self.grammar.score(context, label) + self.frame_score(frame_scores, self.frame_sampling - 1, label)
            hyps.update(key, score, self.TracebackNode(label, None, boundary = True))
            if hyps[key].score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = hyps[key].score, hyps[key].traceback

        segment = Segment(final_hyp.traceback.label)
        segment.length=self.frame_sampling
        return hyps,segment


    def decode_frame(self, t, old_hyp, frame_scores,dual=False):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        new_hyp = self.HypDict()
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            new_key = context + (label, min(length + self.frame_sampling, self.length_model.max_length()))
            score_temp = hyp.score + self.frame_score(frame_scores, t, label)#+self.length_model.score_half_poisson(min(length + self.frame_sampling, self.length_model.max_length()),label)
            score = hyp.score + self.frame_score(frame_scores, t, label)
            new_hyp.update(new_key, score, self.TracebackNode(label, hyp.traceback, boundary = False))
            if score_temp >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score_temp, new_hyp[new_key].traceback
            # ... or go to the next label
            context = context + (label,)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol():
                    continue
                new_key = context + (new_label, self.frame_sampling)
                if dual:
                    score_temp = hyp.score + self.frame_score(frame_scores, t, new_label) + 2*self.length_model.score(length, label) + self.grammar.score(context, new_label)
                    score = hyp.score + self.frame_score(frame_scores, t, new_label) + 2*self.length_model.score(length,label) + self.grammar.score(context, new_label)
                else:
                    score_temp = hyp.score + self.frame_score(frame_scores, t, new_label) + self.length_model.score(length,label) + self.grammar.score(context, new_label)
                    score = hyp.score + self.frame_score(frame_scores, t, new_label) + self.length_model.score(length, label) + self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary = True))
                if score_temp >= final_hyp.score:
                    assert new_hyp[new_key].score==score_temp
                    final_hyp.score, final_hyp.traceback = new_hyp[new_key].score, new_hyp[new_key].traceback

        # return new hypotheses
        return new_hyp,final_hyp



    def decode_frame_online(self, t, old_hyp, frame_scores):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        new_hyp = self.HypDict()
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            new_key = context + (label, min(length + self.frame_sampling, self.length_model.max_length()))
            curr_len=min(length + self.frame_sampling, self.length_model.max_length())
            score_temp = hyp.score + self.frame_score(frame_scores, t, label)+self.length_model.score_half_poisson(curr_len,label)
            score = hyp.score + self.frame_score(frame_scores, t, label)
            new_hyp.update(new_key, score, self.TracebackNode(label, hyp.traceback, boundary = False))
            if score_temp > final_hyp.score:
                final_hyp.score, final_hyp.traceback = score_temp, new_hyp[new_key].traceback
            # ... or go to the next label
            context = context + (label,)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol():
                    continue
                new_key = context + (new_label, self.frame_sampling)
                score_temp = hyp.score + self.frame_score(frame_scores, t, new_label) + self.length_model.score(length,label) + self.grammar.score(context, new_label)
                score = hyp.score + self.frame_score(frame_scores, t, new_label) + self.length_model.score(length, label) + self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary = True))
                if score_temp > final_hyp.score:
                    assert new_hyp[new_key].score==score_temp,str(new_key)+"   "+str(t)
                    final_hyp.score, final_hyp.traceback = new_hyp[new_key].score, new_hyp[new_key].traceback
        # return new hypotheses
        segment = Segment(final_hyp.traceback.label)
        segment.length=self.frame_sampling
        return new_hyp,segment



    def finalize_decoding(self, old_hyp,dual=False):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            context = context + (label,)
            if dual:
                score = hyp.score + 2*self.length_model.score(length, label) + self.grammar.score(context,self.grammar.end_symbol()) + self.grammar.is_positive(context[1:])
            else:
                score = hyp.score + self.length_model.score(length, label)+ self.grammar.score(context, self.grammar.end_symbol())+self.grammar.is_positive(context[1:])
            if score > final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        return final_hyp




    def finalize_decoding_online(self, old_hyp,t):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            context = context + (label,)
            score = hyp.score + self.length_model.score(length, label)#+self.grammar.time_score(t, context[1:], label)
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        segment = Segment(final_hyp.traceback.label)
        segment.length = self.frame_sampling
        return segment



    def finalize_decoding_online2(self, old_hyp,dual=False):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            context = context + (label,)
            if dual:
                score = hyp.score + 2*self.length_model.score(length, label)
            else:
                score = hyp.score + self.length_model.score(length, label)
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        return final_hyp








    def traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        labels = []
        segments = [Segment(traceback.label)]
        while not traceback == None:
            segments[-1].length += self.frame_sampling
            labels += [traceback.label] * self.frame_sampling
            if traceback.boundary and not traceback.predecessor == None:
                segments.append( Segment(traceback.predecessor.label) )
            traceback = traceback.predecessor
        segments[0].length += n_frames - len(labels) # append length of missing frames
        labels = [hyp.traceback.label] * (n_frames - len(labels)) + labels # append labels for missing frames
        return list(reversed(labels)), list(reversed(segments))


    def patch_segments(self,segments, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0

        patched_segments= [Segment(segments[0].label)]
        labels=[]
        for i in range(len(segments)):
            if i !=0:
                if segments[i].label!=segments[i-1].label:
                    patched_segments.append(Segment(segments[i].label))

            patched_segments[-1].length += self.frame_sampling
            labels += [segments[i].label] * self.frame_sampling

        assert n_frames-len(labels)>=0
        patched_segments[-1].length+= n_frames-len(labels)
        labels += [patched_segments[-1].label] * (n_frames - len(labels))
        return labels,patched_segments

    def patch_segments2(self,segments, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0

        patched_segments= [Segment(segments[0][-1].label)]
        patched_segments[-1].length=self.frame_sampling
        labels= [segments[0][-1].label] * self.frame_sampling
        for i in range(len(segments)):
            if i !=0:
                if segments[i][-1].label!=segments[i-1][-1].label:
                    patched_segments.append(Segment(segments[i][-1].label))

            patched_segments[-1].length += self.frame_sampling
            labels += [segments[i][-1].label] * self.frame_sampling

        assert n_frames-len(labels)>=0
        patched_segments[-1].length+= n_frames-len(labels)
        labels += [patched_segments[-1].label] * (n_frames - len(labels))
        return labels,patched_segments

    def lookup_gt(self, gt):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        segments=[Segment(gt[0])]
        labels = []
        ii=0
        for i in range(len(gt)):
            labels.append(gt[i])
            if i!=0:
                if gt[i]!=gt[i-1]:
                    segments[-1].length=i-ii
                    segments.append(Segment(gt[i]))
                    ii=i
        segments[-1].length=i-ii
        return labels, segments

    def init_forward(self):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        key = context + (-1, )
        hyps.update(key, 0, None)
        return hyps
    
    def init_discriminative_forward(self):
        hyps = self.HypDict()
        key = -1
        hyps.update(key, 0, None)
        return hyps

    def forward_frame(self, t, old_hyp, frame_scores, next_label):
        new_hyp = self.HypDict()
        new_key = None
        for key, hyp in list(old_hyp.items()):
            context, idx = key[0:-1], key[-1]
            if t <= idx:
                continue

            # ... go to the next label
            if idx < 0:
                segment_score = frame_scores[t, next_label]
            else:
                segment_score = frame_scores[t, next_label] - frame_scores[idx, next_label]
            score = hyp.score + segment_score           
            context = context + (next_label,)
            new_key = context + (t,)
            new_hyp.update(new_key, score, None, logadd=True)
        return new_hyp, new_key
    
    def discriminative_forward_frame(self, t, old_hyp, frame_scores):
        new_hyp = self.HypDict()
        new_key = t
        for key, hyp in list(old_hyp.items()):
            idx = key
            if t <= idx:
                continue

            # ... go to the next label
            for label in range(self.grammar.n_classes()):                
                if idx < 0:
                    segment_score = frame_scores[t, label]
                else:
                    segment_score = frame_scores[t, label] - frame_scores[idx, label]
                score = hyp.score + segment_score
                new_hyp.update(new_key, score, None, logadd=True)  
        return new_hyp, new_key

    def forward_score(self, log_frame_probs, segments, transcript, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = torch.cumsum(log_frame_probs, dim=0)  # cumulative frame scores
        hyps = self.init_forward()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2-1 , cum_segments[i] + window // 2, step):
                t = min(t, length - 1)
                hyp, _ = self.forward_frame(t, hyps, frame_scores, transcript[i])
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))        
            hyps = new_hyps
        # transition to end symbol
        final_hyp, final_key = self.forward_frame(length - 1, hyps, frame_scores, transcript[-1])
        return final_hyp[final_key].score

    def discriminative_forward_score(self, log_frame_probs, segments, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = torch.cumsum(log_frame_probs, dim=0)  # cumulative frame scores
        hyps = self.init_discriminative_forward()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2-1 , cum_segments[i] + window // 2, step):
                t = min(t, length - 1)
                hyp, _ = self.discriminative_forward_frame(t, hyps, frame_scores)
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))        
            hyps = new_hyps
        # transition to end symbol
        final_hyp, final_key = self.discriminative_forward_frame(length - 1, hyps, frame_scores)

        return final_hyp[final_key].score


    def init_incremental_forward(self):
        hyps = self.HypDict()
        key = -1
        hyps.update(key, torch.Tensor([0.]).cuda(), None)
        return hyps
    
    
    def incremental_forward_frame(self, t, old_hyp, frame_scores, next_label):
        new_hyp = self.HypDict()
        new_key = t
        for key, hyp in list(old_hyp.items()):
            idx = key
            if t <= idx:
                continue

            # ... go to the next label
            if idx < 0:
                segment_score = frame_scores[t, :]
            else:
                segment_score = frame_scores[t, :] - frame_scores[idx, :]
            for label in range(self.grammar.n_classes()):                
                if segment_score[label] < segment_score[next_label]:
                    score = hyp.score + segment_score[label]
                else:
                    score = hyp.score
                new_hyp.update(new_key, score, None, logadd=True)  
        return new_hyp

    
    def incremental_forward_score(self, log_frame_probs, segments, transcript, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = torch.cumsum(log_frame_probs, dim=0)  # cumulative frame scores
        hyps = self.init_incremental_forward()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2 -1, cum_segments[i] + window // 2, step):
                if t >= length:
                    t = (cum_segments[i] + cum_segments[i + 1]) / 2
                hyp = self.incremental_forward_frame(t, hyps, frame_scores, transcript[i])
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))
            hyps = new_hyps
        # transition to end symbol
        final_hyp = self.incremental_forward_frame(length - 1, hyps, frame_scores, transcript[-1])
        final_key = length - 1

        return final_hyp[final_key].score


    def stn_decode(self, log_frame_probs, segments, trancript, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores
        hyps = self.init_stn_decoding()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2 - 1, cum_segments[i] + window // 2, step):
                t = min(t, length - 1)
                hyp = self.stn_decode_frame(t, hyps, frame_scores, trancript[i])
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))        
            hyps = new_hyps
        final_hyp, final_score = self.stn_finalize_decoding(length - 1, hyps, frame_scores, trancript[-1])
        
        stn_labels, stn_segments = self.stn_traceback(final_hyp, frame_scores.shape[0])

        return final_score, stn_labels, stn_segments
    
    
    def init_stn_decoding(self):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        idx = -1
        key = context + (idx, )
        hyps.update(key, 0, self.TracebackNode(self.grammar.start_symbol(), None, 0, boundary = True))
        return hyps

    def stn_decode_frame(self, t, old_hyp, frame_scores, next_label):
        new_hyp = self.HypDict()
        for key, hyp in list(old_hyp.items()):
            context, idx = key[0:-1], key[-1]
            if t <= idx:
                continue

            # ... go to the next label
            if idx < 0:
                segment_score = frame_scores[t, next_label]
            else:
                segment_score = frame_scores[t, next_label] - frame_scores[idx, next_label]
            length = t - idx
            score = hyp.score + segment_score + self.length_model.score(length, next_label) + self.grammar.score(context, next_label)
            context = context + (next_label,)
            new_key = context + (t,)
            new_hyp.update(new_key, score, self.TracebackNode(next_label, hyp.traceback, length, boundary = True), logadd=False)
        return new_hyp
    

    def stn_finalize_decoding(self, t, old_hyp, frame_scores, next_label):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in list(old_hyp.items()):
            context, idx = key[0:-1], key[-1]
            if idx < 0:
                segment_score = frame_scores[t, next_label]
            else:
                segment_score = frame_scores[t, next_label] - frame_scores[idx, next_label]
            context = context + (next_label,)
            length = t - idx
            score = hyp.score + segment_score + self.length_model.score(length, next_label) + self.grammar.score(context, self.grammar.end_symbol())
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, self.TracebackNode(next_label, hyp.traceback, length, boundary = True)
        return final_hyp, final_hyp.score

    def stn_traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        labels = []
        segments = []
        while not traceback.predecessor == None:
            segments.append(Segment(traceback.label))
            segments[-1].length = traceback.length
            labels += [traceback.label] * traceback.length
            traceback = traceback.predecessor
        return list(reversed(labels)), list(reversed(segments))




