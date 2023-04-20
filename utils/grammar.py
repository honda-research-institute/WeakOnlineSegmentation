#!/usr/bin/python3.7
#References:
# https://github.com/JunLi-Galios/CDFL
#https://github.com/alexanderrichard/NeuralNetwork-Viterbi
import numpy as np
import random
np.random.seed(0)
random.seed(0)

class Grammar(object):
    
    # @context: tuple containing the previous label indices
    # @label: the current label index
    # @return: the log probability of label given context p(label|context)
    def score(self, context, label): # score is a log probability
        return 0.0

    # @return: the number of classes
    def n_classes(self):
        return 0

    # @return sequence start symbol
    def start_symbol(self):
        return -1

    # @return sequence end symbol
    def end_symbol(self):
        return -2

    # @context: tuple containing the previous label indices
    # @return: list of all possible successor labels for the given context
    def possible_successors(context):
        return set()

# grammar containing all action transcripts seen in training
# used for inference
class PathGrammar(Grammar):
    
    def __init__(self, transcript_file, label2index_map):
        self.num_classes = len(label2index_map)
        transcripts = self._read_transcripts(transcript_file, label2index_map)
        self.transcripts=transcripts
        # generate successor sets
        self.successors = dict()
        for transcript in transcripts:
            transcript = transcript + [self.end_symbol()]
            for i in range(len(transcript)):
                context = (self.start_symbol(),) + tuple(transcript[0:i])
                self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )
        self._get_trans_mat()

    def _read_transcripts(self, transcript_file, label2index_map):
        transcripts = []
        with open(transcript_file, 'r') as f:
            lines = f.read().split('\n')[0:-1]
        for line in lines:
            transcripts.append( [ label2index_map[label] for label in line.split() ] )
        return transcripts

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf

    def time_score(self,t,context,label):
        if len(context)==0:
            context =tuple([label])
        elif context[-1]!=label:
            context=context+tuple([label])
        if t not in self.time_grammar:
            t=np.max(self.time_grammar.keys())
        if context in self.time_grammar[t]:
            return 0.0
        else:
            return -np.inf


    def trans_score(self, context, label):
        if self.T_mat[context,label]==1:
            return 0.0
        else:
            return -np.inf

    def _get_trans_mat(self):
        self.T_mat=np.zeros((self.num_classes,self.num_classes),dtype=int)
        for i in range(self.num_classes):self.T_mat[i,i]=1
        for transcript in self.transcripts:
            for i in range(len(transcript)-1):
                self.T_mat[transcript[i],transcript[i+1]]=1

    def create_time_variant_grammars(self,mean_lengths,transcript2duration):
        def get_rel_boundaries(transcript,mean_lengths,D):
            rel_boundaries=np.zeros((len(transcript)))
            mean_length_sum=0
            for c in range(len(transcript)):
                rel_boundaries[c]=mean_lengths[transcript[c]]

            rel_boundaries=np.cumsum(rel_boundaries)
            rel_boundaries=(rel_boundaries/rel_boundaries[-1])*D
            rel_boundaries=rel_boundaries.astype(int)-1
            assert rel_boundaries[0]>=0 and rel_boundaries[-1]==int(D)-1
            return rel_boundaries


        self.time_grammar=dict()
        for transcript in self.transcripts:

            for d in transcript2duration[tuple(transcript)]:
                D=d
                rel_boundaries=get_rel_boundaries(transcript,mean_lengths,D)
                assert len(rel_boundaries)==len(transcript)
                idx=0
                for t in range(int(D)):
                    if t not in self.time_grammar:
                        self.time_grammar[t]=set()
                    if(t>rel_boundaries[idx]):
                        idx=idx+1

                    if idx>0:
                        self.time_grammar[t].add(tuple(transcript[:idx]+transcript[idx:idx+1]))
                    else:
                        self.time_grammar[t].add(tuple([transcript[idx]]))









# grammar that generates only a single transcript
# use during training to align frames to transcript
class SingleTranscriptGrammar(Grammar):

    def __init__(self, transcript, n_classes):
        self.num_classes = n_classes
        self.transcripts = transcript
        transcript = transcript + [self.end_symbol()]
        self.successors = dict()
        for i in range(len(transcript)):
            context = (self.start_symbol(),) + tuple(transcript[0:i])
            self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf


    def create_time_variant_grammars(self,mean_lengths,transcript2duration):
        def get_rel_boundaries(transcript,mean_lengths,D):
            rel_boundaries=np.zeros((len(transcript)))
            mean_length_sum=0
            for c in range(len(transcript)):
                rel_boundaries[c]=mean_lengths[transcript[c]]

            rel_boundaries=np.cumsum(rel_boundaries)
            rel_boundaries=(rel_boundaries/rel_boundaries[-1])*D
            rel_boundaries=rel_boundaries.astype(int)-1
            assert rel_boundaries[0]>=0 and rel_boundaries[-1]==int(D)-1
            return rel_boundaries


        self.time_grammar=dict()
        transcript =self.transcripts
        for D in transcript2duration[tuple(transcript)]:
            rel_boundaries=get_rel_boundaries(transcript,mean_lengths,D)
            assert len(rel_boundaries)==len(transcript)
            idx=0
            for t in range(int(D)):
                if t not in self.time_grammar:
                    self.time_grammar[t]=set()
                if(t>rel_boundaries[idx]):
                    idx=idx+1

                if idx>0:
                    self.time_grammar[t].add(tuple(transcript[:idx]+transcript[idx:idx+1]))
                else:
                    self.time_grammar[t].add(tuple([transcript[idx]]))


class MultiTranscriptGrammar(Grammar):

    def __init__(self, transcript_file, label2index_map,pos_transcript,n_neg):
        self.pos_transcript=pos_transcript
        self.num_classes = len(label2index_map)
        transcripts = self._read_transcripts(transcript_file, label2index_map)
        transcripts.sort()
        self.transcripts = transcripts
        random.shuffle(self.transcripts)
        # generate successor sets
        self.successors = dict()
        for idx,transcript in enumerate(self.transcripts):
            if idx>=n_neg:
                break
            transcript = transcript + [self.end_symbol()]
            for i in range(len(transcript)):
                context = (self.start_symbol(),) + tuple(transcript[0:i])
                self.successors[context] = set([transcript[i]]).union(self.successors.get(context, set()))

        transcript=pos_transcript
        transcript = transcript + [self.end_symbol()]
        for i in range(len(transcript)):
            context = (self.start_symbol(),) + tuple(transcript[0:i])
            self.successors[context] = set([transcript[i]]).union(self.successors.get(context, set()))


    def _read_transcripts(self, transcript_file, label2index_map):
        transcripts = []
        with open(transcript_file, 'r') as f:
            lines = f.read().split('\n')[0:-1]
        for line in lines:
            transcripts.append([label2index_map[label] for label in line.split()])
        return transcripts

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf

    def is_positive(self,context):
        if list(context)==self.pos_transcript:
            return 0.0
        else:
            return -np.inf

    def create_time_variant_grammars(self, mean_lengths, transcript2duration):
        def get_rel_boundaries(transcript, mean_lengths, D):
            rel_boundaries = np.zeros((len(transcript)))
            mean_length_sum = 0
            for c in range(len(transcript)):
                rel_boundaries[c] = mean_lengths[transcript[c]]

            rel_boundaries = np.cumsum(rel_boundaries)
            rel_boundaries = (rel_boundaries / rel_boundaries[-1]) * D
            rel_boundaries = rel_boundaries.astype(int) - 1
            assert rel_boundaries[0] >= 0 and rel_boundaries[-1] == int(D) - 1
            return rel_boundaries

        self.time_grammar = dict()
        transcript = self.pos_transcript
        for D in transcript2duration[tuple(transcript)]:
            rel_boundaries = get_rel_boundaries(transcript, mean_lengths, D)
            assert len(rel_boundaries) == len(transcript)
            idx = 0
            for t in range(int(D)):
                if t not in self.time_grammar:
                    self.time_grammar[t] = set()
                if (t > rel_boundaries[idx]):
                    idx = idx + 1

                if idx > 0:
                    self.time_grammar[t].add(tuple(transcript[:idx] + transcript[idx:idx + 1]))
                else:
                    self.time_grammar[t].add(tuple([transcript[idx]]))