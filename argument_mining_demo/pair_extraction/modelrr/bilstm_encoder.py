
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ContextEmb
from modelrr.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from allennlp.nn.util import batched_index_select

from overrides import overrides

class BiLSTMEncoder(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(BiLSTMEncoder, self).__init__()

        self.num_layers = 1 # ty editing

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2labels

        self.input_size = config.embedding_dim
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.char_feature = CharBiLSTM(config, print_info=print_info)
            self.input_size += config.charlstm_hidden_dim
        
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        self.type_embedding = nn.Embedding(2,20).to(self.device)

        if print_info:
            print("[Model Info] Input size to LSTM: {}".format(self.input_size))
            print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))

        self.lstm = nn.LSTM(788, config.hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True).to(self.device)
        self.lstm_token = nn.LSTM(input_size=768, hidden_size=768 // 2, num_layers=self.num_layers, batch_first=True,
                            bidirectional=True).to(self.device)

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)

        final_hidden_dim = config.hidden_dim

        if print_info:
            print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))

        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

        self.pair2score_first = nn.Linear(final_hidden_dim, 100).to(self.device)
        self.pair2score_second = nn.Linear(100, 50).to(self.device)
        #self.pair2score4 = nn.Linear(50, 10).to(self.device)
        self.pair2score_final = nn.Linear(50, 2).to(self.device)

        self.pair2score_first2 = nn.Linear(final_hidden_dim, 100).to(self.device)
        self.pair2score_second2 = nn.Linear(100, 2).to(self.device)

        self.pair2score_first1 = nn.Linear(final_hidden_dim, 2).to(self.device)




    @overrides
    def forward(self, sent_emb_tensor: torch.Tensor,
                      type_id_tensor: torch.Tensor,
                      sent_seq_lens: torch.Tensor,
                      num_tokens: torch.Tensor,
                initial_sent_emb_tensor: torch.Tensor,
                      batch_context_emb: torch.Tensor,
                      char_inputs: torch.Tensor,
                      char_seq_lens: torch.Tensor,
                pairs_eval: torch.Tensor,
                pair_padding_eval: torch.Tensor,
                      tags: torch.Tensor,
                      review_index: torch.Tensor,
                      reply_index: torch.Tensor,
                        pairs: torch.Tensor,
                        pair_padding_tensor: torch.Tensor,
                        max_review_id: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :param batch_context_emb: (batch_size, sent_len, context embedding) ELMo embedings
        :param char_inputs: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """

        # word_emb = self.word_embedding(word_seq_tensor)
        # if self.context_emb != ContextEmb.none:
        #     word_emb = torch.cat([word_emb, batch_context_emb.to(self.device)], 2)
        # if self.use_char:
        #     char_features = self.char_feature(char_inputs, char_seq_lens)
        #     word_emb = torch.cat([word_emb, char_features], 2)
        # print(type_id_tensor)

        initial_sent_emb_tensor = initial_sent_emb_tensor.to(self.device)
        # (batch_size * max_seq * max_tokens * 768)

        # sorted_num_tokens, tokenIdx = num_tokens.sort(0, descending=True) # num_tokens is 10 * paragraph_length
        # print(num_tokens.size())
        # print(num_tokens)
        # _, recover_token_idx = tokenIdx.sort(0, descending=False)
        # sorted_token_tensor = initial_sent_emb_tensor[tokenIdx]



        # for instance_idx in range(len(initial_sent_emb_tensor)):
        #     instance_sent_emb_tensor = initial_sent_emb_tensor[instance_idx][:sent_seq_lens[instance_idx]]
        #
        #     sorted_num_tokens, tokenIdx = num_tokens[instance_idx][:sent_seq_lens[instance_idx]].sort(0, descending=True)
        #     _, recover_token_idx = tokenIdx.sort(0, descending=False)
        #     sorted_sent_emb_tensor = instance_sent_emb_tensor[tokenIdx]
        #
        #     packed_tokens = pack_padded_sequence(sorted_sent_emb_tensor, sorted_num_tokens, True)
        #     _, (h_n, _) = self.lstm_token(packed_tokens, None) # hidden is of size
        #     h_n = self.drop_lstm(h_n)
        #     print(h_n.size())
        #     h_n = h_n.view(self.num_layers, 2, len(instance_sent_emb_tensor), 768//2)
        #     print(h_n.size())
        #     instance_result = torch.cat((h_n[-1, 0],h_n[-1, 1]), dim=1) # of size (length of sentence * 768)
        #     instance_result = instance_result[recover_token_idx]
        #     sent_emb_tensor[instance_idx, :sent_seq_lens[instance_idx], :] = instance_result

        initial_sent_emb_tensor_flatten = initial_sent_emb_tensor.view(-1, initial_sent_emb_tensor.size()[2], 768)
        num_tokens_flatten = num_tokens.view(-1)
        sorted_num_tokens, tokenIdx = num_tokens_flatten.sort(0, descending=True)
        _, recover_token_idx = tokenIdx.sort(0, descending=False)
        sorted_sent_emb_tensor_flatten = initial_sent_emb_tensor_flatten[tokenIdx]
        # print(sorted_num_tokens)
        sorted_num_tokens[sorted_num_tokens<=0]=1
        packed_tokens = pack_padded_sequence(sorted_sent_emb_tensor_flatten, sorted_num_tokens.cpu(), True)
        _, (h_n, _) = self.lstm_token(packed_tokens, None)
        h_n = self.drop_lstm(h_n)
        # print(h_n.size())
        h_n = h_n.view(self.num_layers, 2, len(initial_sent_emb_tensor_flatten), 768//2)
        # print(h_n.size())
        instance_result = torch.cat((h_n[-1, 0],h_n[-1, 1]), dim=1) # of size (length of sentence * 768)
        # print(instance_result.size())
        instance_result = instance_result[recover_token_idx].view(initial_sent_emb_tensor.size()[0], initial_sent_emb_tensor.size()[1], 768)
        sent_emb_tensor = instance_result




        # print(initial_sent_emb_tensor.size())


        # print('lstm_out_token, ', lstm_out_token.size())

        # sent_emb_tensor = lstm_out_token[recover_token_idx]





        type_emb = self.type_embedding(type_id_tensor)

        # sent_rep = sent_emb_tensor
        sent_rep = torch.cat([sent_emb_tensor,type_emb],2)


        sent_rep = self.word_drop(sent_rep)

        sorted_seq_len, permIdx = sent_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = sent_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)
        # print('feature_out, ', feature_out.size())

        outputs = self.hidden2tag(feature_out)
        feature_out = feature_out[recover_idx]

        lstm_review_rep = torch.gather(feature_out, 1, review_index.unsqueeze(2).expand(feature_out.size()))
        lstm_reply_rep = torch.gather(feature_out, 1, reply_index.unsqueeze(2).expand(feature_out.size()))
        batch_size, max_review, hidden_dim = lstm_review_rep.size()
        max_reply = lstm_reply_rep.size()[1]

        lstm_review_rep = lstm_review_rep.unsqueeze(2).expand(batch_size,max_review,max_reply,hidden_dim)
        lstm_reply_rep = lstm_reply_rep.unsqueeze(1).expand(batch_size,max_review,max_reply,hidden_dim)
        #lstm_pair_rep = torch.cat([lstm_review_rep, lstm_reply_rep], dim=-1)
        lstm_pair_rep = lstm_review_rep + lstm_reply_rep

        y = self.pair2score_first(lstm_pair_rep)
        y = F.relu(y)
        y = self.pair2score_second(y)
        #y = F.relu(y)
        #y = self.pair2score4(y)
        y = F.relu(y)
        score = self.pair2score_final(y)


        return feature_out,outputs[recover_idx],score
