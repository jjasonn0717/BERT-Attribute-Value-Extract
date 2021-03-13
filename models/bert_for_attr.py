import torch
import torch.nn as nn
from .crf import CRF
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers import Attention
from torch.nn import LayerNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    pack the sequence with PackSequence
    """

    def __init__(self,
                 input_size:int,
                 hidden_size: int,
                 num_layers: int =1,
                 batch_first: bool = True,
                 bidirectional: bool = True):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        lstm_out, (h, c) = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.

        return lstm_out[recover_idx], (h[:, recover_idx], c[:, recover_idx])


class BertCrfForAttr(BertPreTrainedModel):
    def __init__(self, config, label2id, device):
        super(BertCrfForAttr, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.t_lstm = BiLSTMEncoder(input_size=config.hidden_size,hidden_size=config.hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.a_lstm = BiLSTMEncoder(input_size=config.hidden_size,hidden_size=config.hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.attention = Attention()
        self.ln = LayerNorm(config.hidden_size* 2)
        self.classifier = nn.Linear(config.hidden_size* 2, len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device, is_bert=True)
        self.init_weights()

    def forward(self,
                t_input_ids,
                a_input_ids,
                t_orig_to_tok_index,
                a_orig_to_tok_index,
                t_token_type_ids=None,
                a_token_type_ids=None,
                t_attention_mask=None,
                a_attention_mask=None,
                labels=None,
                t_input_lens=None,
                a_input_lens=None,
                t_word_lens=None,
                a_word_lens=None):
        # bert
        outputs_title = self.bert(t_input_ids, t_token_type_ids, t_attention_mask)
        outputs_attr = self.bert(a_input_ids, a_token_type_ids, a_attention_mask)
        #print("t bert:", outputs_title[0].size())
        #print("a bert:", outputs_attr[0].size())
        # get the word embeddings
        batch_size, _, rep_size = outputs_title[0].size()
        _, t_max_word_len = t_orig_to_tok_index.size()
        title_embeddings = torch.gather(outputs_title[0], 1, t_orig_to_tok_index.unsqueeze(-1).expand(batch_size, t_max_word_len, rep_size))
        #print("t bert:", title_embeddings.size())
        _, a_max_word_len = a_orig_to_tok_index.size()
        attr_embeddings = torch.gather(outputs_attr[0], 1, a_orig_to_tok_index.unsqueeze(-1).expand(batch_size, a_max_word_len, rep_size))
        #print("a bert:", attr_embeddings.size())
        # bilstm
        title_output, _ = self.t_lstm(title_embeddings, t_word_lens)
        _, attr_hidden = self.a_lstm(attr_embeddings, a_word_lens)
        #print("t lstm:", title_output.size())
        #assert type(attr_hidden) == tuple and len(attr_hidden) == 2
        #print("a lstm:", attr_hidden[0].size(), attr_hidden[1].size())
        # attention
        attr_output = torch.cat([attr_hidden[0][-2], attr_hidden[0][-1]], dim=-1)
        attention_output = self.attention(title_output, attr_output)
        #print("attn out:", attention_output.size())
        # catconate
        outputs = torch.cat([title_output, attention_output], dim=-1)
        #print("out:", outputs.size())
        outputs = self.ln(outputs)
        sequence_output = self.dropout(outputs)
        logits = self.classifier(sequence_output)
        #print("logits:", logits.size())
        outputs = (logits,)
        if labels is not None:
            loss = self.crf.calculate_loss(logits, tag_list=labels, lengths=t_word_lens)
            outputs =(loss,)+outputs
        return outputs # (loss), scores
