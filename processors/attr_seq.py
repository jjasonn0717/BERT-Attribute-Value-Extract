""" attribute value extrtaction fine-tuning: utilities to work  """
import torch
import logging
import os
import copy
import json
from .utils_attr import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, title, attribute,labels):
        self.guid = guid
        self.title = title
        self.attribute = attribute
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, 
                 t_input_ids, t_input_mask, t_segment_ids, t_input_len, t_orig_to_tok_index, t_word_len,
                 a_input_ids, a_input_mask, a_segment_ids, a_input_len, a_orig_to_tok_index, a_word_len,
                 label_ids):

        self.t_input_ids = t_input_ids
        self.t_input_mask = t_input_mask
        self.t_segment_ids = t_segment_ids
        self.t_input_len = t_input_len
        self.t_orig_to_tok_index = t_orig_to_tok_index
        self.t_word_len = t_word_len
        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.a_input_len = a_input_len
        self.a_orig_to_tok_index = a_orig_to_tok_index
        self.a_word_len = a_word_len
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens, \
    a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens, \
    all_label_ids = map(torch.stack, zip(*batch))
    t_max_len = max(t_all_lens).item()
    t_max_word_len = max(t_all_word_lens).item()
    a_max_len = max(a_all_lens).item()
    a_max_word_len = max(a_all_word_lens).item()

    t_all_input_ids = t_all_input_ids[:, :t_max_len]
    t_all_input_mask = t_all_input_mask[:, :t_max_len]
    t_all_segment_ids = t_all_segment_ids[:, :t_max_len]
    t_all_orig_to_tok_index = t_all_orig_to_tok_index[:, :t_max_word_len]
    all_label_ids = all_label_ids[:, :t_max_word_len]

    a_all_input_ids = a_all_input_ids[:, :a_max_len]
    a_all_input_mask = a_all_input_mask[:, :a_max_len]
    a_all_segment_ids = a_all_segment_ids[:, :a_max_len]
    a_all_orig_to_tok_index = a_all_orig_to_tok_index[:, :a_max_word_len]

    return t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens, \
    a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens, \
    all_label_ids


def tokenize_per_word(words, tokenizer, len_limit, prefix_tokens=[]):
    orig_to_tok_index = list(range(len(prefix_tokens)))
    tokens = [token for token in prefix_tokens]
    for i, word in enumerate(words):
        """
        Note: by default, we use the first wordpiece token to represent the word
        If you want to do something else (e.g., use last wordpiece to represent), modify them here.
        """
        ## tokenize the word into word_piece / BPE
        ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
        ## Related GitHub issues:
        ##      https://github.com/huggingface/transformers/issues/1196
        ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
        ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
        #word_tokens = tokenizer.tokenize(" " + word)
        word_tokens = tokenizer.tokenize(word) #TODO: for the transformers in this repo, need to add `add_prefix_space=True`
        if len(tokens) + len(word_tokens) <= len_limit: 
            orig_to_tok_index.append(len(tokens))
            for sub_token in word_tokens:
                tokens.append(sub_token)
        else:
            break
    return tokens, orig_to_tok_index

def convert_examples_to_features(examples, tokenizer, label_list, max_seq_length, max_attr_length):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2

        prefix_tokens = [tokenizer.cls_token]

        t_tokens, t_orig_to_tok_index = tokenize_per_word(example.title, tokenizer,
                                                          max_seq_length-special_tokens_count+len(prefix_tokens),
                                                          prefix_tokens)
        a_tokens, a_orig_to_tok_index = tokenize_per_word(example.attribute, tokenizer,
                                                          max_attr_length-special_tokens_count+len(prefix_tokens),
                                                          prefix_tokens)
        label_ids = [label_map[tokenizer.cls_token]] + [label_map[x] for x in example.labels]
        label_ids = label_ids[:len(t_orig_to_tok_index)]
        assert len(t_tokens) <= max_seq_length - special_tokens_count + len(prefix_tokens)
        assert len(a_tokens) <= max_attr_length - special_tokens_count + len(prefix_tokens)
        assert len(label_ids) <= max_seq_length - special_tokens_count + len(prefix_tokens)


        t_orig_to_tok_index += [len(t_tokens)]
        a_orig_to_tok_index += [len(a_tokens)]
        t_tokens += [tokenizer.sep_token]
        a_tokens += [tokenizer.sep_token]
        label_ids += [label_map[tokenizer.sep_token]]
        
        t_segment_ids = [0] * len(t_tokens)
        a_segment_ids = [0] * len(a_tokens)

        t_input_ids = tokenizer.convert_tokens_to_ids(t_tokens)
        a_input_ids = tokenizer.convert_tokens_to_ids(a_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        t_input_mask = [1] * len(t_input_ids)
        a_input_mask = [1] * len(a_input_ids)
        t_input_len = len(t_input_ids)
        a_input_len = len(a_input_ids)
        t_word_len = len(t_orig_to_tok_index)
        a_word_len = len(a_orig_to_tok_index)
        # Zero-pad up to the sequence length.
        t_padding_length = max_seq_length - len(t_input_ids)
        a_padding_length = max_attr_length - len(a_input_ids)
        t_word_padding_length = max_seq_length - len(t_orig_to_tok_index)
        a_word_padding_length = max_attr_length - len(a_orig_to_tok_index)

        t_orig_to_tok_index += [0] * t_word_padding_length
        t_input_ids += [tokenizer.pad_token_id] * t_padding_length
        t_input_mask += [0] * t_padding_length
        t_segment_ids += [0] * t_padding_length
        label_ids += [tokenizer.pad_token_id] * t_word_padding_length

        a_orig_to_tok_index += [0] * a_word_padding_length
        a_input_ids += [tokenizer.pad_token_id] * a_padding_length
        a_input_mask += [0] * a_padding_length
        a_segment_ids += [0] * a_padding_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens_title: %s", " ".join([str(x) for x in t_tokens]))
            logger.info("input_ids_title: %s", " ".join([str(x) for x in t_input_ids]))
            logger.info("input_mask_title: %s", " ".join([str(x) for x in t_input_mask]))
            logger.info("input_len_title: %s", str(t_input_len))
            logger.info("segment_ids_title: %s", " ".join([str(x) for x in t_segment_ids]))
            logger.info("orig_to_tok_index_title: %s", " ".join([str(x) for x in t_orig_to_tok_index]))
            logger.info("word_len_title: %s", str(t_word_len))
            logger.info("tokens_attr: %s", " ".join([str(x) for x in a_tokens]))
            logger.info("input_ids_attr: %s", " ".join([str(x) for x in a_input_ids]))
            logger.info("input_mask_attr: %s", " ".join([str(x) for x in a_input_mask]))
            logger.info("input_len_attr: %s", str(a_input_len))
            logger.info("segment_ids_attr: %s", " ".join([str(x) for x in a_segment_ids]))
            logger.info("orig_to_tok_index_attr: %s", " ".join([str(x) for x in a_orig_to_tok_index]))
            logger.info("word_len_attr: %s", str(a_word_len))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            '''
            print("*** Example ***")
            print("guid:", example.guid)
            print("tokens_title:", " ".join([str(x) for x in t_tokens]))
            print("input_ids_title:", " ".join([str(x) for x in t_input_ids]))
            print("input_mask_title:", " ".join([str(x) for x in t_input_mask]))
            print("input_len_title:", str(t_input_len))
            print("segment_ids_title:", " ".join([str(x) for x in t_segment_ids]))
            print("orig_to_tok_index_title:", " ".join([str(x) for x in t_orig_to_tok_index]))
            print("word_len_title:", str(t_word_len))
            print("tokens_attr:", " ".join([str(x) for x in a_tokens]))
            print("input_ids_attr:", " ".join([str(x) for x in a_input_ids]))
            print("input_mask_attr:", " ".join([str(x) for x in a_input_mask]))
            print("input_len_attr:", str(a_input_len))
            print("segment_ids_attr:", " ".join([str(x) for x in a_segment_ids]))
            print("orig_to_tok_index_attr:", " ".join([str(x) for x in a_orig_to_tok_index]))
            print("word_len_attr:", str(a_word_len))
            print("label_ids:", " ".join([str(x) for x in label_ids]))
            '''
        assert len(t_input_ids) == max_seq_length
        assert len(t_input_mask) == max_seq_length
        assert len(t_segment_ids) == max_seq_length
        assert len(t_orig_to_tok_index) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(a_input_ids) == max_attr_length
        assert len(a_input_mask) == max_attr_length
        assert len(a_segment_ids) == max_attr_length
        assert len(a_orig_to_tok_index) == max_attr_length

        features.append(InputFeatures(t_input_ids=t_input_ids,
                                      t_input_mask=t_input_mask,
                                      t_input_len = t_input_len,
                                      t_segment_ids=t_segment_ids,
                                      t_orig_to_tok_index=t_orig_to_tok_index,
                                      t_word_len=t_word_len,
                                      a_input_ids=a_input_ids,
                                      a_input_mask=a_input_mask,
                                      a_input_len=a_input_len,
                                      a_segment_ids=a_segment_ids,
                                      a_orig_to_tok_index=a_orig_to_tok_index,
                                      a_word_len=a_word_len,
                                      label_ids=label_ids))
    return features

class AttrProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_json(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_json(data_path), "dev")

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_json(data_path), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-a","I-a",'S-a','O',"[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            title= line['title']
            attribute = line['attr']
            labels = line['labels']
            examples.append(InputExample(guid=guid, title=title, attribute = attribute,labels=labels))
        return examples
ner_processors = {
    'attr':AttrProcessor
}

