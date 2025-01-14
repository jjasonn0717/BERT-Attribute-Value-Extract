import argparse
import glob
import logging
import os
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME,BertConfig,BertTokenizer
from models.bert_for_attr import BertCrfForAttr
from processors.utils_attr import CNerTokenizer,get_entities
from processors.attr_seq import convert_examples_to_features
from processors.attr_seq import ner_processors as processors
from processors.attr_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext,roberta
    'bert': (BertConfig, BertCrfForAttr, BertTokenizer),
}

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    t_lstm_param_optimizer = list(model.t_lstm.named_parameters())
    a_lstm_param_optimizer = list(model.a_lstm.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in t_lstm_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.001},
        {'params': [p for n, p in t_lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.001},
        {'params': [p for n, p in a_lstm_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.001},
        {'params': [p for n, p in a_lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.001},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.001},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.001},
    ]

    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens, all_label_ids
            inputs = {'t_input_ids': batch[0],
                      'a_input_ids': batch[6],
                      't_orig_to_tok_index': batch[4],
                      'a_orig_to_tok_index': batch[10],
                      't_attention_mask': batch[1],
                      'a_attention_mask': batch[7],
                      'labels': batch[12],
                      't_input_lens': batch[3],
                      'a_input_lens':batch[9],
                      't_word_lens': batch[5],
                      'a_word_lens':batch[11]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["t_token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                inputs["a_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            '''
            print("t_input", tokenizer.decode(inputs['t_input_ids'][0].detach().cpu().numpy()))
            print("a_input", tokenizer.decode(inputs['a_input_ids'][0].detach().cpu().numpy()))
            print("t_orig_to_tok_index", inputs['t_orig_to_tok_index'][0])
            print("a_orig_to_tok_index", inputs['a_orig_to_tok_index'][0])
            print("t_attention_mask", inputs['t_attention_mask'][0])
            print("a_attention_mask", inputs['a_attention_mask'][0])
            print("labels", inputs['labels'][0])
            print("t_input_lens", inputs['t_input_lens'][0])
            print("a_input_lens", inputs['a_input_lens'][0])
            print("t_word_lens", inputs['t_word_lens'][0])
            print("a_word_lens", inputs['a_word_lens'][0])
            '''
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            #input()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)
                        Testevaluate(args, model, tokenizer)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens, all_label_ids
        with torch.no_grad():
            inputs = {'t_input_ids': batch[0],
                      'a_input_ids': batch[6],
                      't_orig_to_tok_index': batch[4],
                      'a_orig_to_tok_index': batch[10],
                      't_attention_mask': batch[1],
                      'a_attention_mask': batch[7],
                      'labels': batch[12],
                      't_input_lens': batch[3],
                      'a_input_lens':batch[9],
                      't_word_lens': batch[5],
                      'a_word_lens':batch[11]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["t_token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                inputs["a_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags,_ = model.crf._obtain_labels(logits, args.id2label, inputs['t_word_lens'])
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif out_label_ids[i][j] == args.label2id['[SEP]']:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(tags[i][j])
        pbar(step)
    print(' ')
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    return results

def Testevaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='test')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running Test evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens, all_label_ids
        with torch.no_grad():
            inputs = {'t_input_ids': batch[0],
                      'a_input_ids': batch[6],
                      't_orig_to_tok_index': batch[4],
                      'a_orig_to_tok_index': batch[10],
                      't_attention_mask': batch[1],
                      'a_attention_mask': batch[7],
                      'labels': batch[12],
                      't_input_lens': batch[3],
                      'a_input_lens':batch[9],
                      't_word_lens': batch[5],
                      'a_word_lens':batch[11]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["t_token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                inputs["a_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags,_ = model.crf._obtain_labels(logits, args.id2label, inputs['t_word_lens'])
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif out_label_ids[i][j] == args.label2id['[SEP]']:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(tags[i][j])
        pbar(step)
    print(' ')
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    return results

def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens, all_label_ids
        with torch.no_grad():
            inputs = {'t_input_ids': batch[0],
                      'a_input_ids': batch[6],
                      't_orig_to_tok_index': batch[4],
                      'a_orig_to_tok_index': batch[10],
                      't_attention_mask': batch[1],
                      'a_attention_mask': batch[7],
                      'labels': None,
                      't_input_lens': batch[3],
                      'a_input_lens':batch[9],
                      't_word_lens': batch[5],
                      'a_word_lens':batch[11]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["t_token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                inputs["a_token_type_ids"] = (batch[8] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            preds, _ = model.crf._obtain_labels(logits, args.id2label, inputs['t_word_lens'])
        preds = preds[0][1:-1] # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(preds)
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    print(" ")
    output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file,test_submit)

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if data_type == 'train':
            logger.info("Creating features from dataset file at %s", args.train_data_path)
            examples = processor.get_train_examples(args.train_data_path)
        elif data_type == 'dev':
            logger.info("Creating features from dataset file at %s", args.dev_data_path)
            examples = processor.get_dev_examples(args.dev_data_path)
        else:
            logger.info("Creating features from dataset file at %s", args.test_data_path)
            examples = processor.get_test_examples(args.test_data_path)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                max_attr_length = args.max_attr_length
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    t_all_input_ids = torch.tensor([f.t_input_ids for f in features], dtype=torch.long)
    t_all_input_mask = torch.tensor([f.t_input_mask for f in features], dtype=torch.long)
    t_all_segment_ids = torch.tensor([f.t_segment_ids for f in features], dtype=torch.long)
    t_all_lens = torch.tensor([f.t_input_len for f in features], dtype=torch.long)
    t_all_orig_to_tok_index = torch.tensor([f.t_orig_to_tok_index for f in features], dtype=torch.long)

    t_all_word_lens = torch.tensor([f.t_word_len for f in features], dtype=torch.long)

    a_all_input_ids = torch.tensor([f.a_input_ids for f in features], dtype=torch.long)
    a_all_input_mask = torch.tensor([f.a_input_mask for f in features], dtype=torch.long)
    a_all_segment_ids = torch.tensor([f.a_segment_ids for f in features], dtype=torch.long)
    a_all_lens = torch.tensor([f.a_input_len for f in features], dtype=torch.long)
    a_all_orig_to_tok_index = torch.tensor([f.a_orig_to_tok_index for f in features], dtype=torch.long)
    a_all_word_lens = torch.tensor([f.a_word_len for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, t_all_orig_to_tok_index, t_all_word_lens,
                            a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, a_all_orig_to_tok_index, a_all_word_lens,
                            all_label_ids)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir",default=None,type=str,required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",)
    parser.add_argument("--train_data_path",default=None,type=str,required=True,
                        help="The input train data.",)
    parser.add_argument("--dev_data_path",default=None,type=str,required=True,
                        help="The input dev data.",)
    parser.add_argument("--test_data_path",default=None,type=str,required=True,
                        help="The input test data.",)
    parser.add_argument("--model_type",default=None,type=str,required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path",default=None,type=str,required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--output_dir",default=None,type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Other parameters
    parser.add_argument('--markup',default='bios',type=str,choices=['bios','bio'])
    parser.add_argument( "--labels",default="",type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",)
    parser.add_argument( "--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",default="",type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir",default="",type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--eval_max_seq_length",default=512,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--max_attr_length",default=16,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument( "--max_steps", default=-1,type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints",action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument('--predict_all_checkpoints',action="store_true",
                        help="Predict all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16",action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level",type=str,default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1#torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,device,args.n_gpu, bool(args.local_rank != -1),args.fp16,)
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,)
    model = model_class.from_pretrained(args.model_name_or_path,from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,cache_dir=args.cache_dir if args.cache_dir else None,
                                        label2id=args.label2id,device=args.device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint,config=config,label2id=args.label2id, device=args.device)
            model.to(args.device)
            result= evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_all_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint,config=config,label2id=args.label2id, device=args.device)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)

if __name__ == "__main__":
    main()

