CURRENT_DIR=`pwd`
export BERT_BASE_DIR=bert-base-uncased
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="attr"
TRAIN_DATA_PATH="/home/ec2-user/SageMaker/data/attribute_extraction/milk_rule_based_annotations.jsonl"
DEV_DATA_PATH="/home/ec2-user/SageMaker/data/attribute_extraction/milk_rule_based_annotations_5.jsonl"
TEST_DATA_PATH="/home/ec2-user/SageMaker/data/attribute_extraction/milk_rule_based_annotations_5.jsonl"

python run_attr_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --train_data_path=$TRAIN_DATA_PATH \
  --dev_data_path=$DEV_DATA_PATH \
  --test_data_path=$TEST_DATA_PATH \
  --train_max_seq_length=65 \
  --eval_max_seq_length=65 \
  --max_attr_length=16 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=636 \
  --save_steps=636 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

