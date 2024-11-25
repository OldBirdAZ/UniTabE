# UniTabE
code for paper: [UniTabE: A Universal Pretraining Protocol for Tabular Foundation Model in Data Science](https://arxiv.org/abs/2307.09249)

## Environment 
```bash
conda env create -f experiment-envs/a100_env.txt -n YOUR-ENV-NAME
```

## Training
- train your own model with [trainer.py](unitab%2Ftrainer.py) for single gpu training;
- train your own model with [trainer_mgpu.py](unitab%2Ftrainer_mgpu.py) for multi-gpu training;
- infer [common_const.py](unitab%2Fcommon_const.py) for more training tasks
  - e.g. "dynamic_mask_span_task" means dynamically mask-then-predict task
  - "classification_predict_task" is used for classification task

## Finetuning
finetune in classification task:
```bash
python train.py \
--task_names "classification_predict_task" \
--restore_path your_checkpoint_path \
--save_dir your_save_dir \
--train_data_dir ../../datas/transtab_official_datas/n_folds_normed/credit-g/fold_0/train \
--valid_data_path ../../datas/transtab_official_datas/n_folds_normed/credit-g/fold_0/test.jsonl \
--emb_size=768 --hidden_size=768 --n_head=12 --ffn_size=1536 --n_enc_layer=12 \
--datatype_aware --num_workers=1 --lr=1e-6 --batch_size=8 --n_g_accum=1 --num_epoch=200 --dropout=0.0 
```

## Infer
```bash
python inference.py \
--task_names "classification_predict_task" \
--restore_path ../outputs/finetune/kaggle-all-patchs/bs-m4/cls/cg-fold_0/EP-state-25.pt \
--test_data_path ../../datas/transtab_official_datas/n_folds_normed/credit-g/fold_0/test.jsonl \
--emb_size=768 --hidden_size=768 --n_head=12 --ffn_size=1536 --n_enc_layer=12 \
--num_workers=1 --lr=1e-6 --batch_size=8 --n_g_accum=1 --num_epoch=200 --dropout=0.0 \
--test_out_path ../outputs/finetune/kaggle-all-patchs/bs-m4/cls/cg-fold_0/test_out_ep25.jsonl 
```

