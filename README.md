# Commonsense-guided Search Query Generation

This repo provides code and data accompanying our EMNLP 2023 Findings paper, [Social Commonsense-Guided Search Query Generation for
Open-Domain Knowledge-Powered Conversation]().

If you find this useful, please cite:
```
```

## Install

Make a new Python 3.7+ environment using `virtualenv` or `conda`.

```
pip install -r requirements.txt
```

## Training the Query Generator

```
python train_t5_query.py 
    --model_name google/flan-t5-large 
    --dataset_path dataset/train_query.json 
    --out_dir <output checkpoint folder>
    --device_batch_size 4 
    --gradient_accumulate_steps 16 
    --epochs 2 
    --lr 1e-5 
    --logging_steps 50 
    --save_steps 500 
    --eval_steps 500  
    --cosmo 1
```

After the above training, you can run the `eval_query.py` script to see the below numbers on the validation set (which assumes ChatGPT-generated queries as ground truth). The paper uses Flan-T5-Large (770M) as underlying model. Here, we also show numbers for Flan-T5-XL (3B).

```
python eval_query.py \
    --model_path <model_path> \
    --eval_set dataset/val_query.py \
    --cosmo 1 \
    --out_path <out_path> \
```

where 
- <model_path>: Path of pretrained/finetuned query generator 
- <query_set>: evaluation dataset
- <out_path>: Json file to store evaluation results

| Model | Version | Rouge Score |
|  ----  | ----  | ---- | 
| Flan-T5-Large | Zero-shot | 0.1749 |
| Flan-T5-Large | Finetuned | 0.4477 |
| Flan-T5-XL | Zero-shot | 0.2032 |
| Flan-T5-XL | Finetuned | 0.4997 | 

If you directly wish to use the trained models, they are available here: [Flan-T5-Large](https://huggingface.co/google/flan-t5-large) and [Flan-T5-XL]
(https://huggingface.co/google/flan-t5-xl). 

## Query Generation

You can use the script to generate a query, given a dialog at inference time:

The finetuned topic tracking model is [here]().



## Evaluation

We follow an approach similar to G-Eval for evaluating the quality of the search queries. Given N search queries (per example), you can use the G-Eval script as follows:
