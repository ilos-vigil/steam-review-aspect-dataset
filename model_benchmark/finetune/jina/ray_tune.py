import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments
)
import numpy as np
import random
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def encode(examples, tokenizer):
    outputs = tokenizer(examples['cleaned_review'], truncation=True)
    return outputs


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'jinaai/jina-embeddings-v2-base-en', trust_remote_code=True,
        num_labels=8, problem_type='multi_label_classification'
    )


def sigmoid(x):
    return 1/(1 + np.exp(-x))


metric = evaluate.combine(['precision', 'recall', 'f1'])
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return metric.compute(predictions=predictions, references=labels.astype(int).reshape(-1), average='macro')


def run():
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
    ds_all = load_dataset('ilos-vigil/steam-review-aspect-dataset')

    # split and create validation dataset using stratified split
    X_train_dummy = np.zeros(shape=ds_all['train'].num_rows)
    y_train = np.array(ds_all['train']['labels'])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_index, val_index = list(msss.split(X_train_dummy, y_train))[0]

    ds_train = ds_all['train'].select(train_index)
    ds_val = ds_all['train'].select(val_index)

    ds_train = ds_train.map(encode, batched=True, fn_kwargs={'tokenizer': tokenizer})
    ds_val = ds_val.map(encode, batched=True, fn_kwargs={'tokenizer': tokenizer})

    training_args = TrainingArguments(
        output_dir='tune',
        eval_strategy='epoch',
        bf16=True,
        dataloader_drop_last=False,
        report_to='tensorboard',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        disable_tqdm=True,
    )
    # ref: https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html
    tune_config = {
        'gradient_accumulation_steps': tune.choice([16, 32]),
        'num_train_epochs': tune.randint(1, 6),
        'learning_rate': tune.loguniform(5e-6, 5e-5),
        'weight_decay': tune.loguniform(0.00001, 0.01),
        'warmup_ratio': tune.loguniform(0.01, 0.1)
    }

    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        model_init=model_init,
        compute_metrics=compute_metrics,
    )
    # Trainer automatically load model to GPU, but Ray Tune will spawn it's process.
    # So to free GPU memory, model (on this notebook) should be transfered to CPU.
    # Note that small GPU memory still used by the notebook due to CUDA context.
    trainer.model.cpu()
    torch.cuda.empty_cache()

    best_trial = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        direction='maximize',
        backend='ray',
        n_trials=30,
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray.tune.TuneConfig
        search_alg=HyperOptSearch(metric='objective', mode='max', random_state_seed=SEED),
        scheduler=ASHAScheduler(metric='objective', mode='max'),
        progress_reporter=CLIReporter(
            print_intermediate_tables=True,
            max_report_frequency=60*5,
            parameter_columns={
                'gradient_accumulation_steps': 'bs',
                'num_train_epochs': 'epoch',
                'learning_rate': 'lr',
                'weight_decay': 'w_decay',
                'warmup_ratio': 'warmup_ratio'
            },
            metric_columns=['eval_loss', 'eval_precision', 'eval_recall', 'eval_f1']
        ),
    )
    print(best_trial)
    print('='*50)
    return best_trial


run()
