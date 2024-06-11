import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
)
import gc
import numpy as np
import random
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


SEED = 42
MAX_LENGTH = 32768
INSTRUCTION = 'Classify the aspect mentioned in the given Steam Review into up to of the eight aspects: recommended, story, gameplay, visual, audio, technical, price, and suggestion.'  # This mimic paper's string instruction

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def encode(examples, tokenizer):
    outputs = tokenizer(
        [INSTRUCTION + s for s in examples['cleaned_review']],
        truncation=True, max_length=MAX_LENGTH
    )
    return outputs


def model_init():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        # 8, 16, ... where lora_alpha usually is 2x of r
        r=8,
        # few suggest lora_alpha have biggest impact
        lora_alpha=16,
        # usually 0.0, 0.05 or 0.1 
        lora_dropout=0.05,
        bias='none',
        use_rslora=True,
        task_type='CAUSAL_LM',
        # Possible configuration
        # 1. Doesn't specify and let LoraConfig decide
        # 2. Only key and value
        # 3, Only query, key, value and o_proj
        # 4. All dense layer
        # target_modules = [
        #     'q_proj',
        #     'v_proj',
        # ],
        # target_modules = [
        #     'q_proj',
        #     'k_proj',
        #     'v_proj',
        #     'o_proj'
        # ],
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
            'embed_tokens',
            'lm_head',
        ],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        'intfloat/e5-mistral-7b-instruct', trust_remote_code=True,
        num_labels=8, problem_type='multi_label_classification',
        quantization_config=quantization_config,
        # token='HF_XXX'
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def sigmoid(x):
    return 1/(1 + np.exp(-x))


metric = evaluate.combine(['precision', 'recall', 'f1'])
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return metric.compute(predictions=predictions, references=labels.astype(int).reshape(-1), average='macro')


tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
ds_all = load_dataset('ilos-vigil/steam-review-aspect-dataset')

X_train_dummy = np.zeros(shape=ds_all['train'].num_rows)
y_train = np.array(ds_all['train']['labels'])

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_index, val_index = list(msss.split(X_train_dummy, y_train))[0]

ds_train = ds_all['train'].select(train_index)
ds_val = ds_all['train'].select(val_index)

ds_train = ds_train.map(encode, batched=True, fn_kwargs={'tokenizer': tokenizer})
ds_val = ds_val.map(encode, batched=True, fn_kwargs={'tokenizer': tokenizer})


training_args = TrainingArguments(
    #
    output_dir='tune',
    logging_steps=5,
    report_to='tensorboard',
    #
    dataloader_drop_last=False,
    eval_strategy='epoch',
    #
    bf16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
)
tune_config = {
    'learning_rate': tune.loguniform(3e-5, 3e-4),
    'weight_decay': tune.loguniform(0.00001, 0.01),
    'warmup_ratio': tune.loguniform(0.03, 0.1)
}
print(training_args)
print(tune_config)
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
gc.collect()
torch.cuda.empty_cache()


best_trial = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    direction='maximize',
    backend='ray',
    n_trials=6,
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
