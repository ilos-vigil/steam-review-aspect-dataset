# Model benchmark

Model benchmark on Steam Review aspect dataset split into 3 categories,

* Base: Non-attention based language model.
* Embedding: Inspired by MTEB, obtained embedding trained on Logistic Regressor for up to 100 epochs.
* Fine-tune.

Source code for running these models is also provided. But take note it may not follow best practice as it was written with the goal of using it only once. I ran those on Linux, RTX 3060 and 32GB RAM.

## Base

| Model              | Macro precision | Macro recall | Macro F1 | Note                                                                         |
| ------------------ | --------------- | ------------ | -------- | ---------------------------------------------------------------------------- |
| Spacy Bag of Words | 0.6203          | 0.5391       | 0.5494   |                                                                              |
| FastText           | 0.6284          | 0.5713       | 0.5871   | Minimum text preprocessing, use pretrained vector                            |
| FastText           | 0.6933          | 0.5821       | 0.6027   | Minimum text preprocessing, choose hyperparameter based on K-5 fold autotune |
| Spacy Ensemble     | 0.6043          | 0.6773       | 0.6299   | Choose hyperparameter based on simple grid search                            |

## Embedding

| Model                                                     | Param | Max tokens | Macro precision | Macro recall | Macro F1 | Note                                 |
| --------------------------------------------------------- | ----- | ---------- | --------------- | ------------ | -------- | ------------------------------------ |
| sentence-transformers/all-mpnet-base-v2                   | 110M  | 514        | 0.7074          | 0.5431       | 0.5853   |                                      |
| jinaai/jina-embeddings-v2-small-en                        | 137M  | 8192       | 0.7068          | 0.6075       | 0.6437   |                                      |
| jinaai/jina-embeddings-v2-base-en                         | 137M  | 8192       | 0.6813          | 0.6501       | 0.6618   |                                      |
| Alibaba-NLP/gte-large-en-v1.5                             | 434M  | 8192       | 0.7001          | 0.6501       | 0.6729   |                                      |
| nomic-ai/nomic-embed-text-v1.5                            | 137M  | 8192       | 0.7075          | 0.6498       | 0.6756   |                                      |
| McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised | 7111M | 32768      | 0.7238          | 0.6697       | 0.6928   | NF4 double quantization, instruction |
| WhereIsAI/UAE-Large-V1                                    | 335M  | 512        | 0.7245          | 0.6718       | 0.6946   |                                      |
| mixedbread-ai/mxbai-embed-large-v1                        | 335M  | 512        | 0.7215          | 0.6817       | 0.6989   |                                      |
| intfloat/e5-mistral-7b-instruct                           | 7111M | 32768      | 0.7345          | 0.7000       | 0.7137   | NF4 double quantization, instruction |

## Fine-tune

| Model                             | Param | Max tokens | Macro precision | Macro recall | Macro F1 | Note                                            |
| --------------------------------- | ----- | ---------- | --------------- | ------------ | -------- | ----------------------------------------------- |
| jinaai/jina-embeddings-v2-base-en | 137M  | 8192       | 0.7485          | 0.7257       | 0.7354   | Choose hyperparameter from Ray Tune (30 trials) |
| Alibaba-NLP/gte-large-en-v1.5     | 434M  | 8192       | 0.8403          | 0.8152       | 0.8231   | Choose hyperparameter from Ray Tune (16 trials) |
