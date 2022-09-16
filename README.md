# Active Learning for Neural Machine Translation

## Goal and Purpose
The machine translation mechanism translates texts automatically between different natural languages, and Neural Machine Translation (NMT) has gained attention for its rational context analysis and fluent translation accuracy. However, processing low-resource languages that lack relevant training attributes like supervised data is a current challenge for Natural Language Processing (NLP). We incorporated a technique known as active learning with the NMT toolkit Joey NMT to reach sufficient accuracy and robust predictions of low-resource language translation. With active learning, a semi-supervised machine learning strategy, the training algorithm determines which unlabeled data would be the most beneficial for obtaining labels using selected query techniques. We implemented two model-driven acquisition functions for selecting the samples to be validated. This work uses transformer-based NMT systems;  baseline model (NMT-1), fully trained model (NMT-2), active learning least confidence based model (NMT-3), and active learning margin sampling based model (NMT-4) when translating English to Hindi. The Bilingual Evaluation Understudy (BLEU) metric has been used to evaluate system results. The BLEU scores of NMT- 1, NMT- 2, NMT- 3 and NMT- 4 systems are 21, 22, 23 and 24, respectively. The findings demonstrate that active learning techniques improve the quality of the translation system.


## About the baseline Architecture
Joey NMT was initially developed and is maintained by [Jasmijn Bastings](https://github.com/bastings) (University of Amsterdam) and [Julia Kreutzer](https://juliakreutzer.github.io/) (Heidelberg University), now both at Google Research. [Mayumi Ohta](https://www.cl.uni-heidelberg.de/statnlpgroup/members/ohta/) at Heidelberg University is continuing the legacy.

## Features
Active Learning for Neural Machine Translation implements the following features (aka the minimalist toolkit of NMT :wrench:):
### Baseline Neural Machine Translation Model
- Transformer Encoder-Decoder
- BPE tokenization
- BLEU, PPL evaluation
- Beam search with length penalty and greedy decoding

### Active Learning Model
- Human-in-the-loop or non-Integrative for Query mechanism
- Random Strategy
- Margin Query Strategy
- Least Confidence Strategy
- Customizable initialization for Active Learning Dataset
- Learning curve plotting for BLEU and PPL
- Scoring hypotheses and references



## Installation
Active Learning for Neural Machine Translation is built on [JoeyNMT](https://github.com/joeynmt/joeynmt) and [PyTorch](https://pytorch.org/). Please make sure you have a compatible environment.
We tested Joey NMT 2.0 with
- python 3.9
- torch 1.12.1
- cuda 11.3

> :warning: **Warning**
> When running on **GPU** you need to manually install the suitable PyTorch version 
> for your [CUDA](https://developer.nvidia.com/cuda-zone) version.
> For example, you can install PyTorch 1.11.0 with CUDA v11.3 as follows:
> ```
> $ pip install --upgrade torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
> ```
> See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).


You can run the code from source.

### From source (for local development)
Clone this repository:
  ```bash
  $ git clone https://github.com/kritisingh24/joeynmt.git
  $ cd joeynmt
  ```

**[Optional]** For fp16 training, install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

1. Install conda from [Anaconda](https://www.anaconda.com/products/distribution#linux)
2. To build a reproducible environment follow below
```commandline
$ conda create --name test39 python=3.9
$ conda activate test39
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ pip install -r requirement.txt
```

## Usage

Active Learning for Neural Machine Translation has 4 modes: `train`, `test`, `translate` and `active_learning`, and all of them takes a
[YAML](https://yaml.org/)-style config file as argument.
You can find examples in the `configs` directory.

Most importantly, the configuration contains the description of the model architecture
(e.g. number of hidden units in the encoder RNN), paths to the training, development and
test data, and the training hyperparameters (learning rate, validation frequency etc.).

> :memo: **Info**
> Note that subword model training and joint vocabulary creation is not included
> in the 3 modes above, has to be done separately.
> We provide a script that takes care of it: `scritps/build_vocab.py`.
> ```
> $ python scripts/build_vocab.py configs/small.yaml --joint
> ```



### `active_learning` mode
For training active learning, run 
```bash
$ python main.py active_learning configs/baseline.yaml
```
This will train a baseline model on the training data, and keep aside a section of active learning data, validate on validation data, and store
model parameters, vocabularies, validation outputs. All needed information should be
specified in the `data`, `training` and `model` section of the config file (here
`configs/baseline.yaml`).

```
model_dir/
├── *.ckpt          # checkpoints
├── *.hyps          # translated texts at validation
├── config.yaml     # config file
├── spm.model       # sentencepiece model / subword-nmt codes file
├── src_vocab.txt   # src vocab
├── trg_vocab.txt   # trg vocab
├── train.log       # train log
└── validation.txt  # validation scores
```

> :bulb: **Tip**
> Be careful not to overwrite `model_dir`, set `overwrite: False` in the config file.

### `train` mode
For training, run 
```bash
$ python main.py train configs/fully_trained.yaml
```
This will train a model on the training data, validate on validation data, and store
model parameters, vocabularies, validation outputs. All needed information should be
specified in the `data`, `training` and `model` section of the config file (here
`configs/baseline.yaml`).


### `test` mode
This mode will generate translations for validation and test set (as specified in the
configuration) in `model_dir/out.[dev|test]`.
```
$ python -m joeynmt test configs/small.yaml --ckpt model_dir/avg.ckpt
```
If `--ckpt` is not specified above, the checkpoint path in `load_model` of the config
file or the best model in `model_dir` will be used to generate translations.

You can specify i.e. [sacrebleu](https://github.com/mjpost/sacrebleu) options in the
`test` section of the config file.

> :bulb: **Tip**
> `scripts/average_checkpoints.py` will generate averaged checkpoints for you.
> ```
> $ python scripts/average_checkpoints.py configs/small.yaml --joint
> ```

If you want to output the log-probabilities of the hypotheses or references, you can
specify `return_score: 'hyp'` or `return_score: 'ref'` in the testing section of the
config. And run `test` with `--output_path` and `--save_scores` options.
```
$ python -m joeynmt test configs/small.yaml --ckpt model_dir/avg.ckpt --output_path model_dir/pred --save_scores
```
This will generate `model_dir/pred.{dev|test}.{scores|tokens}` which contains scores and corresponding tokens.

> :memo: **Info**
> - If you set `return_score: 'hyp'` with greedy decoding, then token-wise scores will be returned. The beam search will return sequence-level scores, because the scores are summed up per sequence during beam exploration.
> - If you set `return_score: 'ref'`, the model looks up the probabilities of the given ground truth tokens, and both decoding and evaluation will be skipped.
> - If you specify `n_best` >1 in config, the first translation in the nbest list will be used in the evaluation.



### `translate` mode
This mode accepts inputs from stdin and generate translations.

- File translation
  ```
  $ python -m joeynmt translate configs/small.yaml < my_input.txt > output.txt
  ```

- Interactive translation
  ```
  $ python -m joeynmt translate configs/small.yaml
  ```
  You'll be prompted to type an input sentence. Joey NMT will then translate with the 
  model specified in `--ckpt` or the config file.

  > :bulb: **Tip**
  > Interactive `translate` mode doesn't work with Multi-GPU.
  > Please run it on single GPU or CPU.

> :memo: **Info**
> For interactive translate mode, you should specify `pretokenizer: "moses"` in the both src's and trg's `tokenizer_cfg`,
> so that you can input raw sentence. Then `MosesTokenizer` and `MosesDetokenizer` will be applied internally.
> For test mode, we used the preprocessed texts as input and set `pretokenizer: "none"` in the config.



## Model Results

### IIT-Bombay Parallel corpus

Pre-processing with Moses decoder tools as in [this script](scripts/get_iwslt14_bpe.sh).

The processed dataset is present at [GDrive](https://drive.google.com/drive/folders/10PopLXhotmWY1SfEkWwoZ-jdDAO6htZh?usp=sharing). This contains all the configuration and vocabulary on which the below model results have been acquired. 

Model | Architecture | tok | BLEU dev | BLEU test | #params | download
--------- | :----------: | :-- | --: | ---: | ------: | :-------
baseline | Transformer | subword-nmt | 16.55 | 18.26 | 19M | [enhi_transformer_t2_baseline.zip](https://drive.google.com/file/d/1uL7GE0Y4GDPq9GXfjESXmH1nFDVXWA5u/view?usp=sharing) (217MB)
fully_trained | Transformer | subword-nmt | 23.44 | 22.56 | 19M | [enhi_transformer_t2_fully_trained.zip](https://drive.google.com/file/d/1ly9klZyb0hr8d7Y-4lnE8upNnpRchX_6/view?usp=sharing) (219MB)
margin | Transformer | subword-nmt | 24.25 | 23.20 | 19M | [enhi_transformer_t2_margin.tar.gz](https://drive.google.com/file/d/1t-vOuhcUp8Li3rwVfDvi-ujeHf1sJl1L/view?usp=sharing) (216MB)
least_confidence | Transformer | subword-nmt | 24.11 | 23.36 | 19M | [enhi_transformer_t2_least_confidence.tar.gz](https://drive.google.com/file/d/1t-vOuhcUp8Li3rwVfDvi-ujeHf1sJl1L/view?usp=sharing) (215MB)


