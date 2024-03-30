# SPT-Code

## Requirements

3.2.1 Code Tokens. As we can see from Figure 2, the first part of the input is the code token sequence of a method. We use a lexical analyzer to tokenize the source code and then obtain the tokens𝐶 = {𝑐1,𝑐2,...,𝑐𝑙},where𝑙 isthenumberofcodetokens. Specifically, we use the Python standard library3 to tokenize Python codes.

3 https://docs.python.org/3.8/library/tokenize.html

``python 3.8``

```
conda remove -n spt-code --all
conda create -n spt-code python==3.8.17

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(MAC OS)
pip3 install torch torchvision torchaudio

python -m pip install transformers
conda install --yes --file requirements.txt
--
requirements.txt
	nltk
	tqdm
	psutil
	scikit-learn
	prettytable
	numpy
	dataclasses
	chardet
--
pip install rouge==1.0.0
pip install accelerate
pip install typing
pip install antlr4-tools
conda install -c conda-forge tensorboard

conda list -n spt-code
```

``
https://github.com/microsoft/CodeBERT/blob/c0de43d3aaf38e89290f1efb771f8de845e7a489/GraphCodeBERT/translation/parser/my-languages.so
``

### Minimize requirements

The list of minimize requirements can be found in `requirements.txt`.

### Additional requirements

If you need to reprocess the raw dataset, or use your own dataset,
then you will also need to install the following packages.
```
tree_sitter==0.19.0
antlr4-python3-runtime==4.9.2
```
```
pip install tree_sitter==0.19.0
pip install --upgrade tree-sitter
pip install antlr4-python3-runtime==4.9.2
```

Besides, `antlr4` need to be installed,
[installation guidance here](https://github.com/antlr/antlr4/blob/master/doc/getting-started.md).

If you encounter errors about `my-languages.so` when preprocessing the dataset, 
please run `sources/data/asts/build_lib.py` first.

```
git clone https://github.com/tree-sitter/tree-sitter-go
git clone https://github.com/tree-sitter/tree-sitter-javascript
git clone https://github.com/tree-sitter/tree-sitter-java
git clone https://github.com/tree-sitter/tree-sitter-python
git clone https://github.com/tree-sitter/tree-sitter-php
git clone https://github.com/tree-sitter/tree-sitter-ruby
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
python build_lib.py # See https://github.com/tree-sitter/py-tree-sitter/blob/master/README.md
```
Fix two errors below.
(1) Fix the 1st error in php below.
```
(spt-code) √ asts % python build_lib.py       
FileNotFoundError: [Errno 2] No such file or directory: 'vendor/tree-sitter-php/src/parser.c'

cd data/asts/vendor/tree-sitter-php
cp -r php/src .
```
(2) Fix the 2nd error in php below.
```
vendor/tree-sitter-php/src/scanner.c:1:10: fatal error: '../../common/scanner.h' file not found
#include "../../common/scanner.h"
         ^~~~~~~~~~~~~~~~~~~~~~~~

Edit the file scanner.c as follows.
(New)
< #include "../common/scanner.h"
---
(Old)
> #include "../../common/scanner.h"
```

Fixed an error on tree-sitter due to the incompatible version between 'tree-sitter' and 'tree-sitter-python'. (The error occurred at `parser.set_language(LANGUAGE[lang])` in a function `parse_ast` in `ast_parser.py`. A test file `test_ast_parser.py` has been used to fix this bug.)
```
pip install --upgrade tree-sitter
```

## Directory Structure with Code and Data

```
myoungkyu@oisit-selab2 √ ~/Documents/0-research-spt-code $ tree -L 2 .
.
├── dataset
│   ├── dataset_saved
│   ├── dataset_saved_org
│   ├── fine_tune_org
│   ├── pre_train
│   └── vocab_saved
├── pre_trained.zip
└── spt-code
    ├── git-proc-macos.sh
    ├── git-proc.sh
    ├── LICENSE
    ├── Makefile
    ├── outputs
    ├── pre_trained
    ├── README.md
    ├── requirements-org.txt
    ├── requirements.txt
    └── sources
```

```
myoungkyu@oisit-selab2 √ ~/Documents/0-research-spt-code $ tree -L 4 -D spt-code/pre_trained/
[Aug 27  2021]  spt-code/pre_trained/
├── [Aug 27  2021]  models
│   └── [Aug 27  2021]  all
│       ├── [Aug  1  2021]  config.json
│       ├── [Aug  1  2021]  pytorch_model.bin
│       └── [Aug  1  2021]  training_args.bin
└── [Aug 27  2021]  vocabs
    ├── [Aug 27  2021]  ast
    │   └── [Aug  1  2021]  ast.pk
    ├── [Aug 27  2021]  code
    │   └── [Aug  1  2021]  code.pk
    └── [Aug 27  2021]  nl
        └── [Aug  1  2021]  nl.pk
```

```
myoungkyu@oisit-selab2 √ ~/Documents/0-research-spt-code $ tree -D -t dataset/*saved
[Mar 17 00:24]  dataset/dataset_saved
└── [Mar 17 00:24]  pre_train.pk
```

## Datasets and Tokenizers

We provide pre-processed datasets, saved as pickle binary files, 
which can be loaded directly as instances.

The pre-processed datasets can be downloaded here: ([OneDrive](https://1drv.ms/u/s!Aj4XBdlu8BS0geoX0UgaslHdGvUCpg?e=sjBC6J), [iCloud](https://www.icloud.com.cn/iclouddrive/0158Oqc01mJDU9hOTsdsyoFDw#dataset), [GoogleDrive](https://drive.google.com/file/d/1Uf78WZYd_OqsV46j2Z7zWqtgmiDAFJb8/view?usp=sharing)).
Put the downloaded dataset pickle file into `{dataset_root}/dataset_saved/` (default to`.../dataset/dataset_saved`), 
the program will automatically detect and use it.

It is also possible to use a custom dataset, 
simply by placing it in the specified location according to the relevant settings in the source code, 
or by modifying the corresponding dataset loading script in the source code. 
The dataset loading code is located in the `sources/data/data.py` and `sources/data/data_utils.py` files.

##  Pre-trained Tokenizers and Models

Custom tokenizers (we call "vocab") can be downloaded here: ([OneDrive](https://1drv.ms/u/s!Aj4XBdlu8BS0geoV78e2KLC41sfasw?e=kfukTw), [iCloud](https://www.icloud.com.cn/iclouddrive/033gKQZigREGSYzRef-2yP6Bg#pre%5Ftrained), [Google Drive](https://drive.google.com/file/d/1PhVf5u8_uq5Tsl-OIvOGpqjA2y7D-9Dr/view?usp=sharing)). 
```
gdown https://drive.google.com/uc?id=ID where ID is 1PhVf5u8_uq5Tsl-OIvOGpqjA2y7D-9Dr
```
Extract it in a certain directory. 
Specific the argument `trained_vocab` of `main.py` 
where the tokenizers are located or put it in `{dataset_root}/vocab_saved` (default to`.../dataset/vocab_saved`).

You may pre-train SPT-Code by yourself. We also provide pre-trained models available [here](https://1drv.ms/u/s!Aj4XBdlu8BS0geoV78e2KLC41sfasw?e=kfukTw).
Extract and put it in a directory, then specific the argument `trained_model` like tokenizers before.

## Updates for Code-AST Prediction (CAP) Task

Executing the `CAP` task caused a runtime error so that the following updates have been made to execute the `forward()` function in `bart.py` while avoiding the runtime exception by `ValueError` below.

```shell
# pre-training
python main.py --do-pre-train --pre-train-tasks cap --batch-size 16 --eval-batch-size 32 --cuda-visible-devices 0 --fp16 --model-name pre_train --n-epoch 1 --n-epoch-pre-train 1 --remove-existing-saved-file fine_tune --copy-existing-saved-file pre_train_org
```

### pre_train.py
```
for task in tasks:
    if task == enums.TASK_CODE_AST_PREDICTION:
        + model.set_model_mode(enums.MODEL_MODE_GEN) 
        - model.set_model_mode(enums.MODEL_MODE_CLS)  
```

### bart.py
```
class BartForClassificationAndGeneration(BartForConditionalGeneration):
    def forward(self,...):
        if self.mode == enums.MODEL_MODE_GEN:
            return self.forward_gen(input_ids=input_ids,
            ...
        else:
            raise ValueError # <- caused an runtime exception.
```

### bug-fix
In addition, a new error introduced by the above edits was resolved by the following update in `data_collator.py`. Appending `unsqueeze(-1)` adds an extra dimension to `model_inputs['labels']`. Otherwise, the runtime error occurred due to dimension mismatch.

### code update
```
- model_inputs['labels'] = torch.tensor(is_ast, dtype=torch.long)
+ model_inputs['labels'] = torch.tensor(is_ast, dtype=torch.long).unsqueeze(-1)
```

### bug
```
    nll_loss = log_probs.gather(dim=-1, index=labels)
RuntimeError: Index tensor must have the same number of dimensions as input tensor
```

## Runs

Run `main.py` to start pre-train, fine-tune or test. 
All arguments are located in `args.py`, specific whatever you need.

Some example scripts are as following.
```shell
# pre-training
python main.py \
--do-pre-train \
--pre-train-tasks cap,mass,mng \
--batch-size 64 \
--eval-batch-size 64 \
--cuda-visible-devices 0,1,2,3 \
--fp16 \
--model-name pre_train

python main.py \
--do-pre-train \
--pre-train-tasks cap,mass,mng \
--batch-size 64 \
--eval-batch-size 64 \
--cuda-visible-devices 0 \
--fp16 \
--model-name pre_train

# summarization on pre-trained model and vocab
python main.py \
--do-fine-tune \
--task summarization \
--summarization-language java \
--model-name summarization_java \
--trained_vocab '../pre_trained/vocabs/' \
--trained_model '../pre_trained/models/all/'

# bug fixing without pre-training
python main.py \
--do-fine-tune \
--train-from-scratch \
--task bug_fix \
--bug_fix_scale medium

# only test on translation
python main.py \
--only-test \
--task translation \
--translation-source-language java \
--translation-target-language c_sharp \
--trained_vocab '../pre_trained/vocabs/' \
--trained_model '../outputs/translation_java_c_sharp_20210826_052653/models/'
```
