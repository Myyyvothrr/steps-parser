# STEPS Parser

[![License](https://img.shields.io/badge/license-AGPL%20v3-orange)](LICENSE)

This repository contains the companion code for STEPS, the modular [Universal Dependencies](
https://universaldependencies.org/) parser described in the paper

> [Stefan Grünewald, Annemarie Friedrich, and Jonas Kuhn (2020): **Applying Occam's Razor to Transformer-Based Dependency Parsing: What Works, What Doesn't, and What is Really Necessary.** arXiv:2010.12699](https://arxiv.org/abs/2010.12699)

The code allows users to reproduce and extend the results reported in the study.
Please cite the above paper when using our code, and direct any questions or 
feedback regarding our parser at [Stefan Grünewald](mailto:stefan.gruenewald@de.bosch.com).

### Disclaimer: Purpose of the Project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

<img src="img/steps_model.png" alt="STEPS parser architecture" width="400"/>


## Project Structure

The repository has the following directory structure:
```
steps-parser/
    configs/                            Example configuration files for the parser
    data/
        corpora/                        Folder to put training/validation UD corpora
            en_ewt/                     Folder for English-EWT corpus files
            lv_lvtb/                    Folder for Latvian-LVTB corpus files
            download_corpora.sh         Script for downloading example corpus files (English-EWT, Latvian-LVTB)
            delexicalize_corpus.py      Script for replacing lexical material in enhanced dependency labels with placeholders in a UD corpus
        pretrained-embeddings/          Folder for pre-trained word embeddings
            download_embeddings.sh      Script for downloading pre-trained word embeddings
        saved_models/                   Folder for saved models
            download_models.sh          Script for downloading trained parser models (forthcoming)
    src/
        data_handling/                  Code for processing dependency-annotated sentence data
        logger/                         Code for logging (--> boring)
        models/                         Actual model code (parser, classifier)
            embeddings/                 Code for handling contextualized word embeddings
            outputs/                    Code for output modules (arc scorer, label scorer etc.)
            post_processing/            Dependency tree/graph post-processing
            multi_parser.py             The main class for computing outputs from input embeddings
        trainer/                        Training logic, loss scheduling, evaluation
        util/                           Util scripts, e.g. for maximum spanning trees and label lexicalization
        init_config.py                  Initialization of model, trainer, and data loaders
        parse_corpus.py                 Main script for parsing UD corpora using a trained model
        parse_raw.py                    Main script for parsing raw text using a trained model
        train.py                        Main script for training models
    environment.yml                     Conda environment file for STEPS
```


## Using the Parser

### Requirements
STEPS requires the following dependencies:
* [Python](https://www.python.org/) == 3.7.7
* [Huggingface Transformers](https://github.com/huggingface/transformers) == 3.1.0
* [MLFlow](https://mlflow.org/)
* [Stanza](https://stanfordnlp.github.io/stanza/)
* [pyconll](https://github.com/pyconll/pyconll/)

You can install all the above dependencies easily using [Conda](https://docs.conda.io/en/latest/)
and the ```environment.yml``` file provided by us:
```bash
conda env create -f environment.yml
conda activate stepsenv
```

For **post-training evaluation**, you will also need to download the [conll18_ud_eval.py](http://universaldependencies.org/conll18/conll18_ud_eval.py)
and [iwpt20_xud_eval.py](https://universaldependencies.org/iwpt20/iwpt20_xud_eval.py) scripts and replace the respective placeholder
files in `src/util/` with them.
Note that we are **not distributing** these files with our code due to licensing complications.

**Example corpus files** (English-EWT, Latvian-LVTB) and **example transformer language models** (mBERT, XLM-R-large) can be downloaded by running the 
`download_corpora.sh` and `download_embeddings.sh` scripts in the respective folders.

**Note:** Since the model files are quite large, the downloads might take a long time depending on your internet connection. You may want to edit the
download scripts in order to download only the particular models you are actually interested in (see comments in the respective scripts).


### Training Models
To train your own parser model, run `python src/train.py [CONFIG_PATH] -e [EVALUATION_MODE]`.

We have provided two example configuration files to get started:
* **English-EWT, basic dependencies, mBERT:** `python src/train.py configs/en-basic-mbert.json -e basic`
* **English-EWT, enhanced dependencies, XLM-R:** `python src/train.py configs/lv-enhanced-xlmr.json -e enhanced`

(Note: Training is GPU memory intensive, particularly for the XLM-R-large model. If you run out of memory, try 
reducing the `batch_size` in the configuration files while lowering the learning rate and raising the
number of warmup steps in `lr_lambda` accordingly.)

For these configurations, you will need to download the example corpora and transformer language models (see above).
After training, the respective trained models will be saved to the `saved_models` folder.


### Parsing Corpora Using a Trained Model
**Note:** Trained models for STEPS will be released in the coming days. For now, you can already download and try trained models for English, which are available [on Zenodo](https://zenodo.org/record/4614023#.YFJZbv4o_IE).

To parse a given corpus from a CoNLL-U file, run `python src/parse_corpus.py [MODEL_DIR] [CORPUS_FILENAME] -o [OUTPUT_FILENAME]`.

You can also evaluate against the input corpus directly after parsing. To do so, add the following options:
* `-e basic` for basic dependency parsing (this will use the `conll18_ud_eval.py` script for evaluation)
* `-e enhanced -k 6 7` for enhanced dependency parsing (this will use the `iwpt20_xud_eval.py` script for evaluation and copy over
   the basic layer annotation columns)

To parse raw text (using the Stanza tokenizer for the provided language code), run `python src/parse_raw.py [MODEL_DIR] [LANGUAGE_CODE] [TEXT_CORPUS_FILENAME] -o [OUTPUT_FILENAMEs]`.

Note: Make sure to download the appropriate Stanza model first (e.g. `stanza.download(lang="en", processors="tokenize,mwt")`).

## License
STEPS is open-sourced under the AGPL v3 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in STEPS, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

The software, including its dependencies, may be covered by third party rights, including patents.
You should not execute this code unless you have obtained the appropriate rights, which the authors
are not purporting to give.

