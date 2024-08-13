# StaICC: Standardized Benchmark for In-context Classification

This is a standard implementation of paper "StaICC: Standardized Benchmark for In-context Classification" by Hakaze Cho.

## Content

1. [Installation](#installation)
2. [Introduction](#introduction)
3. [Quick Start](#quick-start)
4. [Custom Experiment](#custom-experiment)
5. [Examples](#examples)
6. [Benchmark Results](#benchmark-results)
7. [Citation](#citation)
8. [Detailed Documentation](#detailed-documentation)

## Installation

We ensure that under normal usage, this library only relies on Python's default dependency libraries.

You can only download a release pack of StaICC and unfold it into your work path with the top folder `StaICC` in your work path, like:

```
---- work_path
 |-- StaICC
 |-- experiment_code.py
 ...
```

Also, we release PyPl package `StaICC`. You can use:

```
pip install StaICC
```

to install this library.

## Introduction

### Sub-benchmarks

| Name | Import name | Describe |
|:---:|:---:|:---:|
| StaICC-Normal | `from StaICC import Normal` | A standard classification accuracy-based benchmark for normal classification tasks. |
| StaICC-Diagnosis: Bias | `from StaICC import Triplet_bias` | A prediction logits bias (3 types) detector. |

## Quick Start

A standard process of the usage of StaICC is shown as below.

### 1. Write your ICL inference

You should write a function or partial function with a prototype `my_function(prompt, label_space)`. Make sure the name of the formal parameter is consistent with the above. __Typically__, the parameter `prompt` is fed with a `str` variable with a ICL-formatted string, and the `label_space` is fed with a `list[str]` to describe which token in the vocabulary should the model focus as the label.

You can refer to the functions in `prefabricate_inference/model_kernel.py` as examples. Also, as a quick start, you can reload these functions by `functools.partial` like:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("<huggingface_model_name>")
model = AutoModelForCausalLM.from_pretrained("<huggingface_model_name>").cuda()

from StaICC.prefabricate_inference import model_kernel
import functools

my_inference = functools.partial(
    model_kernel.standard_ICL_inference_with_torch_Causal_LM, 
    model = model, 
    tokenizer = tokenizer, 
    cache_empty = torch.cuda.empty_cache, 
    return_hidden_state = False, 
    return_full_vocab_prob = False
) 
```

### 2. Load a sub-benchmark and instantiate it

Choose one sub-benchmark introduced in [Introduction](#introduction). As a start, you can choose `StaICC-Normal` as a trial:

```python
from StaICC import Normal
benchmark = Normal()
```

### 3. Test your inference function

```python
result_dictionary = benchmark(inference)
```

A typical output is a dictionary as:

```python
'Divided results': {
    'GLUE-SST2': {
        'accuracy': 0.580078125,
        'averaged_truelabel_likelihood': 0.5380006989248597,
        'macro_F1': 0.49802777081100796,
        'expected_calibration_error_1': 0.08539132766414247},
    'rotten_tomatoes': {
        'accuracy': 0.5693359375,
        'averaged_truelabel_likelihood': 0.5196389624283041,
        'macro_F1': 0.5286056525483442,
        'expected_calibration_error_1': 0.03482341199255428},},
    ...
'Averaged results': {
    'accuracy': 0.40244140625,
    'averaged_truelabel_likelihood': 0.38355543726209307,
    'macro_F1': 0.2863139242653286,
    'expected_calibration_error_1': 0.33533775010105704}
```

## Custom Experiment

We support the following custom settings of expeirment.

1. [Demonstration numbers k](#k)
2. [Different inference function for different dataset](#list_inference)

<span id="k"></span>

### Use various demonstration numbers in your experiment




<span id="list_inference"></span>

### Use different inference function for each dataset




## Examples

## Benchmark Results

You are welcome to use issue to report your own results.

See 'issue' for further information.

## Detailed Documentation

## Citation

