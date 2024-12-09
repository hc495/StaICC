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
--- work_path
 |--- StaICC
 | |- __init__.py
 | |- ...
 |--- experiment_code.py
 |--- ...
```

Also, we release PyPl package `StaICC`. You can use:

```
pip install StaICC
```

to install this library.

## Introduction

### Sub-benchmarks

`StaICC` provides several sub-benchmarks for in-context classification evaluations. The following table shows the sub-benchmarks we provide:

| Name | Import name | Describe |
|:---:|:---:|:---:|
| StaICC-Normal | `from StaICC import Normal` | A standard classification accuracy-based benchmark for normal classification tasks. |
| StaICC-Diagnosis: Bias | `from StaICC import Triplet_bias` | A prediction logits bias (3 types) detector. |

#### StaICC-Normal

`StaICC-Normal` is a standard classification accuracy-based benchmark for normal classification tasks. It returns the averaged metrics of accuracy, averaged truelabel likelihood, macro F1, and expected calibration error.

#### StaICC-Diagnosis: Bias

`StaICC-Diagnosis: Bias` is a prediction logits bias detector of 3 types:

1. **Contextual Bias**: Introduced by [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2106.06328), contextual bias measures the bias when some demonstrations and an empty query is fed into the model. We use the entropy of the averaged prediction probabilites as the metric.

2. **Domainal Bias**: Introduced by [Mitigating Label Biases for In-context Learning](http://arxiv.org/abs/2305.19148), domainal bias measures the bias when some demonstrations and a query of randomly sampled tokens from the test dataset is fed into the model. We use the entropy of the averaged prediction probabilites as the metric.

3. **Posterior Bias**: 

### Datasets

In StaICC, we use the following original datasets:

|Index| Name | Task | Label Space | Citation |
|:---:|:---:|:---:|:---:|:---:|
|0|GLUE-SST2 | Sentiment Classification | negative, positive | https://aclanthology.org/D13-1170/|
|1|Rotten Tomatoes | Sentiment Classification | negative, positive | https://arxiv.org/abs/cs/0506075|
|2|Financial Phrasebank | Sentiment Classification | negative, neutral, positive | https://arxiv.org/abs/1307.5336|
|3|SST5 | Sentiment Classification | very negative, negative, neutral, positive, very positive | https://aclanthology.org/D13-1170/|
|4|TREC | Topic Classification | abbreviation, entity, description and abstract concept, human being, location, numeric value | https://www.aclweb.org/anthology/C02-1150|
|5|AGNews | Topic Classification | world, sports, business, sci/tech | https://arxiv.org/abs/1509.01626|
|6|Subjective | Subjectivity Classification | objective, subjective|https://dl.acm.org/doi/10.5555/2390665.2390688|
|7|Tweet Eval Emotion | Sentiment Classification | anger, joy, optimism, sadness|https://aclanthology.org/S18-1001/|
|8|Tweet Eval Hate |Hate Speech Classification | non-hate, hate|https://aclanthology.org/S19-2007/|
|9|Hate Speech 18	|Hate Speech Classification| noHate, hate, idk/skip, relation| - |

## Quick Start

A standard process of the usage of StaICC is shown as below.

### 1. Write your ICL inference

You should write a function or partial function with a prototype `my_function(prompt, label_space)`. Make sure the name of the formal parameter is consistent with the above. __Typically__, the parameter `prompt` is fed with a `str` variable with a ICL-formatted string, and the `label_space` is fed with a `list[str]` to describe which token in the vocabulary should the model focus as the label.

You can refer to the functions in `prefabricate_inference/model_kernel.py` as examples. Also, as a quick start, you can reload these functions by `functools.partial` as shown below. (if you import `StaICC.prefabricate_inference.model_kernel`, make sure you have dependencies of `torch` and `transformers >= 4.43`)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from StaICC.prefabricate_inference import model_kernel
import functools

tokenizer = AutoTokenizer.from_pretrained("<huggingface_model_name>") 
model = AutoModelForCausalLM.from_pretrained("<huggingface_model_name>").cuda()

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

In our implementation, we use one `experimentor` object for each dataset. So, you can customize your experiment by setting the parameters of the `experimentor` object. One you load a sub-benchmark like `Normal`, you can access the `experimentor` object by `benchmark[dataset_index]`. For example, you can access the `experimentor` object of the `GLUE-SST2` dataset by `Normal[0]`.

When you access the `experimentor`, you can control the experiment. We support the following custom settings of experiment with respect to each step in the ICL pipeline:

### Demonstration Sampling

#### Various Demonstration Numbers

You are recommended to define the demonstration number in the initialization of the sub-benchmark. The default value is 4, and you can set the `k` parameter in the instantiation of the sub-benchmark.

For example:

```python
from StaICC import Normal
benchmark = Normal(k = 16)
```

- We didn't set a upper limit for this parameter, but some of the demonstrations may be repeated if the `k` exceed the existing number of the demonstration samples.

Also, you can set the expected demonstration number after the initialization by `experimentor.set_k(k)`.

#### Manual Demonstration Sampling

You can use `experimentor.set_demonstration_sampler(sampler)` function to manually sample the demonstrations for each test sample. You can input any list-styled object `sampler` with the same length as the test samples, and each element in your `sampler` should be a list of indices of the demonstrations you want to use for the corresponding test sample, in sequence.

In this processing, you are likely to need access to these demonstration and test sets. You can access them by `experimentor.demonstration_set` and `experimentor.test_set`. An example is shown [below](prompt_sample).

- Notice that setting a `sampler` will set the repeat experiment times from 2 to 1.

- To reset the `sampler` to default, you can call `experimentor.reset_demonstration_sampler()`.

## Examples

<span id="prompt_sample"></span>

### Use manual demonstration sequence in your experiment

<span id="list_inference"></span>

As an example, we repeat the k-NN demonstration experiment proposed by paper [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804). 

### Use different inference function for each dataset





## Benchmark Results

You are welcome to use issue to report your own results. 

See 'issue' for further information.

## Detailed Documentation

## Citation

