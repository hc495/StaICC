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

## Quick Start

A standard process of the usage of StaICC is shown as below.

### Write your ICL inference

You should write a function or partial function with a prototype `my_function(prompt, label_space)`. Make sure the name of the formal parameter is consistent with the above. __Typically__, the parameter `prompt` is fed with a `str` variable with a ICL-formatted string, and the `label_space` is fed with a `list[str]` to describe which token in the vocabulary should the model focus as the label.

You can refer to the code in `prefabricate_inference/model_kernel.py` as examples

## Custom Experiment

## Examples

## Benchmark Results

You are welcome to use issue to report your own results.

See 'issue' for further information.

## Detailed Documentation

## Citation

