STANDARD_SETTINGS = {
    "calibration_number": 1024,
    "demonstration_number": 4096,
    "test_number": 512,
    "test_times_for_each_test_sample": 2,
    "cut_by_length_remain_short": 1024,
    "cut_by_length_remain_long": 8192,
    "ece_bins": 10,
    "random_seed": 42,
    "random_A": 1664525,
    "random_B": 1013904223,
    "random_C": 2**32,
    "split_for_FP": {
        "calibration_number": 1024,
        "demonstration_number": 512,
        "test_number": 512
    },
    "split_for_TEE": {
        "calibration_number": 1024,
        "demonstration_number": 3192,
        "test_number": 512
    },
}

STRICT_MODE = True

WARNING_SETTINGS = {
    "tampering": "You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.",
    "FP_length_warning": "We are spliting the financial_phrasebank with a shorter dataset length. The default spliting can't be remained. Be careful.",
    "basic_dataset_template_protect": "You are editing the basic dataset template in the strict mode. Canceled.\n If you want to edit the prompt template, please edit the dataset_interface.prompt_writter.",
    "strict_mode_protect": "The setting can't be changed in the strict mode. Return to default."
}