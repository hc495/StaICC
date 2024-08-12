from . import normal
from .util import configs, functional, experimentor
import copy

class triplet_bias():
    def __init__(self):
        self.contextual = contextual_bias()
        self.domain = domain_bias()
        self.post = post_bias()
    
    def __call__(
        self, 
        list_of_forward_inference: list[callable], # for each dataset, you should give a forward_inference function. If you just give one, we will expand it to the length of the benchmark.
        return_divided_results = True,
        batched_inference = False
    ):
        return self.auto_run(list_of_forward_inference, return_divided_results, batched_inference)
    
    def auto_run(self, list_of_forward_inference, return_divided_results, batched_inference):
        return {
            "contextual": self.contextual.auto_run(list_of_forward_inference, return_divided_results, batched_inference),
            "domain": self.domain.auto_run(list_of_forward_inference, return_divided_results, batched_inference),
            "post": self.post.auto_run(list_of_forward_inference, return_divided_results, batched_inference)
        }

class contextual_bias(normal.benchmark):
    def __init__(
        self, 
        k = 4,
        noisy_channel = False,
        metrics: dict = {
            "entropy": functional.bias_mean_entropy_metric,
            "distribution": functional.bias_mean_metric,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        print("Loading data...\n")
        count = 0
        for data_loader in self._default_data:
            self._original_data.append(data_loader())
            count += 1
            print("{} in {}".format(count, len(self._default_data)), "Data loaded: ", self._original_data[-1].get_dataset_name(), "\n")

        print("Data loaded successfully.\n")
        self.metrics = metrics
        self.noisy_channel = noisy_channel

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)

    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False): 
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "contextual"
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "contextual"
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel, bias_type = "contextual")
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")


class domain_bias(normal.benchmark):
    def __init__(
        self, 
        k = 4,
        noisy_channel = False,
        metrics: dict = {
            "entropy": functional.bias_mean_entropy_metric,
            "distribution": functional.bias_mean_metric,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
        domain_query_length = 128
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        print("Loading data...\n")
        count = 0
        for data_loader in self._default_data:
            self._original_data.append(data_loader())
            count += 1
            print("{} in {}".format(count, len(self._default_data)), "Data loaded: ", self._original_data[-1].get_dataset_name(), "\n")

        print("Data loaded successfully.\n")
        self.metrics = metrics
        self.noisy_channel = noisy_channel
        self.domain_query_length = domain_query_length

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)

    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False): 
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "domain",
                        domain_query_length = self.domain_query_length
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        bias_type = "domain",
                        domain_query_length = self.domain_query_length
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.prior_bias_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel, bias_type = "domain", domain_query_length = self.domain_query_length)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")


class post_bias(normal.benchmark):
    def __init__(
        self, 
        k = 4,
        noisy_channel = False,
        metrics: dict = {
            "DL div.": functional.post_bias_dl_metric,
            "distribution": functional.post_bias_dis_metric,
        },
        datasets = normal.ORIGINAL_DATA_LOADER_NORMAL,
        domain_query_length = 128
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = datasets
        print("Loading data...\n")
        count = 0
        for data_loader in self._default_data:
            self._original_data.append(data_loader())
            count += 1
            print("{} in {}".format(count, len(self._default_data)), "Data loaded: ", self._original_data[-1].get_dataset_name(), "\n")

        print("Data loaded successfully.\n")
        self.metrics = metrics
        self.noisy_channel = noisy_channel
        self.domain_query_length = domain_query_length

        self.re_initialize(k = k, noisy_channel = self.noisy_channel)

    def re_initialize(self, k: int = 4, noisy_channel = False, keep_prompter = False): 
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.post_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]],
                        noisy_channel = noisy_channel,
                        )
                    )
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.post_bias_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]],
                        noisy_channel = noisy_channel,
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.post_bias_experimentor(original_dataset = data, k=k, metrics=self.metrics, noisy_channel=noisy_channel)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1
        print("Ready.\n")