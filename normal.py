from .util import experimentor
from .util import hgf_dataset_loader
from .util import functional
from .util import configs
import copy

ORIGINAL_DATA_LOADER_NORMAL = [
    hgf_dataset_loader.glue_sst2,
    hgf_dataset_loader.rotten_tomatoes,
    hgf_dataset_loader.financial_phrasebank,
    hgf_dataset_loader.sst5,
    hgf_dataset_loader.trec,
    hgf_dataset_loader.agnews,
    hgf_dataset_loader.subjective,
    hgf_dataset_loader.tweet_eval_emotion,
    hgf_dataset_loader.tweet_eval_hate,
    hgf_dataset_loader.hate_speech_18,
]

class benchmark():
    def __init__(
        self, 
        k = 4,
        metrics: dict = {
            "accuracy": functional.accuracy,
            "averaged_truelabel_likelihood": functional.averaged_truelabel_likelihood,
            "macro_F1": functional.macro_F1,
            "expected_calibration_error_1": functional.expected_calibration_error_1
        }
    ):
        self.experimentor = []
        self._original_data = []
        self._default_data = ORIGINAL_DATA_LOADER_NORMAL
        print("Loading data...\n")
        count = 0
        for data_loader in self._default_data:
            self._original_data.append(data_loader())
            count += 1
            print("{} in {}".format(count, len(self._default_data)), "Data loaded: ", self._original_data[-1].get_dataset_name(), "\n")

        print("Data loaded successfully.\n")
        self.metrics = metrics

        self.re_initialize(k)

    def __call__(self, forward_inference: callable):
        return self.auto_run(forward_inference)

    def re_initialize(self, k: int = 4, keep_prompter = False):
        if keep_prompter:
            old_prompter = []
            for exp in self.experimentor:
                old_prompter.append(copy.deepcopy(exp.prompt_former))
        for data in self._original_data:
            if data.get_dataset_name() == "financial_phrasebank":
                self.experimentor.append(
                    experimentor.single_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]]
                        )
                    )
            else:
                self.experimentor.append(
                    experimentor.single_experimentor(original_dataset = data, k=k, metrics=self.metrics)
                )
        if keep_prompter:
            count = 0
            for exp in self.experimentor:
                exp.prompt_former = old_prompter[count]
                count += 1

    def auto_run(self, forward_inference: callable, return_divided_results = True):
        ret_divided = {}
        ret_sum = {}
        for name, metric in self.metrics.items():
            ret_sum[name] = 0
        for exp in self.experimentor:
            temp_res = exp(forward_inference)
            ret_divided[exp.triplet_dataset.dataset_name] = temp_res
            for name, metric in self.metrics.items():
                ret_sum[name] += temp_res[name]
        for name, metric in self.metrics.items():
            ret_sum[name] /= len(self.experimentor)
        
        if return_divided_results:
            return ret_divided, ret_sum
        else:
            return ret_sum