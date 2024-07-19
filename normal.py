from .util import experimentor
from .util import hgf_dataset_loader
from .util import functional
from .util import configs
import copy
import warnings

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
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        ret = "--- Benchmark: StaICC Normal ---\n"
        for exp in self.experimentor:
            ret += str(exp) + "\n"
        return ret

    def re_initialize(self, k: int = 4, keep_prompter = False): # keep_prompter: UNTESTED
        print("Initializing experimentor on k = {}...\n".format(k))
        self.experimentor = []
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
            elif data.get_dataset_name() == "tweet_eval_emotion":
                self.experimentor.append(
                    experimentor.single_experimentor(
                        original_dataset = data, 
                        k=k, 
                        metrics=self.metrics, 
                        dividing=[configs.STANDARD_SETTINGS["split_for_TEE"]["calibration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["demonstration_number"], configs.STANDARD_SETTINGS["split_for_TEE"]["test_number"]]
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
        print("Ready.\n")
    
    def get_experiment_data(self):
        return [exp.triplet_dataset for exp in self.experimentor]

    def get_experimentors(self):
        return self.experimentor

    def get_label_spaces_for_experimentors(self):
        return [exp.triplet_dataset.get_label_space() for exp in self.experimentor]

    def auto_run(
        self, 
        list_of_forward_inference: list[callable], # for each dataset, you should give a forward_inference function. If you just give one, we will expand it to the length of the benchmark.
        return_divided_results = True
    ):
        count = 0
        if len(list_of_forward_inference) != len(self.experimentor):
            if len(list_of_forward_inference) == 1:
                list_of_forward_inference = list_of_forward_inference * len(self.experimentor)
            else:
                raise ValueError("The length of list_of_forward_inference must be the same as the number of datasets in the benchmark. You can use the get_experiment_data method to get the datasets and their order.")
        ret_divided = {}
        ret_sum = {}
        for name, metric in self.metrics.items():
            ret_sum[name] = 0
        for i, exp in enumerate(self.experimentor):
            count += 1
            print("\n\nExperiment {} in {}".format(count, len(self._default_data)))
            try:
                temp_res, success = exp(list_of_forward_inference[i])
            except Exception as r:
                print(r)
                success = False
                temp_res = {}
                for name, metric in self.metrics.items():
                    temp_res[name] = 0
            else:
                ret_divided[exp.triplet_dataset.dataset_name] = temp_res
            if not success:
                warnings.warn("The experimentor on the dataset " + exp.triplet_dataset.get_dataset_name() + " failed.")
                continue
            for name, metric in self.metrics.items():
                ret_sum[name] += temp_res[name]
        for name, metric in self.metrics.items():
            ret_sum[name] /= len(self.experimentor)
        
        if return_divided_results:
            return ret_divided, ret_sum
        else:
            return ret_sum