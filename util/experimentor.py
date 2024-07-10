from . import configs
from . import dataset_interface
from . import functional
import copy
import warnings

class single_experimentor():
    def __init__(self, 
        triplet_dataset = None, 
        original_dataset = None, 
        k: int = 4, 
        metrics: dict = {
            "accuracy": functional.accuracy,
            "averaged_truelabel_likelihood": functional.averaged_truelabel_likelihood,
            "macro_F1": functional.macro_F1,
            "expected_calibration_error_1": functional.expected_calibration_error_1
        }, # DICT: {metric_name: metric_function}  metric_function: (ground_truth: list[int], prediction: list[list[float]] <logits>) -> float
        repeat_times = configs.STANDARD_SETTINGS["test_times_for_each_test_sample"],
        dividing = [configs.STANDARD_SETTINGS["calibration_number"], configs.STANDARD_SETTINGS["demonstration_number"], configs.STANDARD_SETTINGS["test_number"]] # A list of integers that divides the test samples into 3 splits. The first split will be used for calibration, the second split will be used for demonstration, and the third split will be used for testing. Only can be used when original_dataset is given.
    ):
        if k < 0:
            raise ValueError("k should be a positive integer.")
        if repeat_times < 0:
            raise ValueError("repeat_times should be a positive integer.")

        if repeat_times != configs.STANDARD_SETTINGS["test_times_for_each_test_sample"]:
            warnings.warn(configs.WARNING_SETTINGS["tampering"])
        
        if triplet_dataset is None and original_dataset is None:
            raise ValueError("You should provide at least one dataset.")
        if triplet_dataset is not None and original_dataset is not None:
            raise ValueError("You should provide only one dataset.")
        
        self._k = k
        if triplet_dataset is not None:
            self.triplet_dataset = triplet_dataset
        if original_dataset is not None:
            self.triplet_dataset = dataset_interface.triplet_dataset(original_dataset, dividing[0], dividing[1], dividing[2])
        if k > len(self.triplet_dataset.demonstration):
            raise ValueError("k should be less than the length of the demonstration dataset.")
        
        self.prompt_former = dataset_interface.prompt_writter(self.triplet_dataset)
        self._demonstration_sampler = dataset_interface.demonstration_sampler(self._k, len(self.triplet_dataset.demonstration), repeat_times * len(self.triplet_dataset.test))
        self._default_demonstration_sampler = copy.deepcopy(self._demonstration_sampler)
        self._default_repeat_times = repeat_times
        self._repeat_times = repeat_times
        self.metrics = metrics

    def __call__(self, forward_inference: callable):
        return self.auto_run(forward_inference)
    
    def __str__(self) -> str:
        ret = ("--- single experimentor ---\n" +
        "\nTriplet Dataset: " + str(self.triplet_dataset) +
        "\nPrompt Former: " + str(self.prompt_former) +
        "\nK: " + str(self._k) +
        "\nMetrics: " + str(self.metrics) +
        "\nSamples for each test sample: " + str(self._repeat_times))
        return ret

    def __repr__(self) -> str:
        return self.__str__()

    def _get_prompts_for_test_sample(self, test_sample_index: int, repeat_time: int):
        # repeat_time_from_0
        demos_indexes = self._demonstration_sampler[test_sample_index + repeat_time * len(self.triplet_dataset.test)]
        return self.prompt_former.write_prompt(demos_indexes, test_sample_index)

    def reset_demonstration_sampler(self):
        self._demonstration_sampler = copy.deepcopy(self._default_demonstration_sampler)
        self._repeat_times = self._default_repeat_times
    
    def set_demonstration_sampler(self, sampler):
        # The sampler can be a list-shaped list of integers. 
        # The self._default_repeat_times will be set to 1 since no repeat time is needed with a fixed sampler.
        # The demonstrations will be sampled as: sampler[test_sample_index].
        # For example: when the sampler is: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], the demonstrations for the first test sample will be the [1, 2, 3] samples in the demonstration set and so on.
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if len(sampler) != len(self.triplet_dataset.test):
            raise ValueError("The length of the sampler should be equal to the number of the test samples.")
        if all([len(x) != self._k for x in sampler]):
            raise ValueError("The length of each sample in the sampler should be equal to k.")
        self._demonstration_sampler = sampler
        self._repeat_times = 1

    def auto_run(
        self, 
        forward_inference: callable # forward_inference: (prompt: str) -> list[float] <logits> or int <label>
    ):
        # The forward_inference function should be a callable that takes a prompt and returns a list of label logits or a label index.
        # We encourage the forward_inference function to be a function that takes a prompt and returns a list of logits for each label, so that we can calculate more metrics.
        # >> If you use a function that returns a label index, the metrics that require logits will be calculated as if the logits are one-hot encoded.
        print("Start testing the forward inference function " + str(forward_inference) + " on the dataset: " + str(self.triplet_dataset.test.dataset_name))
        ground_truth = []
        prediction = []
        total_samples = len(self.triplet_dataset.test) * self._repeat_times
        for time in range(self._repeat_times):
            for index in range(len(self.triplet_dataset.test)):
                prompt = self._get_prompts_for_test_sample(index, time)
                result = forward_inference(prompt)
                ground_truth.append(self.triplet_dataset.get_default_ground_truth_label_index(index))
                prediction.append(result)
                print("\r", end="")
                print("Process: {}%, {} in {}".format(
                    int((index + time * len(self.triplet_dataset.test) + 1) / total_samples * 100), 
                    (index + time * len(self.triplet_dataset.test) + 1), 
                    total_samples
                ), ">>" * int((index + time * len(self.triplet_dataset.test)) / total_samples * 32), end="")
        ret = {}
        for metric_name, metric_function in self.metrics.items():
            ret[metric_name] = metric_function(ground_truth, functional.extend_onehot_prediction_to_logits(prediction))
        return ret
    
    def calibration_set(self):
        return self.triplet_dataset.calibration
    
    def demonstration_set(self):
        return self.triplet_dataset.demonstration
    
    def test_set(self):
        return self.triplet_dataset.test