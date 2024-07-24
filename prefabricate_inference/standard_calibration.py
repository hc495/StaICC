from ..util import stable_random
from ..util import functional

class calibration():
    def __init__(self) -> None:
        pass

    def train(self) -> None:
        pass

    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        pass

    def __call__(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return self.inference(label_space_prob, full_vocab_prob, hidden_state)


class contextual_calibration(calibration):
    # https://arxiv.org/abs/2102.09690
    def __init__(self, label_space) -> None:
        self.label_space = label_space
        n_label = len(label_space)
        self.n_label = n_label
        self.calibrationA = [1e-5] * n_label
    
    def train(
        self, 
        default_prompt_maker: callable, # input: demos_lines: <list[(list[str], str)]>, query_line: <list[str]> return: prompt, recommendation: prompt_writter.write_prompt_from_dataline
        feedforward: callable, # feedforward function, input: prompt: <str> return: label_space_prob
        calibration_set = None,
        calibration_number = 128,
        k = 4
    ) -> None:
        my_random = stable_random.stable_random()
        demonstration_samples = my_random.sample_index_set(calibration_number * k, len(calibration_set), allow_repetition=True)
        false_data_line = ["" for _ in range(len(calibration_set[0][0]))]
        for i in range(calibration_number):
            prompt = default_prompt_maker([calibration_set[demonstration_samples[j]] for j in range(i * k, (i + 1) * k)], false_data_line)
            label_space_prob = feedforward(prompt = prompt, label_space = self.label_space)
            self.calibrationA = [self.calibrationA[j] + label_space_prob[j] for j in range(self.n_label)]
        self.calibrationA = [self.calibrationA[j] / calibration_number for j in range(self.n_label)]
    
    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return functional.softmax([label_space_prob[j] / self.calibrationA[j] for j in range(self.n_label)])


class domain_calibration(calibration):
    # https://arxiv.org/abs/2305.19148
    def __init__(self, label_space) -> None:
        self.label_space = label_space
        n_label = len(label_space)
        self.n_label = n_label
        self.calibrationA = [1e-5] * n_label
    
    def _get_domain_sampleline(self, calibration_set, sample_length):
        my_random = stable_random.stable_random()
        while True:
            ret = []
            for i in range(len(calibration_set[0][0])):
                output = []
                while len(output) < sample_length:
                    random_sample = calibration_set[my_random.get_int_from_range(0, len(calibration_set) - 1)][0][i]
                    random_sample = random_sample.split(' ')
                    random_index = my_random.get_int_from_range(0, len(random_sample) - 1)
                    output.append(random_sample[random_index])
                output = ' '.join(output)
                ret.append(output)
            yield ret

    def train(
        self, 
        default_prompt_maker: callable, # input: demos_lines: <list[(list[str], str)]>, query_line: <list[str]> return: prompt, recommendation: prompt_writter.write_prompt_from_dataline
        feedforward: callable, # feedforward function, input: prompt: <str> return: label_space_prob
        calibration_set = None,
        calibration_number = 128,
        sample_length = 64,
        k = 4
    ) -> None:
        my_random = stable_random.stable_random()
        demonstration_samples = my_random.sample_index_set(calibration_number * k, len(calibration_set), allow_repetition=True)
        gen = self._get_domain_sampleline(calibration_set, sample_length)
        false_data_line = next(gen)
        for i in range(calibration_number):
            prompt = default_prompt_maker([calibration_set[demonstration_samples[j]] for j in range(i * k, (i + 1) * k)], false_data_line)
            label_space_prob = feedforward(prompt = prompt, label_space = self.label_space)
            self.calibrationA = [self.calibrationA[j] + label_space_prob[j] for j in range(self.n_label)]
        self.calibrationA = [self.calibrationA[j] / calibration_number for j in range(self.n_label)]
    
    def inference(self, label_space_prob, full_vocab_prob, hidden_state) -> list[float]:
        return functional.softmax([label_space_prob[j] / self.calibrationA[j] for j in range(self.n_label)])
    

def batch_calibration(
    label_space_probs: list[list[float]], 
    batch_size = 128, 
) -> list[list[float]]:
    ret = []
    step = len(label_space_probs) // batch_size
    for i in range(step):
        batch = label_space_probs[i * batch_size: (i + 1) * batch_size]
        mean_bias = [0] * len(batch[0])
        for j in range(batch_size):
            for k in range(len(batch[j])):
                mean_bias[k] += batch[j][k]
        mean_bias = [x / batch_size for x in mean_bias]
        for j in range(batch_size):
            ret.append(functional.softmax([batch[j][k] - mean_bias[k] for k in range(len(batch[j]))]))
    last_batch = label_space_probs[step * batch_size:]
    if len(last_batch) == 0:
        return ret
    mean_bias = [0] * len(last_batch[0])
    for j in range(len(last_batch)):
        for k in range(len(last_batch[j])):
            mean_bias[k] += last_batch[j][k]
    mean_bias = [x / len(last_batch) for x in mean_bias]
    for j in range(len(last_batch)):
        ret.append(functional.softmax([last_batch[j][k] - mean_bias[k] for k in range(len(last_batch[j]))]))
    return ret