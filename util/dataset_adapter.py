from . import stable_random
from . import hgf_dataset_loader
from . import configs
import warnings

class triplet_dataset():
    # Split the dataset into three parts: calibration, demonstration, and test.
    def __init__(self, 
                 overloader: hgf_dataset_loader.basic_datasets_loader, 
                 calibration_number = configs.STANDARD_SETTINGS["calibration_number"], 
                 demonstration_number = configs.STANDARD_SETTINGS["demonstration_number"], 
                 test_number = configs.STANDARD_SETTINGS["test_number"]
        ):
        if calibration_number != configs.STANDARD_SETTINGS["calibration_number"] or demonstration_number != configs.STANDARD_SETTINGS["demonstration_number"] or test_number != configs.STANDARD_SETTINGS["test_number"]:
            if overloader.get_dataset_name() == "financial_phrasebank":
                if calibration_number != configs.STANDARD_SETTINGS["split_for_FP"]["calibration_number"] or demonstration_number != configs.STANDARD_SETTINGS["split_for_FP"]["demonstration_number"] or test_number != configs.STANDARD_SETTINGS["split_for_FP"]["test_number"]:
                    warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
            else:
                warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        unsplited_dataset = overloader
        if len(unsplited_dataset) < calibration_number + demonstration_number + test_number:
            raise ValueError("The dataset is too small to split.")
        my_random = stable_random.stable_random()
        indexes = my_random.sample_unique_index_set(calibration_number + demonstration_number + test_number, len(unsplited_dataset))
        self.calibration, self.demonstration, self.test = overloader.split([indexes[:calibration_number], indexes[calibration_number:calibration_number+demonstration_number], indexes[calibration_number+demonstration_number:]])
        self.calibration.rename_dataset(overloader.get_dataset_name()+"-calibration")
        self.demonstration.rename_dataset(overloader.get_dataset_name()+"-demonstration")
        self.test.rename_dataset(overloader.get_dataset_name()+"-test")
    
    def __str__(self) -> str:
        return ("Calibration set: " + self.calibration.__str__() + "\nDemonstration set: " + self.demonstration.__str__() + "\nTest set: " + self.test.__str__())
    
    def __repr__(self) -> str:
        ret = "--- Triplet Dataset ---" + "\n"
        ret += "Calibration set: " + self.calibration.__repr__() + "\nDemonstration set: " + self.demonstration.__repr__() + "\nTest set: " + self.test.__repr__()
        return ret

    def change_instruction_triple(self, instruction: str):
        self.calibration.change_instruction(instruction)
        self.demonstration.change_instruction(instruction)
        self.test.change_instruction(instruction)
    
    def change_input_text_prefixes_triple(self, input_text_prefixes: list[str]):
        self.calibration.change_input_text_prefixes(input_text_prefixes)
        self.demonstration.change_input_text_prefixes(input_text_prefixes)
        self.test.change_input_text_prefixes(input_text_prefixes)
    
    def change_input_text_affixes_triple(self, input_text_affixes: list[str]):
        self.calibration.change_input_text_affixes(input_text_affixes)
        self.demonstration.change_input_text_affixes(input_text_affixes)
        self.test.change_input_text_affixes(input_text_affixes)

    def change_label_prefix_triple(self, label_prefix: str):
        self.calibration.change_label_prefix(label_prefix)
        self.demonstration.change_label_prefix(label_prefix)
        self.test.change_label_prefix(label_prefix)
    
    def change_label_affix_triple(self, label_affix: str):
        self.calibration.change_label_affix(label_affix)
        self.demonstration.change_label_affix(label_affix)
        self.test.change_label_affix(label_affix)

    def change_query_prefix_triple(self, query_prefix: str):
        self.calibration.change_query_prefix(query_prefix)
        self.demonstration.change_query_prefix(query_prefix)
        self.test.change_query_prefix(query_prefix)

    def change_label_space_triple(self, label_space: list[str]):
        self.calibration.change_label_space(label_space)
        self.demonstration.change_label_space(label_space)
        self.test.change_label_space(label_space)


class prompt_writter():
    # Prompt will be structured as: 
    # <prompt_writter.instruction> (notice that all the \n here are not default, you should add it if you want to split the instruction and the following input texts)
    # [ (for multiple-input tasks)
    #   <prompt_writter.input_text_prefixes[0]> <prompt_writter.triplet_dataset.demonstration.get_input_text(index)[0]> <prompt_writter.input_text_prefixes[0]>
    #   <prompt_writter.input_text_prefixes[1]> <prompt_writter.triplet_dataset.demonstration.get_input_text(index)[1]> <prompt_writter.input_text_prefixes[1]>
    #   ...
    #   <prompt_writter.label_prefix> <prompt_writter.label(index)> <prompt_writter.label_afffix>
    # ] * k (k = demostration numbers)
    # <prompt_writter.query_prefix>
    # [ (for multiple-input tasks)
    #   <prompt_writter.input_text_prefixes[0]> <prompt_writter.triplet_dataset.test.get_input_text(index)[0]> <prompt_writter.input_text_prefixes[0]>
    #   <prompt_writter.input_text_prefixes[1]> <prompt_writter.triplet_dataset.test.get_input_text(index)[1]> <prompt_writter.input_text_prefixes[1]>
    #   ...
    #   <prompt_writter.label_prefix> [MASKED]
    # ]
    def __init__(self, triplet_dataset: triplet_dataset):
        self.triplet_dataset = triplet_dataset
        self.instruction = self.triplet_dataset.demonstration.get_instruction()
        self.input_text_prefixes = self.triplet_dataset.demonstration.get_input_text_prefixes()
        self.input_text_affixes = self.triplet_dataset.demonstration.get_input_text_affixes()
        self.label_prefix = self.triplet_dataset.demonstration.get_label_prefix()
        self.label_affix = self.triplet_dataset.demonstration.get_label_affix()
        self.query_prefix = self.triplet_dataset.test.get_query_prefix()
        self.label_space = self.triplet_dataset.demonstration.get_label_space()
        self._random_for_example = stable_random.stable_random()
    
    def get_label_of_test_samples(self, query_index: int):
        if query_index < 0 or query_index >= len(self.triplet_dataset.test):
            raise ValueError("Index out of range.")
        return self.triplet_dataset.test.get_label(query_index)
    
    def __str__(self) -> str:
        return (
            "In-context Learning prompt writter of" + 
            " demonstrations from " + self.triplet_dataset.demonstration.get_dataset_name() + 
            " and queries from " + self.triplet_dataset.test.get_dataset_name() + 
            " with instruction: " + self.instruction + 
            " and input text prefixes: " + str(self.input_text_prefixes) + 
            " and input text affixes: " + str(self.input_text_affixes) + 
            " and label prefix: " + self.label_prefix + 
            " and label affix: " + self.label_affix + 
            " and query prefix: " + self.query_prefix + 
            " and label space: " + str(self.label_space)
        )
    
    def __repr__(self):
        ret = self.__str__() + "\n"
        ret += "------------An Example of Prompt------------\n"
        ret += self.example()
        ret += "\nWith:\n"
        ret += self.triplet_dataset.__repr__()
        ret += "\n-------------------------------------------"
        return ret

    def change_instruction(self, instruction: str):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(instruction) is not str:
            raise ValueError("Instruction should be a string.")
        self.instruction = instruction

    def change_input_text_prefixes(self, input_text_prefixes: list[str]):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(input_text_prefixes) is not list:
            raise ValueError("Input text prefixes should be a list.")
        for prefix in input_text_prefixes:
            if type(prefix) is not str:
                raise ValueError("Input text prefixes should be a list of strings.")
        if len(input_text_prefixes) != self.input_element_numbers:
            raise ValueError("The number of input text prefixes should be equal to the number of input elements.")
        self.input_text_prefixes = input_text_prefixes
    
    def change_input_text_affixes(self, input_text_affixes: list[str]):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(input_text_affixes) is not list:
            raise ValueError("Input text affixes should be a list.")
        for affix in input_text_affixes:
            if type(affix) is not str:
                raise ValueError("Input text affixes should be a list of strings.")
        if len(input_text_affixes) != self.input_element_numbers:
            raise ValueError("The number of input text affixes should be equal to the number of input elements.")
        self.input_text_affixes = input_text_affixes
    
    def change_label_prefix(self, label_prefix: str):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(label_prefix) is not str:
            raise ValueError("Label prefix should be a string.")
        self.label_prefix = label_prefix
    
    def change_label_affix(self, label_affix: str):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(label_affix) is not str:
            raise ValueError("Label affix should be a string.")
        self.label_affix = label_affix
    
    def change_query_prefix(self, query_prefix: str):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(query_prefix) is not str:
            raise ValueError("Query prefix should be a string.")
        self.query_prefix = query_prefix

    def change_label_space(self, label_space: list[str]):
        warnings.warn("You are editing the standard settings of StaICC. You should not use the result after editing as any baselines. Be careful.")
        if type(label_space) is not list:
            raise ValueError("Label space should be a list.")
        for label in label_space:
            if type(label) is not str:
                raise ValueError("Label space should be a list of strings.")
        self.label_space = label_space
    
    def write_prompt(self, demos_indexes: list[int], query_index: int):
        if query_index < 0 or query_index >= len(self.triplet_dataset.test):
            raise ValueError("Index out of range.")
        prompt = self.instruction

        for demosindex in demos_indexes:
            if demosindex < 0 or demosindex >= len(self.triplet_dataset.demonstration):
                raise ValueError("Index out of range.")
            for i in range(self.triplet_dataset.demonstration.get_input_element_numbers()):
                prompt += self.input_text_prefixes[i] + self.triplet_dataset.demonstration.get_input_text(demosindex)[i] + self.input_text_affixes[i]
            prompt += self.label_prefix + self.triplet_dataset.demonstration.get_label(demosindex) + self.label_affix
        
        prompt += self.query_prefix
        for i in range(self.triplet_dataset.test.get_input_element_numbers()):
            prompt += self.input_text_prefixes[i] + self.triplet_dataset.test.get_input_text(query_index)[i] + self.input_text_affixes[i]
        prompt += self.label_prefix
        return prompt
    
    def example(self, k = 8):
        if k < 0 or k > len(self.triplet_dataset.demonstration):
            raise ValueError("Invalid number of demonstrations.")
        Dindexes = self._random_for_example.sample_unique_index_set(k, len(self.triplet_dataset.demonstration))
        Qindex = self._random_for_example.get_int_from_range(0, len(self.triplet_dataset.test))
        return self.write_prompt(Dindexes, Qindex)