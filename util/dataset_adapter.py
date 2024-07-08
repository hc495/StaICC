from . import stable_random
from . import basic_datasets_loader
from . import configs
import warnings

class triplet_dataset():
    def __init__(self, 
                 overloader: basic_datasets_loader, 
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
        return (self.calibration.__str__() + "\n" + self.demonstration.__str__() + "\n" + self.test.__str__())
    
    def __repr__(self) -> str:
        return (self.calibration.__repr__() + "\n" + self.demonstration.__repr__() + "\n" + self.test.__repr__())

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