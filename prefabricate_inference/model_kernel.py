from ..util import functional
import torch

def standard_ICL_inference_with_torch_Causal_LM(
    prompt: str,
    model: callable,
    tokenizer: callable,
    label_space: list[str],
    cache_empty: callable = torch.cuda.empty_cache(), # GPU cache empty function. Can be torch.cuda.empty_cache.
    calibration_function: callable = None, # standard calibration receives label_space_prob, full_vocab_prob, hidden_state, returns probabilities distribution aligned to the label_space
):
    if cache_empty is not None:
        cache_empty()
    tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.cuda() # flexable??
    result = model(tknzd_data, output_hidden_states = True)
    full_vocab_prob = result['logits'][0][-1].detach().to(torch.float).cpu().numpy()
    last_hidden_state = result.hidden_states[-1][-1][-1].detach().to(torch.float).cpu().numpy()
    tokenized_label_space = [tokenizer(label).input_ids[-1] for label in label_space] # The last token only
    label_space_logits = [full_vocab_prob[token] for token in tokenized_label_space]
    label_space_prob = functional.softmax(label_space_logits)
    del result
    if calibration_function is not None:
        return calibration_function(label_space_prob, full_vocab_prob, last_hidden_state)
    else:
        return label_space_prob