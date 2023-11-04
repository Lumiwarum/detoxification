
import warnings
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from os.path import exists

warnings.filterwarnings("ignore")


def translate(model, inference_request, tokenizer):
    """
    This function allows to run the model

    Args:
        model (transformers.modeling_utils.PreTrainedModel): model to use for inference
        inference_request (str): input string to transform
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer): tokenizer to use for inference
    """
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)



if __name__ == "__main__":
    done_pretrain = exists("best")
    print("Please, enter a sentence to detoxify into the console")
    input1 = input()
    input_prompt = "Detoxify "+ input1

    # load model
    if done_pretrain:
        model = AutoModelForSeq2SeqLM.from_pretrained("best", local_files_only=True)
    else:
        print("Unable to load the fine-tuned model. Using the pretrained from skolkovo")
        model = AutoModelForSeq2SeqLM.from_pretrained('SkolkovoInstitute/bart-base-detox')

    # get tokenizer from model
    base_model_name = 'facebook/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # translate
    result = translate(model.to("cpu"), input_prompt, tokenizer)
    print("Model result:", result)
