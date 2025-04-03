from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
model_id="mistralai/Mistral-7B-Instruct-v0.3"
def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=0.1):
    """
    Returns a language model for HuggingFace inference.

    Parameters:
    - model_id (str): The ID of the HuggingFace model repository.
    - max_new_tokens (int): The maximum number of new tokens to generate.
    - temperature (float): The temperature for sampling from the model.

    Returns:
    - llm (HuggingFaceEndpoint): The language model for HuggingFace inference.
    """
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token = os.getenv("HF_TOKEN")
    )
    return llm
