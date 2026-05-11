from copy import deepcopy
from typing import Optional

import torch
from torch import nn


def linear_interpolation_merge(
    model_a: nn.Module, model_b: nn.Module, alpha: float
) -> nn.Module:
    """
    Merge two models using linear interpolation.

    Returns: alpha * model_a + (1 - alpha) * model_b

    Args:
        model_a: First model
        model_b: Second model (must have same architecture as model_a)
        alpha: Interpolation weight in [0, 1]
            - alpha = 1.0 returns model_a
            - alpha = 0.0 returns model_b
            - alpha = 0.5 returns average of both models

    Returns:
        Merged model with interpolated weights
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure both models are on the same device (GPU) for the merge computation
    model_a.to(device)
    model_b.to(device)

    # Check that both models have the same parameters
    keys_a = set(model_a.state_dict().keys())
    keys_b = set(model_b.state_dict().keys())
    if keys_a != keys_b:
        raise ValueError(
            "Models must have the same architecture (different parameter names)"
        )

    # Compute merged state dict by iterating over state_dict items
    # Use torch.no_grad() to avoid creating computation graph
    merged_state_dict = {}
    with torch.no_grad():
        state_dict_a = model_a.state_dict()
        state_dict_b = model_b.state_dict()

        for key in state_dict_a.keys():
            param_a = state_dict_a[key]
            param_b = state_dict_b[key]

            if param_a.shape != param_b.shape:
                raise ValueError(
                    f"Parameter {key} has different shapes: {param_a.shape} vs {param_b.shape}"
                )
            # Linear interpolation on GPU, then move to CPU for the merged state dict
            merged_state_dict[key] = (alpha * param_a + (1 - alpha) * param_b).cpu()

        # Explicitly delete state dicts to free GPU memory
        del state_dict_a, state_dict_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create merged model entirely on CPU (deepcopy + load), then move to GPU
    # This avoids having 3 full models on GPU simultaneously
    original_device = next(model_a.parameters()).device
    model_a_cpu = model_a.cpu()
    merged_model = deepcopy(model_a_cpu)
    merged_model.load_state_dict(merged_state_dict)
    model_a.to(original_device)  # Move model_a back to original device
    merged_model.to(device)  # Move merged model to GPU

    return merged_model


def merge_olmo_models(
    model_a,
    model_b,
    alpha: float = 0.5,
    output_dir: Optional[str] = None,
):
    """
    Merge two OLMo models using linear interpolation.

    Returns: alpha * model_a + (1 - alpha) * model_b

    Args:
        model_a: First model (OLMoForCausalLM instance or path/model_name)
        model_b: Second model (OLMoForCausalLM instance or path/model_name)
        alpha: Interpolation weight in [0, 1], default 0.5
            - alpha = 1.0 returns model_a
            - alpha = 0.0 returns model_b
            - alpha = 0.5 returns average of both models
        output_dir: Optional directory to save the merged model.
                   If provided, the model will be saved to this directory.

    Returns:
        Merged OLMo model with interpolated weights

    Example:
        >>> merged = merge_olmo_models(
        ...     "allenai/OLMo-1B",
        ...     "allenai/OLMo-1B-instruct",
        ...     alpha=0.5,
        ...     output_dir="./merged_olmo"
        ... )
    """
    from hf_olmo import OLMoForCausalLM

    # Load models if they are paths/model names
    if isinstance(model_a, str):
        model_a = OLMoForCausalLM.from_pretrained(model_a)

    if isinstance(model_b, str):
        model_b = OLMoForCausalLM.from_pretrained(model_b)

    # Use the generic linear interpolation merge function
    merged_model = linear_interpolation_merge(model_a, model_b, alpha)

    # Save the merged model if output_dir is provided
    if output_dir is not None:
        print(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(output_dir)
        print(f"Model saved successfully to {output_dir}")
        return None  # Return None to prevent Fire from trying to use the model

    return merged_model


def merge_huggingface_models(
    model_a,
    model_b,
    alpha: float = 0.5,
    model_class=None,
    output_dir: Optional[str] = None,
):
    """
    Merge two HuggingFace models using linear interpolation.

    Returns: alpha * model_a + (1 - alpha) * model_b

    Args:
        model_a: First model (PreTrainedModel instance or path/model_name)
        model_b: Second model (PreTrainedModel instance or path/model_name)
        alpha: Interpolation weight in [0, 1], default 0.5
            - alpha = 1.0 returns model_a
            - alpha = 0.0 returns model_b
            - alpha = 0.5 returns average of both models
        model_class: Optional model class for loading (e.g., AutoModelForCausalLM).
                    If None, uses AutoModel.
        output_dir: Optional directory to save the merged model.
                   If provided, the model will be saved to this directory.

    Returns:
        Merged HuggingFace model with interpolated weights

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> merged = merge_huggingface_models(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     "meta-llama/Llama-2-7b-chat-hf",
        ...     alpha=0.5,
        ...     model_class=AutoModelForCausalLM,
        ...     output_dir="./merged_llama"
        ... )
    """
    # Load models if they are paths/model names
    if isinstance(model_a, str):
        if model_class is None:
            from transformers import AutoModel
            model_class = AutoModel
        model_a = model_class.from_pretrained(model_a)

    if isinstance(model_b, str):
        if model_class is None:
            from transformers import AutoModel
            model_class = AutoModel
        model_b = model_class.from_pretrained(model_b)

    # Use the generic linear interpolation merge function
    merged_model = linear_interpolation_merge(model_a, model_b, alpha)

    # Save the merged model if output_dir is provided
    if output_dir is not None:
        print(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(output_dir)
        print(f"Model saved successfully to {output_dir}")
        return None  # Return None to prevent Fire from trying to use the model

    return merged_model


def main():
    """
    CLI interface for model merging using Fire.

    Usage:
        # Merge OLMo models and save
        python -m src.merge merge_olmo_models --model_a=path/to/model_a --model_b=path/to/model_b --alpha=0.5 --output_dir=./merged_olmo

        # Merge HuggingFace models and save
        python -m src.merge merge_huggingface_models --model_a=path/to/model_a --model_b=path/to/model_b --alpha=0.5 --output_dir=./merged_hf
    """
    import fire
    fire.Fire(
        {
            "merge_olmo_models": merge_olmo_models,
            "merge_huggingface_models": merge_huggingface_models,
        }
    )


if __name__ == "__main__":
    main()
