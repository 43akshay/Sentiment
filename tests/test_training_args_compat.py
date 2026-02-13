from train import build_training_args_kwargs


def test_build_training_args_contains_common_fields():
    kwargs = build_training_args_kwargs(epochs=2, batch_size=4, use_eval=True)

    assert kwargs["output_dir"] == "./outputs"
    assert kwargs["num_train_epochs"] == 2
    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["per_device_eval_batch_size"] == 4


def test_build_training_args_uses_available_eval_key():
    kwargs = build_training_args_kwargs(epochs=1, batch_size=2, use_eval=False)

    # depending on transformers version either key can be present
    assert ("evaluation_strategy" in kwargs) or ("eval_strategy" in kwargs)
    key = "evaluation_strategy" if "evaluation_strategy" in kwargs else "eval_strategy"
    assert kwargs[key] == "no"
