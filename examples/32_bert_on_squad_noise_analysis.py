from datetime import datetime
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)
from torch.utils.tensorboard import SummaryWriter
from evaluate import load
from datasets import load_dataset
from aihwkit.simulator.configs import InferenceRPUConfig, MappingParameter, WeightModifierType, WeightClipType, WeightNoiseType, BoundManagementType, NoiseManagementType, WeightClipParameter, WeightModifierParameter
from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from transformers.integrations import TensorBoardCallback
import numpy as np
import torch

# BERT model from Hugging Face model hub fine-tuned on SQuAD v1
MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Parse some arguments
PARSER = ArgumentParser("Analog BERT on SQuAD example")
PARSER.add_argument("-d", "--digital", help="Add to use digital inference", action="store_true")
PARSER.add_argument(
    "-i",
    "--ideal",
    help="Add to use ideal config instead of default noisy one",
    action="store_true",
)
PARSER.add_argument("-w", "--wandb", help="Add to use wandb", action="store_true")
PARSER.add_argument("-n", "--noise", help="Modifier noise", default=0.1, type=float)
PARSER.add_argument(
    "-r",
    "--run_name",
    help="Tensorboard run name",
    default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    type=str,
)
PARSER.add_argument("-t", "--train_hwa", help="Use Hardware-Aware training", action="store_true")
PARSER.add_argument(
    "-L", "--load", help="Use when loading from training checkpoint", action="store_true"
)
PARSER.add_argument(
    "-c",
    "--checkpoint",
    help="File name specifying where to load/save a checkpoint",
    default="./saved_chkpt.pth",
    type=str,
)
PARSER.add_argument(
    "-l", "--learning_rate", help="Learning rate for training", default=2e-4, type=float
)

ARGS = PARSER.parse_args()

if ARGS.wandb:
    import wandb
    SWEEP_CONFIG = {
        "method": "random",
        "name": "modifier noise sweep",
        "metric": {"goal": "maximize", "name": "exact_match"},
        "parameters": {"modifier_noise": {"values": [0, 0.05, 0.1, 0.2]}},
    }
    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project="bert-weight-noise-experiment")

# max length and stride specific to pretrained model
MAX_LENGTH = 320
DOC_STRIDE = 128

def create_ideal_rpu_config(tile_size=512):
    """Create RPU Config with ideal conditions"""
    rpu_config = InferenceRPUConfig(
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=False,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(is_perfect=True),
        noise_model=PCMLikeNoiseModel(prog_noise_scale=0.0, read_noise_scale=0.0, drift_scale=0.0),
        drift_compensation=None,
    )
    return rpu_config

def create_rpu_config(modifier_noise, tile_size=512, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""
    if ARGS.wandb:
        modifier_noise = wandb.config.modifier_noise

    rpu_config = InferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        modifier=WeightModifierParameter(
            rel_to_actual_wmax=True, type=WeightModifierType.ADD_NORMAL, std_dev=modifier_noise
        ),
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            w_noise_type=WeightNoiseType.PCM_READ,
            w_noise=0.0175,
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=GlobalDriftCompensation(),
    )
    return rpu_config

def create_model(rpu_config):
    """Return Question Answering model and whether or not it was loaded from a checkpoint"""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    if not ARGS.digital:
        model = convert_to_analog(model, rpu_config)
        model.remap_analog_weights()
        
        # Add logging for noise statistics per layer
        for name, layer in model.named_modules():
            if hasattr(layer, 'analog_tile'):
                print(f"Layer {name}: Noise stats: {layer.analog_tile.tile_config}")
                
    print(model)
    return model

def preprocess_train(dataset):
    """Preprocess the training dataset"""
    dataset["question"] = [q.lstrip() for q in dataset["question"]]
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_dataset.pop("offset_mapping")
    tokenized_dataset["start_positions"] = []
    tokenized_dataset["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_dataset["input_ids"][i]
        cls_index = input_ids.index(TOKENIZER.cls_token_id)
        sequence_ids = tokenized_dataset.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = dataset["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_dataset["start_positions"].append(cls_index)
            tokenized_dataset["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_dataset["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_dataset["end_positions"].append(token_end_index + 1)

    return tokenized_dataset

def preprocess_validation(dataset):
    """Preprocess the validation set"""
    dataset["question"] = [q.lstrip() for q in dataset["question"]]
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])
        tokenized_dataset["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
        ]

    return tokenized_dataset

def postprocess_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    """Postprocess raw predictions"""
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    predictions = OrderedDict()
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]

    return predictions

def create_datasets():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    squad = load_dataset("squad", name="plain_text")
    tokenized_data = squad.map(
        preprocess_train, batched=True, remove_columns=squad["train"].column_names
    )
    eval_data = squad["validation"].map(
        preprocess_validation, batched=True, remove_columns=squad["validation"].column_names
    )
    return squad, tokenized_data, eval_data

def create_optimizer(model):
    """Create the analog-aware optimizer"""
    optimizer = AnalogSGD(model.parameters(), lr=ARGS.learning_rate)
    optimizer.regroup_param_groups(model)
    return optimizer

def make_trainer(model, optimizer, tokenized_data):
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir="./",
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.001,
        no_cuda=False,
    )
    collator = DefaultDataCollator()
    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=TOKENIZER,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)],
    )
    return trainer, writer

def do_inference(model, trainer, squad, eval_data, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime.
    SQuAD exact match and f1 metrics are captured in Tensorboard
    """
    def predict():
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(
            squad["validation"], eval_data, raw_predictions.predictions
        )
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)
        return out_metric["f1"], out_metric["exact_match"]

    def write_metrics(f1, exact_match, t_inference):
        writer.add_scalar("val/f1", f1, t_inference)
        writer.add_scalar("val/exact_match", exact_match, t_inference)
        if ARGS.wandb:
            wandb.log({"t_inference": t_inference, "f1": f1, "exact_match": exact_match})
        print(f"Exact match: {exact_match: .2f}\t" f"F1: {f1: .2f}\t" f"Drift: {t_inference: .2e}")

    model.eval()
    metric = load("squad")
    ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad["validation"]]
    t_inference_list = np.logspace(0, np.log10(float(max_inference_time)), n_times).tolist()
    f1, exact_match = predict()
    write_metrics(f1, exact_match, 0.0)
    
    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        f1, exact_match = predict()
        write_metrics(f1, exact_match, t_inference)
        
        # Log layer-wise noise impact
        for name, layer in model.named_modules():
            if hasattr(layer, 'analog_tile'):
                noise_level = layer.analog_tile.tile_config.modifier.std_dev
                writer.add_scalar(f"noise/{name}", noise_level, t_inference)

def main():
    if ARGS.wandb:
        wandb.init()
    if ARGS.ideal:
        rpu_config = create_ideal_rpu_config()
    else:
        rpu_config = create_rpu_config(modifier_noise=ARGS.noise)
    model = create_model(rpu_config)
    squad, tokenized_data, eval_data = create_datasets()
    optimizer = create_optimizer(model)
    trainer, writer = make_trainer(model, optimizer, tokenized_data)
    if ARGS.load:
        print(f"Load model from '{ARGS.checkpoint}'.")
        model.load_state_dict(torch.load(ARGS.checkpoint))
    if ARGS.train_hwa and not ARGS.digital and not ARGS.load:
        trainer.train()
        torch.save(model.state_dict(), ARGS.checkpoint)
    do_inference(model, trainer, squad, eval_data, writer)

if ARGS.wandb:
    wandb.agent(SWEEP_ID, function=main, count=4)
else:
    main()
