# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 24: Example using convert_to_analog to run BERT transformer on SQuAD task
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD

from transformers import (AutoTokenizer, BertForQuestionAnswering,
                        Trainer, TrainingArguments, DefaultDataCollator)

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
model = convert_to_analog(model, SingleRPUConfig())

squad = load_dataset("squad")

print(squad["train"][0])

# From huggingface tutorial on question answering
# Some examples in the dataset may have contexts that exceed the maximum input length
#   We can truncate the context using truncation="only_second"
#
def preprocess(dataset):
    questions = [ q.strip() for q in dataset["question"] ]
    inputs = tokenizer(
        questions,
        dataset["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mappings = inputs.pop("offset_mapping")
    answers = dataset["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mappings):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)

        # offset[context_start][0] gets the character position corresponding to the start
        # of the token at the beginning of the context
        # offset[context_start][1] gets the character position corresponding to the end
        # of the token at the end of the context

        # We then check that the answer is contained within these corresponding positions
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            # The answer isn't contained within the context, so add (0, 0)
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions

            # Shift a pointer from the beginning of the context over to the beginning of
            # the first token in the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # Shift a pointer from the end of the context over to the end of
            # the last token in the answer
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Preprocessing changes number of samples, so we need to remove some columns so
# the data updates properly
squad = squad.map(preprocess, batched=True, remove_columns=squad["train"].column_names)

training_args = TrainingArguments(
    output_dir='./',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

collator = DefaultDataCollator()

optimizer = AnalogSGD(model.parameters(), lr=2e-4)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=squad["train"],
    eval_dataset=squad["validation"],
    tokenizer=tokenizer,
    optimizers=(optimizer, None)
)

trainer.train()

''' Next steps
        - Create analog optimizer
        - Start using tensor board to track info + debugging
        - Use a custom RPU configuration for Inference on PCM device

    Drift Experiment
        - Take digital model and fine-tune it on a task + use it for inference
        - Perform hardware-aware training with same model
        - Show that hwa training creates a more robust model
            since drift is accomodated for over time
'''