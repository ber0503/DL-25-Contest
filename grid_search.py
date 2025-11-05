# Experiment 1 and Experiment 2


from datasets import load_dataset

# Load the full training dataset
full_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")

# Shuffle the dataset for randomness and create our smaller splits
shuffled_dataset = full_dataset.shuffle(seed=42)
train_dataset = shuffled_dataset.select(range(5000))  # Use the first 5,000 for training
validation_dataset = shuffled_dataset.select(range(5000, 5200))  # Use the next 100 for validation

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

######
max_seq_length = 2048  # Choose any sequence length
dtype = None  # This will auto-detect the best data type for your GPU
load_in_4bit = True  # Use 4-bit quantization to save memory

# Load the model and tokenizer from Hugging Face
# Note: We use the base model, not a 4-bit pre-quantized one,
# to ensure we start from the official weights.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  # Competition-approved model
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=1,  # A small rank for lighter training
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=2,  # A common practice is to set alpha = 2 * r
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# The instructional prompt template for training
training_prompt = """You are a great mathematician and you are tasked with finding if a solution to a given maths question is correct or not. Your response should be 'True' if the solution is correct, otherwise 'False'. Below is the Question and Solution.
Question:
{}
Solution:
{}
Output:
{}"""

# We must add an End Of Sequence (EOS) token to tell the model when a completion is finished.
EOS_TOKEN = tokenizer.eos_token


# This function formats our data samples into the prompt template.
def formatting_prompts_func(examples):
    questions = examples["question"]
    solutions = examples["solution"]
    outputs = examples["is_correct"]
    texts = []
    for question, solution, output in zip(questions, solutions, outputs):
        # Format the prompt and add the EOS token
        text = training_prompt.format(question, str(solution), str(output)) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# Apply the formatting function to our training dataset
formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

from tqdm import tqdm

# Create the prompt template for inference (no answer included)
inference_prompt = """You are a great mathematician and you are tasked with finding if a solution to a given maths question is correct or not. Your response should be 'True' if the solution is correct, otherwise 'False'. Below is the Question and Solution.
Question:
{}
Solution:
{}
Output:
"""


def get_prediction(question, solution):
    # Format the prompt with the validation data
    inputs = tokenizer(
        [
            inference_prompt.format(question, str(solution))
        ], return_tensors="pt").to("cuda")

    # Generate the model's response
    outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
    response = tokenizer.batch_decode(outputs)
    return response


def format_prediction(question, solution, y_hat, y):
    result = (
        f"#### QUESTION ####\n{question}\n\n"
        f"#### SOLUTION ####\n{solution}\n\n"
        f"#### MODEL'S PREDICTION ####\n{y_hat}\n\n"
        f"#### CORRECT ANSWER ####\n{y}\n\n"
    )
    return result


# A simple function to parse 'True' or 'False' from the model's raw output
def parse_output(response_text):
    # Find the text after "Output:"
    output_part = response_text.split("Output:\n")[-1]
    # Check if "True" is in that part, case-insensitively
    if 'true' in output_part.lower():
        return True
    return False


def eval_accuracy():
    # Prepare the model for faster inference
    FastLanguageModel.for_inference(model)

    correct = 0
    total = len(validation_dataset)
    log = []
    fail_idx = []

    for i in tqdm(range(total)):
        # Select a sample from the validation set
        example = validation_dataset[i]  # You can change the index (e.g., to 1, 2, 50)
        question = example["question"]
        solution = example["solution"]
        response = get_prediction(question, solution)
        model_output = parse_output(response)
        if model_output == example["is_correct"]:
            correct += 1
        else:
            fail_idx.append(i)
        log.append(format_prediction(question, solution, model_output, example["is_correct"]))

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    return accuracy, log, fail_idx


def build_model(lr_scheduler_type, warmup_ratio, weight_decay, learning_rate):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,

            #######
            lr_scheduler_type=lr_scheduler_type,

            # [0.02, 0.03, 0.05, 0.10]
            warmup_ratio=warmup_ratio,
            # warmup_steps = 5,

            # num_train_epochs = 2,
            max_steps=60,

            # [0.0, 0.01, 0.03, 0.05]
            weight_decay=weight_decay,

            # [1e-4, 2e-4, 3e-4]
            learning_rate=learning_rate,
            ############

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            seed=42,
            output_dir="outputs",
            report_to="none",
        ),
    )
    return trainer


wr_grid = [0.02, 0.03, 0.05, 0.10]
wd_grid = [0.0, 0.01, 0.03, 0.05]
lr_grid = [1e-4, 2e-4, 3e-4, 5e-4, 7e-4]
ls_grid = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "constant_with_warmup"
]

import json

# results = []
# for lr in lr_grid:
#     for wd in wd_grid:
#         trainer=build_model("cosine_with_restarts",0.03,wd,lr)
#         trainer.train()
#         acc,log,fail_idx=eval_accuracy(trainer)
#         results.append({"lr_scheduler_type":"cosine_with_restarts","warmup_ratio": 0.03, "weight_decay": wd, "learning_rate": lr, "accuracy": acc})

# with open("results-lr-wd.txt", "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

# print("✅ Results saved to results.txt")

bs_weight_decay = 0.03
bs_learning_rate = 0.0005

results = []
for ls in ls_grid:
    for wr in wr_grid:
        trainer = build_model(ls, wr, bs_weight_decay, bs_learning_rate)
        trainer.train()
        acc, log, fail_idx = eval_accuracy()
        results.append({"lr_scheduler_type": ls, "warmup_ratio": wr, "weight_decay": bs_weight_decay,
                        "learning_rate": bs_learning_rate, "accuracy": acc})

with open("results-ls-wr.txt", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Results saved to results.txt")
