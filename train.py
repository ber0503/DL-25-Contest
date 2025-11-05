# Train with final hyperparameters with bigger max steps and full dataset

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp")
# Split into train and test sets
train_dataset = dataset['train']

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
    model_name = "unsloth/Meta-Llama-3.1-8B", # Competition-approved model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 1, # A small rank for lighter training
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 2, # A common practice is to set alpha = 2 * r
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 525,
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
    return { "text" : texts }

# Apply the formatting function to our training dataset
formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

def build_model(lr_scheduler_type,warmup_ratio,weight_decay,learning_rate):
  trainer = SFTTrainer(
      model = model,
      tokenizer = tokenizer,
      train_dataset = formatted_train_dataset,
      dataset_text_field = "text",
      max_seq_length = max_seq_length,
      args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,

        #######
        lr_scheduler_type = lr_scheduler_type,

        warmup_ratio = warmup_ratio,
        # warmup_steps = 5,

        # num_train_epochs = 2,
        max_steps = 15000,

        weight_decay = weight_decay,

        # [1e-4, 2e-4, 3e-4]
        learning_rate = learning_rate,
        ############

        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        seed = 525,
        output_dir = "outputs",
        report_to = "none",
      ),
  )
  return trainer

trainer=build_model("cosine_with_restarts",0.1,0.03,0.0005)
trainer.train()

import os
from datetime import datetime

# Define the path to save the model checkpoint in Google Drive
save_path = "llama3_8b_math_verifier_checkpoint_"+datetime.now().strftime("%Y%m%d%H%M%S")

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Save the model and tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model checkpoint and tokenizer saved to: {save_path}")