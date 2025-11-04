from unsloth import FastLanguageModel


save_path = "llama3_8b_math_verifier_checkpoint_20251031055355"
# save_path = "llama3_8b_math_verifier_checkpoint20251101120030"

max_seq_length = 2048  # Choose any sequence length
dtype = None  # This will auto-detect the best data type for your GPU
load_in_4bit = True  # Use 4-bit quantization to save memory
# Load the model and tokenizer from the saved path
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = save_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Prepare the loaded model for faster inference
FastLanguageModel.for_inference(model)

print(f"Model and tokenizer loaded from: {save_path}")

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Load the official test set
test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
predictions = []

# Create the prompt template for inference (no answer included)
inference_prompt = """You are a great mathematician and you are tasked with finding if a solution to a given maths question is correct or not. Your response should be 'True' if the solution is correct, otherwise 'False'. Below is the Question and Solution.
Question:
{}
Solution:
{}
Output:
"""

# A simple function to parse 'True' or 'False' from the model's raw output
def parse_output(response_text):
    # Find the text after "Output:"
    output_part = response_text.split("Output:\n")[-1]
    # Check if "True" is in that part, case-insensitively
    if 'true' in output_part.lower():
        return True
    return False


# Loop through the test dataset and generate a prediction for each example
for example in tqdm(test_dataset):
    question = example["question"]
    solution = example["solution"]

    # Format the prompt
    prompt = inference_prompt.format(question, str(solution))
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate the prediction
    outputs = model.generate(**inputs, max_new_tokens=8, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)[0]

    # Parse the prediction and add it to our list
    prediction = parse_output(response_text)
    predictions.append(prediction)

# Create the submission DataFrame
submission = pd.DataFrame({
    'ID': range(len(predictions)),
    'is_correct': predictions
})

# Save the DataFrame to a CSV file
submission.to_csv('submission4.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("You can now download this file and submit it to the Kaggle competition.")