# Project Structure

- **llama3_8b_math_verifier_checkpoint20251101120030**  
  Checkpoint of seed **41**, providing the final accuracy obtained.

- **llama3_8b_math_verifier_checkpoint_20251031055355**  
  Checkpoint of seed **441**.

- **Contest_DL_2025.ipynb**  
  Main notebook used for experimentation and analysis.

- **grid_search.py**  
  Grid search script written by Victoria.

- **make_csv.py**  
  Loads the trained model and generates CSV outputs.

- **results-lr-wd.txt**  
  Results of Victoria’s grid search over **learning rate (lr)** and **weight decay (wd)**,  
  under fixed scheduler and warmup settings.

- **results-ls-wr.txt**  
  Results of Victoria’s grid search over **lr scheduler (ls)** and **warmup ratio (wr)**,  
  under fixed learning rate and weight decay.

- **train.py**  
  Training script using the selected optimal hyperparameters.

- **voting.py**  
  Script exploring different prediction voting strategies.
