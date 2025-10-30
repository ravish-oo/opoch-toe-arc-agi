# ARC AGI 2025 Competition Data

Downloaded from: https://www.kaggle.com/competitions/arc-prize-2025

## Files

### Training Data (Use for model development)
- **arc-agi_training_challenges.json** (3.8 MB)
  - Training tasks with demo input/output pairs
  - Use these to develop and train your algorithm

- **arc-agi_training_solutions.json** (643 KB)
  - Ground truth test outputs for training tasks
  - Use to verify your predictions during development

### Evaluation Data (Use for local validation)
- **arc-agi_evaluation_challenges.json** (962 KB)
  - Validation tasks with demo input/output pairs
  - Use to validate your approach before submission

- **arc-agi_evaluation_solutions.json** (219 KB)
  - Ground truth test outputs for evaluation tasks
  - Use to check performance locally

### Test Data (For submission reference)
- **arc-agi_test_challenges.json** (991 KB)
  - **PLACEHOLDER** - Contains evaluation tasks for local testing
  - **IMPORTANT**: When your notebook runs on Kaggle during submission, this file is **replaced** with 240 hidden test tasks
  - Your code should read from this file path: `/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json`

### Submission Template
- **sample_submission.json** (19 KB)
  - Example submission format
  - Shows correct JSON structure with attempt_1 and attempt_2

## File Structure

Each challenges file contains tasks in this format:
```json
{
  "task_id": {
    "train": [
      {"input": [[grid]], "output": [[grid]]}
    ],
    "test": [
      {"input": [[grid]]}
    ]
  }
}
```

Each solutions file:
```json
{
  "task_id": [
    [[grid]],  // output for test input 1
    [[grid]]   // output for test input 2 (if exists)
  ]
}
```
