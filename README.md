
# YouTube Toxic Comment Detection using BERT

This project demonstrates a complete pipeline to fine‑tune a BERT‑based transformer model for multi‑label classification of toxic comments from YouTube videos. The dataset focuses on content related to the 2014 Ferguson unrest and includes 13 hierarchical toxicity labels such as abusive, racist, sexist, and radical comments.

The model is trained using the Hugging Face Transformers library on a small, hand‑labeled dataset of 1000 English comments. The goal is to detect various types of toxicity simultaneously in a multi‑label classification setup.

## Dataset

The dataset contains 1000 YouTube comments with the following structure:

- `CommentId`: Unique identifier for the comment  
- `VideoId`: Associated YouTube video ID  
- `Text`: Comment text  
- `IsToxic`, `IsAbusive`, `IsThreat`, `IsProvocative`, `IsObscene`, `IsHatespeech`, `IsRacist`, `IsNationalist`, `IsSexist`, `IsHomophobic`, `IsReligiousHate`, `IsRadicalism`: Boolean labels (multi‑label structure)

Dataset Source: `/kaggle/input/youtube-toxicity-data/youtoxic_english_1000.csv`

## Model Overview

- Model: `bert-base-uncased`  
- Task: Multi‑label text classification  
- Tokenizer: BERT tokenizer from Hugging Face  
- Loss Function: Binary Cross‑Entropy (BCEWithLogitsLoss)  
- Activation: Sigmoid on logits to produce independent probabilities per class

## Pipeline Steps

1. **Data Preprocessing**  
   - Load dataset and convert label columns from boolean to integer format  
   - Split into training and test sets (80/20)

2. **Tokenization**  
   - Use Hugging Face tokenizer with truncation  
   - Apply to both train and test datasets using `datasets` library

3. **Model Configuration**  
   - Load `bert-base-uncased` with output head for 13 labels  
   - Use problem type `multi_label_classification`

4. **Training**  
   - Fine‑tune using Hugging Face `Trainer`  
   - Batch size: 8  
   - Epochs: 2  
   - Optimizer: AdamW  
   - Learning rate: 2e-5

5. **Evaluation**  
   - Macro F1‑score as primary metric using `evaluate` library  
   - Classification report with precision, recall, and F1 per class

6. **Inference**  
   - Sample 5 test comments to display predicted labels

## Dependencies

- Python 3.10+  
- Transformers ≥ 4.41.0  
- Datasets ≥ 2.19.0  
- PyTorch  
- Scikit‑learn  
- Evaluate  
- Accelerate

Install requirements with:

```bash
pip install transformers datasets evaluate scikit-learn accelerate
```

## Results

The model successfully learns to detect multiple toxic traits in YouTube comments using a lightweight BERT variant and a small dataset. Evaluation is performed using macro‑averaged F1 score, and a classification report highlights performance across all labels. Sample predictions demonstrate that the model can recognize various forms of toxicity in real comments.

## Sample Output

```text
Text: They should all be shot for what they did that day.
Predicted labels: ['IsToxic', 'IsAbusive', 'IsThreat']

Text: The Arab dude is absolutely right, he should have led it.
Predicted labels: None

Text: Black lives matter is a terrorist movement.
Predicted labels: ['IsToxic', 'IsAbusive', 'IsRacist']
```

## Limitations and Future Work

- The dataset is limited to 1000 comments, which restricts generalizability  
- Labels are imbalanced; some classes have very few positive samples  
- Future improvements could include class weighting, data augmentation, or experimenting with larger multilingual models like XLM‑R

## Author

Stanley Moses  
University of Hertfordshire  
Research Methods in Data Science (7PAM2015)  
Assignment 2 – Large Language Models  
July 2025
