# harmful-output-detection

[202601-AIE3001-001] Machine Learning Term Project

Detecting and Reducing Harmful Outputs in a Small Language Model  
소형 언어 모델에서의 유해 발화 탐지 및 완화 실험

[colab](https://colab.research.google.com/gist/Moonhannah05/48b611347ab02a6875b7b494a91961ca/detecting_and_reducing_harmful_outputs_in_a_small_language_model.ipynb)

## 1. Project Overview

This project focuses on detecting and reducing harmful outputs in a small language model.

The project consists of two main parts.

1. Harmful text detection using the Jigsaw Toxic Comment Classification Challenge dataset
2. A small-scale safety unlearning experiment using DistilGPT-2

For harmful text detection, two models were compared.

| Model | Description |
|---|---|
| TF-IDF + Logistic Regression | Simple baseline model |
| DistilBERT | Transformer-based main model |

For safety unlearning, Gradient Ascent was applied to harmful samples while retention loss was used on safe samples to reduce catastrophic forgetting.

## 2. Dataset

This project uses the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

The original dataset contains six toxicity labels.

| Label | Meaning |
|---|---|
| toxic | Toxic comment |
| severe_toxic | Severely toxic comment |
| obscene | Obscene or vulgar expression |
| threat | Threatening expression |
| insult | Insulting expression |
| identity_hate | Hate toward identity groups |

In this project, the six labels were converted into a binary label.

```python
harmful = max(toxic, severe_toxic, obscene, threat, insult, identity_hate)
````

If at least one of the six labels is 1, the comment is labeled as harmful.
If all six labels are 0, the comment is labeled as safe.

The test set was created by merging `test.csv` and `test_labels.csv`.
Samples with label value `-1` were removed because they were not used for evaluation.

## 3. Repository Structure

```text
harmful-output-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
└── Detecting_and_Reducing_Harmful_Outputs_in_a_Small_Language_Model.ipynb
```

The main experiment code is included in the Jupyter notebook.

## 4. Environment

The experiments were conducted in Google Colab.

To install the required packages, run:

```bash
pip install -r requirements.txt
```

If running directly in Google Colab, the notebook can be opened using the Colab badge above.


## 5. How to Reproduce

This notebook was designed to run in Google Colab.

### Step 1. Open the notebook in Google Colab

Click the Colab badge at the top of this README, or open the notebook manually.

```text
Detecting_and_Reducing_Harmful_Outputs_in_a_Small_Language_Model.ipynb
````

### Step 2. Prepare Kaggle API token

To download the Jigsaw dataset automatically, a Kaggle API token is required.

1. Go to Kaggle.
2. Open Account Settings.
3. Click Create New Token.
4. Download the `kaggle.json` file.

### Step 3. Upload `kaggle.json` in Colab

When running the second cell of the notebook, upload the downloaded `kaggle.json` file.

The notebook then uses the Kaggle API to download and unzip the Jigsaw Toxic Comment Classification Challenge dataset automatically.


### Step 4. Run the notebook from top to bottom

After uploading `kaggle.json`, run all cells in order.


## 6. Main Results

### Harmful Text Detection

| Model                        | Threshold | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ---------------------------- | --------: | -------: | --------: | -----: | -----: | ------: |
| TF-IDF + Logistic Regression |       0.7 |   0.9153 |    0.5442 | 0.8091 | 0.6507 |  0.9555 |
| DistilBERT                   |       0.7 |   0.9163 |    0.5423 | 0.9117 | 0.6801 |  0.9723 |

DistilBERT achieved a higher recall than the TF-IDF + Logistic Regression baseline.
This means that DistilBERT reduced false negatives, which is important in harmful text detection because missing harmful text can be more risky than over-detecting safe text.

### Safety Unlearning

The adjusted unlearning setup reduced the toxic generation rate while keeping perplexity close to the original DistilGPT-2 model.

| Model                 | Toxic Generation Rate | Perplexity |
| --------------------- | --------------------: | ---------: |
| Original DistilGPT-2  |                0.3333 |      10.86 |
| Unlearned DistilGPT-2 |                0.0667 |      11.15 |

Although the quantitative results improved, qualitative analysis showed that the unlearned model did not completely erase harmful concepts. Some prompts still produced unstable or undesirable generations.

## 7. Limitations

This project has several limitations.

1. The Jigsaw dataset contains ambiguous and noisy labels.
2. Some comments labeled as safe contain strong profanity.
3. Some comments labeled as harmful are difficult to interpret without broader context.
4. DistilGPT-2 is a small language model, so the results may not generalize to larger LLMs.
5. The number of prompts used for unlearning evaluation is limited.
6. The unlearning experiment does not prove complete concept erasure.

Therefore, the results should be interpreted as a small-scale exploratory experiment rather than a complete solution for language model safety.

## 8. Notes on Data

The original Kaggle dataset files are not included in this repository.

This project uses the Kaggle API to download the dataset inside Google Colab.  
To reproduce the experiment, users need to prepare their own API token from Kaggle.

## 9. AI Assistance Disclosure

Generative AI tools such as ChatGPT and Claude were used to assist with code implementation, debugging, and report drafting.
The project design, experiment execution, result interpretation, and final decisions were conducted by the author.
