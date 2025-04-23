# 🧠 Toxic Comment Classification with BERT (Kaggle Challenge)

This project fine-tunes a BERT-based transformer model to classify Wikipedia comments into six types of toxicity. While BERT is a powerful contextual language model, it’s worth noting that traditional machine learning methods such as TF-IDF + Logistic Regression can perform just as well—or even better—for this specific task due to the short, keyword-rich nature of the comments.

## 🚀 Competition

- **Kaggle:** [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Goal:** Multi-label classification (comments may belong to multiple categories)

## 🧩 Labels

Each comment can have one or more of the following labels:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## 📦 Dependencies

Install requirements:

```
pip install transformers datasets scikit-learn pandas
```

GPU is highly recommended. This pipeline is optimized for use with Google Colab.


## 🧠 Model Overview

- Model: `bert-base-uncased` from Hugging Face Transformers
- Problem Type: `multi_label_classification`
- Loss Function: `BCEWithLogitsLoss` (handled automatically)
- Tokenizer: `BertTokenizer` (WordPiece-based)


## 📌 Note on Traditional ML

In this competition, many high-scoring solutions are based on TF-IDF + Logistic Regression or LightGBM. These models benefit from:
- The lexical nature of the problem (keywords like "idiot", "hate", "kill" are highly predictive)
- Short comment lengths that don’t require deep contextual understanding

BERT still offers strong performance, especially when combined with traditional methods in an ensemble.

## 🧠 Notes & Tips

- Uses `sigmoid` activation for multi-label classification
- Probabilities (not binary labels) are submitted for ROC AUC
- Ensemble with traditional ML models (TF-IDF + Logistic Regression) can boost scores
- You can switch to `roberta-base` or `deberta-v3-small` for even better results

## 🏁 Results

| Model         | Public LB Score |
|---------------|-----------------|
| BERT base     | ~0.9823         |


## 📜 License

MIT License. Feel free to reuse or extend!

