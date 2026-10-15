# Financial News Sentiment Prediction using Deep Learning & BERT

Three-class finance sentiment classification:
- `0` = Bearish
- `1` = Bullish
- `2` = Neutral

## Supplied data
- Training: 9,543 rows
- Validation: 2,388 rows
- Columns: `text`, `label`
- Train/validation text overlap: 0

The provided CSV split is treated as the source of truth. No new random validation split is created.

## Project plan
1. EDA and data validation
2. Light tweet preprocessing while preserving tickers/financial terms
3. Embedding + Simple RNN
4. Embedding + LSTM
5. Embedding + GRU
6. Accuracy, macro F1, class-wise precision/recall, confusion matrices
7. FinBERT/BERT fine-tuning and comparison
8. Error analysis
9. Streamlit inference dashboard
10. 2-page comparison report

## Run
```bash
pip install -r requirements.txt
jupyter notebook financial_sentiment_project.ipynb
```

## Notes
The transformer stage needs a compatible pretrained model checkpoint available locally or downloadable from Hugging Face.
