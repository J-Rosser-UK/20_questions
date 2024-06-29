# Instructions to Download the ReRanker Fully Offline

1. Run `save_model.py` to pull the reranker from huggingface
2. Visit this (https://huggingface.co/jinaai/jina-bert-implementation/tree/main) repo and download `configuration_bert.py` and `modeling_bert.py` and place them inside the `jina-reranker-v1-tiny-en/` repo
3. Update the `config.json` as follows:
From:
```
"auto_map": {
    "AutoConfig": "jinaai/jina-bert-implementation--configuration_bert.JinaBertConfig",
    "AutoModel": "jinaai/jina-bert-implementation--modeling_bert.JinaBertModel",
    "AutoModelForMaskedLM": "jinaai/jina-bert-implementation--modeling_bert.JinaBertForMaskedLM",
    "AutoModelForQuestionAnswering": "jinaai/jina-bert-implementation--modeling_bert.JinaBertForQuestionAnswering",
    "AutoModelForSequenceClassification": "jinaai/jina-bert-implementation--modeling_bert.JinaBertForSequenceClassification",
    "AutoModelForTokenClassification": "jinaai/jina-bert-implementation--modeling_bert.JinaBertForTokenClassification"
},
```
To:
```
"auto_map": {
    "AutoConfig": "configuration_bert.JinaBertConfig",
    "AutoModel": "modeling_bert.JinaBertModel",
    "AutoModelForMaskedLM": "modeling_bert.JinaBertForMaskedLM",
    "AutoModelForQuestionAnswering": "modeling_bert.JinaBertForQuestionAnswering",
    "AutoModelForSequenceClassification": "modeling_bert.JinaBertForSequenceClassification",
    "AutoModelForTokenClassification": "modeling_bert.JinaBertForTokenClassification"
},
```
