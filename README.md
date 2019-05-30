# GPT2sQA

This repo includes an experiment of fine-tuning GPT-2 117M for Question Answering (QA). It also runs the model on Stanford Question Answering Dataset 2.0 (SQuAD). It uses Huggingface Inc.'s PyTorch implementation of GPT-2 and adapts from their fine-tuning of BERT for QA. 

SQuAD data can be downloaded from: https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset


To train and validate the model: 

```
python gpt2_squad.py --output_dir=output/ --train_file=data/train-v2.0.json --do_train --train_batch_size=32 --predict_file=data/dev-v2.0.json --do_predict

```

To evaluate: 

```

python evaluate-v2.0.py data/dev-v2.0.json output/predictions.json

```


Different fine-tuning experiments will be uploaded soon for GPT-2 345M on datasets that exclusively target commonsense reasoning in an attempt to bring insight to reasoning abilities of GPT-2. Such an insight could potentially improve our ability to improve Natural Language Understanding through language models in semi-supervised settings. 
