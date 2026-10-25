"""Train baseline, AraBERT or AraPoemBERT. Test set is read only with the explicit test command."""
import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from data_utils import clean, cap_by_era, write_training_info, verify_training_info, guard_run
from classification_config import MODELS, model_folder

ROOT = Path(os.environ.get('SHATR_ROOT', '.'))
DATA, OUT = ROOT/'data', ROOT/'outputs'

def read(split):
    return pd.read_json(DATA/f'cls_{split}.jsonl', lines=True)

def report(y, pred, labels, filename):
    ids = list(range(len(labels)))
    result = classification_report(y, pred, labels=ids, target_names=list(labels),
                                   output_dict=True, zero_division=0)
    result['confusion_matrix'] = confusion_matrix(y, pred, labels=ids).tolist()
    filename.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(classification_report(y, pred, labels=ids, target_names=list(labels), zero_division=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['baseline', 'train', 'test'])
    parser.add_argument('--smoke', action='store_true', help='Pipeline check, not a reportable final model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--model', choices=list(MODELS), default='arabert')
    args = parser.parse_args()
    model_id = MODELS[args.model]
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads((DATA/'labels.json').read_text(encoding='utf-8'))
    if args.mode == 'baseline':
        train, val = cap_by_era(read('train'), 2000), cap_by_era(read('validation'), 500)
        if args.smoke:
            train, val = cap_by_era(train, 100), cap_by_era(val, 40)
        clf = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=60000),
                            LogisticRegression(max_iter=1000, class_weight='balanced'))
        clf.fit(train.text.map(clean), train.label)
        joblib.dump(clf, OUT/'baseline.joblib')
        write_training_info(OUT/'baseline_info', DATA, 'TFIDF + LogisticRegression')
        report(val.label, clf.predict(val.text.map(clean)), labels, OUT/'baseline_validation.json')
        return
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                              TrainingArguments, DataCollatorWithPadding, set_seed)
    set_seed(42)
    target = model_folder(ROOT, args.model, args.smoke)
    if args.mode == 'test':
        # Final evaluation after freezing decisions; never choose settings from test scores.
        verify_training_info(target, DATA)
        tok = AutoTokenizer.from_pretrained(target)
        model = AutoModelForSequenceClassification.from_pretrained(target)
        test = read('test')
        ds = Dataset.from_dict({'text':test.text.map(clean).tolist(), 'label':test.label.tolist()})
        ds = ds.map(lambda b: tok(b['text'], truncation=True, max_length=192), batched=True,
                    remove_columns=['text'])
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(OUT/'test_tmp'),
                          report_to='none', per_device_eval_batch_size=8),
                          data_collator=DataCollatorWithPadding(tok))
        pred = trainer.predict(ds).predictions.argmax(-1)
        report(test.label, pred, labels, OUT/('basma_test.json' if args.model == 'arabert' else 'arapoembert_test.json'))
        if (OUT/'baseline.joblib').exists():
            verify_training_info(OUT/'baseline_info', DATA)
            clf = joblib.load(OUT/'baseline.joblib')
            report(test.label, clf.predict(test.text.map(clean)), labels, OUT/'baseline_test.json')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Choose a GPU runtime for training; baseline works on CPU.')
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels), label2id=labels, id2label={i:e for e,i in labels.items()})
    def encode(split, cap):
        frame = cap_by_era(read(split), cap)
        ds = Dataset.from_dict({'text':frame.text.map(clean).tolist(), 'label':frame.label.tolist()})
        return ds.map(lambda b: tok(b['text'], truncation=True, max_length=192),
                      batched=True, remove_columns=['text'])
    train = encode('train', 100 if args.smoke else 2000)
    val = encode('validation', 40 if args.smoke else 500)
    def metrics(p):
        yhat = p.predictions.argmax(-1)
        return dict(macro_f1=f1_score(p.label_ids, yhat, average='macro'),
                    accuracy=accuracy_score(p.label_ids,yhat))
    cfg = TrainingArguments(output_dir=str(target/'checkpoints'), learning_rate=2e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=1 if args.smoke else args.epochs,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        fp16=True, report_to='none', seed=42)
    guard_run(target, DATA, model_id, cfg.to_dict())
    trainer = Trainer(model=model,args=cfg,train_dataset=train,eval_dataset=val,
        processing_class=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics)
    # نستأنف فقط من checkpoint تابع لنفس التجربة.
    from transformers.trainer_utils import get_last_checkpoint
    checkpoint = get_last_checkpoint(str(target/'checkpoints')) if (target/'checkpoints').exists() else None
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(target))
    tok.save_pretrained(target)
    write_training_info(target, DATA, model_id)
    info = json.loads((target/'training_info.json').read_text(encoding='utf-8'))
    info.update(training_arguments=cfg.to_dict(), smoke=args.smoke, train_examples=len(train), validation_examples=len(val))
    (target/'training_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    (target/'validation_metrics.json').write_text(json.dumps(trainer.evaluate(),indent=2), encoding='utf-8')
    # حفظ تقرير كل عصر ومصفوفة الالتباس على نفس عينة التحقق.
    validation = cap_by_era(read('validation'), 40 if args.smoke else 500)
    prediction = trainer.predict(val).predictions.argmax(-1)
    report(validation.label, prediction, labels, target/'classification_validation.json')
    validation['predicted_era'] = [model.config.id2label[int(i)] for i in prediction]
    validation.to_csv(target/'validation_predictions.csv', index=False, encoding='utf-8-sig')
    validation['correct'] = validation['era'] == validation['predicted_era']
    poet_results = validation.groupby(['era','poet']).agg(
        examples=('correct','size'), accuracy=('correct','mean'))
    poet_results.to_csv(target/'per_poet_validation.csv', encoding='utf-8-sig')
    print('Saved to',target)

if __name__ == '__main__':
    main()
