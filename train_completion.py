"""Qwen LoRA completion training; prompt and padding positions are excluded from loss."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from data_utils import clean, cap_by_era, write_training_info, verify_training_info, guard_run

MODEL = 'Qwen/Qwen3-0.6B'
ROOT = Path(os.environ.get('SHATR_ROOT','.'))
DATA, OUT = ROOT/'data', ROOT/'outputs'

def messages(first):
    return [{'role':'system','content':'أكمل الشطر الأول بشطر ثان واحد فقط، دون شرح أو عنوان.'},
            {'role':'user','content':clean(first)}]

def prompt_ids(tok, first):
    return tok.apply_chat_template(messages(first), tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)

def encode_pair(tok, first, second, max_length=256):
    prompt = prompt_ids(tok,first)
    answer = tok.encode(clean(second), add_special_tokens=False) + [tok.eos_token_id]
    if len(prompt)>192 or len(prompt)+len(answer) > max_length or not clean(second):
        return None  # No truncation that would silently delete the answer.
    return {'input_ids':prompt+answer,'attention_mask':[1]*(len(prompt)+len(answer)),
            'labels':[-100]*len(prompt)+answer}

class CompletionCollator:
    def __init__(self, pad_id):
        self.pad_id=pad_id
    def __call__(self, rows):
        length=max(len(r['input_ids']) for r in rows)
        return {field:torch.tensor([r[field]+[pad]*(length-len(r[field])) for r in rows])
                for field,pad in [('input_ids',self.pad_id),('attention_mask',0),('labels',-100)]}

def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side='right'
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
    return tok,model

@torch.inference_mode()
def generate(model,tok,first,seed=42):
    set_seed(seed)
    ids=prompt_ids(tok,first)
    if len(ids)>192:
        raise ValueError('الشطر طويل جدًا؛ أدخلي شطرًا قصيرًا واحدًا.')
    inputs=torch.tensor([ids],device=model.device)
    model.eval()
    output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
        max_new_tokens=64,do_sample=True,temperature=.8,top_p=.9,
        repetition_penalty=1.1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    text=tok.decode(output[0,len(ids):],skip_special_tokens=True).strip()
    # نحافظ على النص الخام في التقييم؛ لا نخفي تكرارًا أو سطورًا زائدة.
    return text

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['before','train','compare','test'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int,default=2)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    tok,base=load_base()
    target=OUT/('shatr_smoke' if args.smoke else 'shatr_adapter')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode!='train':
        split='test' if args.mode=='test' else 'validation'
        examples=pd.read_json(DATA/f'gen_{split}.jsonl',lines=True)
        valid=[encode_pair(tok,r['first'],r['second']) is not None for r in examples.to_dict('records')]
        examples=cap_by_era(examples.loc[valid], 8).sample(frac=1,random_state=42).head(32)
        if examples.empty:
            raise ValueError('No evaluation pairs fit the token budget.')
        base.to(device)
        results=[]
        for i,row in enumerate(examples.to_dict('records')):
            results.append({'first':row['first'],'reference':row['second'],
                            'before':generate(base,tok,row['first'],42+i)})
        if args.mode!='before':
            verify_training_info(target, DATA)
            adapted=PeftModel.from_pretrained(base,target)
            adapted.to(device)
            for i,row in enumerate(results):
                row['after']=generate(adapted,tok,row['first'],42+i)
        dest=OUT/f'shatr_{args.mode}{"_smoke" if args.smoke else ""}.csv'
        pd.DataFrame(results).to_csv(dest,index=False,encoding='utf-8-sig')
        if args.mode != 'before':
            # جدول تقييم أعمى: لا يعرف المقيم أي نسخة قبل التدريب أو بعده.
            import random
            randomizer = random.Random(42)
            blind, answer_key = [], []
            for number, row in enumerate(results, 1):
                order = ['before','after']
                randomizer.shuffle(order)
                for name, version in zip(['A','B'], order):
                    blind.append(dict(id=number, first=row['first'], version=name,
                        completion=row[version], language='', meaning='', poetry='', no_repetition=''))
                    answer_key.append(dict(id=number, version=name, actual_version=version))
            pd.DataFrame(blind).to_csv(OUT/f'human_eval_{split}{"_smoke" if args.smoke else ""}.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(answer_key).to_csv(OUT/f'human_eval_key_{split}{"_smoke" if args.smoke else ""}.csv', index=False, encoding='utf-8-sig')
        print('Saved',dest)
        print(pd.DataFrame(results).head(3).to_string(index=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Select GPU runtime for LoRA training.')
    from datasets import Dataset
    set_seed(42)
    def ds(split,limit):
        frame=pd.read_json(DATA/f'gen_{split}.jsonl',lines=True)
        frame=cap_by_era(frame, max(1, limit // frame.era.nunique()))
        frame=frame.sample(frac=1,random_state=42).head(limit)
        records=[encode_pair(tok,r['first'],r['second']) for r in frame.to_dict('records')]
        rows=[r for r in records if r is not None]
        print(split,': kept',len(rows),'of',len(records))
        if not rows:
            raise ValueError('No tokenized examples fit max_length=256.')
        return Dataset.from_list(rows)
    train=ds('train',128 if args.smoke else 12000)
    val=ds('validation',32 if args.smoke else 500)
    base.config.use_cache=False
    model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
                         lora_dropout=.05,target_modules=['q_proj','v_proj']))
    model.print_trainable_parameters()
    cfg=TrainingArguments(output_dir=str(target/'checkpoints'),learning_rate=1e-4,
        per_device_train_batch_size=2,per_device_eval_batch_size=2,gradient_accumulation_steps=8,
        num_train_epochs=1 if args.smoke else args.epochs,fp16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
        load_best_model_at_end=True,metric_for_best_model='eval_loss',
        report_to='none',seed=42,label_names=['labels'])
    guard_run(target, DATA, MODEL, cfg.to_dict())
    trainer=Trainer(model=model,args=cfg,train_dataset=train,eval_dataset=val,
                    processing_class=tok,data_collator=CompletionCollator(tok.pad_token_id))
    # نحفظ نقاط التوقف محليًا ونستأنف بنفس إعدادات التجربة.
    from transformers.trainer_utils import get_last_checkpoint
    checkpoint = get_last_checkpoint(str(target/'checkpoints')) if (target/'checkpoints').exists() else None
    trainer.train(resume_from_checkpoint=checkpoint)
    model.config.use_cache=True
    trainer.save_model(str(target))
    tok.save_pretrained(target)
    write_training_info(target, DATA, MODEL)
    info = json.loads((target/'training_info.json').read_text(encoding='utf-8'))
    info.update(training_arguments=cfg.to_dict(), smoke=args.smoke, train_examples=len(train), validation_examples=len(val))
    (target/'training_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    (target/'validation_metrics.json').write_text(json.dumps(trainer.evaluate(),indent=2), encoding='utf-8')
    print('Saved LoRA adapter, not a standalone base model:',target)

if __name__=='__main__':
    main()
