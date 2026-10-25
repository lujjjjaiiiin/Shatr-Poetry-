"""Offline checks: data grouping, contamination filtering, masking, notebook syntax, baseline."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from data_utils import prepare_poems,split_poems,build_examples,save_prepared,key,clean,cap_by_era,data_signature,write_training_info,verify_training_info

def main():
    rows=[]
    for era in range(3):
        for author in range(12):
            for poem in range(3):
                tag=f'{era}{author:02}{poem}'
                rows.append({'poet name':f'شاعر {era} {author}', 'poet era':f'العصر {era}',
                 'poem meter':'بحر الكامل','poem url':f'https://example.org/{tag}',
                 'poem verses':[f'يا ليل قلبي شاعر حالم {tag} صدر',f'وفي الروح شوق وحنين {tag} عجز']})
    raw=pd.DataFrame(rows)
    df=prepare_poems(raw)
    assert len(df)==108
    parts=split_poems(df,sorted(df.era.unique()))
    cls,gen=build_examples(parts)
    seen=set()
    for name in ('train','validation','test'):
        assert set(cls[name].era)==set(df.era)
        assert cls[name].groupby('era').poet.nunique().min() >= 3
        keys={key(s) for r in gen[name].to_dict('records') for s in (r['first'],r['second'])}
        assert not seen & keys
        seen |= keys
    # Author-aware selection retains all authors when the budget allows it.
    sampled=cap_by_era(cls['train'],18)
    assert sampled.groupby('era').poet.nunique().min()==6
    # Fixed seed makes the partition reproducible.
    again=split_poems(df,sorted(df.era.unique()))
    assert set(again['test'].poem_id)==set(parts['test'].poem_id)
    # Exact duplicate poem attributed to another poet must not survive twice.
    dup=raw.iloc[0].to_dict();dup['poet name']='شاعر آخر'
    assert len(prepare_poems(pd.concat([raw,pd.DataFrame([dup])],ignore_index=True)))==108
    blank=raw.iloc[0].to_dict();blank['poem verses']=['صدر واضح جدا','','عجز واضح جدا']
    assert len(prepare_poems(pd.concat([raw,pd.DataFrame([blank])],ignore_index=True)))==108
    repeated_parts={k:v.copy() for k,v in parts.items()}
    train_first=parts['train'].iloc[0].halves[0]
    victim=repeated_parts['validation'].index[0]
    repeated_parts['validation'].at[victim,'halves']=[train_first,'عجز مختلف للاختبار فقط']
    _,gg=build_examples(repeated_parts)
    assert train_first not in set(gg['validation']['first'])
    # Execute only tokenizer/padding helpers without installing/downloading model packages.
    tree=ast.parse(Path('train_completion.py').read_text(encoding='utf-8'))
    defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in
          {'messages','prompt_ids','encode_pair','CompletionCollator'}]
    class FakeTorch:
        tensor=staticmethod(np.array)
    scope={'clean':clean,'torch':FakeTorch}
    exec(compile(ast.Module(body=defs,type_ignores=[]),'helpers','exec'),scope)
    class FakeTok:
        eos_token_id=9
        def apply_chat_template(self,*args,**kwargs):return [1,2,3]
        def encode(self,text,**kwargs):return [4]*len(text.split())
    pair=scope['encode_pair'](FakeTok(),'صدر أول','عجز قصير')
    assert pair['labels']==[-100,-100,-100,4,4,9]
    assert scope['encode_pair'](FakeTok(),'صدر',' '.join(['كلمة']*300)) is None
    longer=scope['encode_pair'](FakeTok(),'صدر','عجز أطول من السابق')
    batch=scope['CompletionCollator'](9)([pair,longer])
    assert batch['labels'][0,-1]==-100 and batch['labels'][0,5]==9
    assert batch['attention_mask'][0,-1]==0
    for path in Path('.').glob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    for path in Path('.').glob('*.ipynb'):
        nb=json.loads(path.read_text(encoding='utf-8'))
        assert nb['nbformat']==4
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source'])
                source='\n'.join(line for line in source.splitlines() if not line.lstrip().startswith(('!','%')))
                compile(source,str(path),'exec')
    with tempfile.TemporaryDirectory() as tmp:
        save_prepared(parts,cls,gen,Path(tmp)/'data',sorted(df.era.unique()))
        env={**os.environ,'SHATR_ROOT':tmp}
        subprocess.run([sys.executable,'train_classifier.py','baseline','--smoke'],env=env,check=True,
                       stdout=subprocess.DEVNULL)
        assert (Path(tmp)/'outputs/baseline_validation.json').exists()
        write_training_info(Path(tmp)/'outputs/check_model',Path(tmp)/'data','test')
        verify_training_info(Path(tmp)/'outputs/check_model',Path(tmp)/'data')
        # Full migration runs against already prepared data in a new project folder.
        from prepare_project import prepare
        final_root=Path(tmp)/'final'
        prepare(Path(tmp),final_root)
        assert (final_root/'data/data_summary.csv').exists()
        # Prevent silent replacement of a fixed prepared split.
        try:
            prepare(Path(tmp),final_root)
            raise AssertionError('Expected existing-data guard')
        except FileExistsError:
            pass
        # Report uses generated metrics, not invented final results.
        report_env={**os.environ,'SHATR_ROOT':tmp}
        subprocess.run([sys.executable,'make_report.py'],env=report_env,check=True,
                       stdout=subprocess.DEVNULL)
        assert (Path(tmp)/'report/Project_Report.md').exists()
        # Test the plot path using synthetic metrics only inside temporary storage.
        model_dir=Path(tmp)/'outputs/basma_model'
        model_dir.mkdir(exist_ok=True)
        baseline=(Path(tmp)/'outputs/baseline_validation.json').read_text(encoding='utf-8')
        (model_dir/'classification_validation.json').write_text(baseline, encoding='utf-8')
        subprocess.run([sys.executable,'make_report.py'],env=report_env,check=True,
                       stdout=subprocess.DEVNULL)
        assert (Path(tmp)/'report/confusion_matrix.png').exists()
        signature_before=data_signature(Path(tmp)/'data')
        with (Path(tmp)/'data/cls_train.jsonl').open('a') as stream:
            stream.write('\n')
        assert signature_before != data_signature(Path(tmp)/'data')
        try:
            verify_training_info(Path(tmp)/'outputs/check_model',Path(tmp)/'data')
            raise AssertionError('Expected signature mismatch')
        except ValueError:
            pass
    print('PASS: grouped split, author/poem separation, duplicate removal, cross-split hemistich filter,')
    print('      author-aware sampling, migration, fixed-split guard, model-data signature,')
    print('      completion-only labels, EOS/padding masks, notebook syntax, baseline, report and plot.')
    print('NOT RUN: real Ashaar download, GPU training, model reload/inference, Gradio UI.')
if __name__=='__main__':
    main()
