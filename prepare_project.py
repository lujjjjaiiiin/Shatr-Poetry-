"""Upgrade the already-reviewed notebook-00 data without changing the old project."""
import argparse
import json
from pathlib import Path
import pandas as pd
from data_utils import key, clean, split_poems, build_examples, save_prepared, data_signature


def prepare(old_root, new_root):
    old_root, new_root = Path(old_root), Path(new_root)
    if old_root.resolve() == new_root.resolve():
        raise ValueError('Use a separate final project folder; keep the original experiment.')
    if (new_root/'data/labels.json').exists():
        raise FileExistsError('Prepared data already exists. Reuse it; do not silently re-split.')
    labels = json.loads((old_root/'data/labels.json').read_text(encoding='utf-8'))
    # نجمع الأزواج القديمة التي راجعتما تنسيقها بالفعل في دفتر 00.
    frames = [pd.read_json(old_root/f'data/gen_{split}.jsonl', lines=True)
              for split in ['train','validation','test']]
    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.drop_duplicates(['poem_id','first','second'])
    records = []
    for poem_id, group in pairs.groupby('poem_id', sort=True):
        if group.poet.map(key).nunique() != 1 or group.era.nunique() != 1:
            raise ValueError(f'Conflicting poem metadata: {poem_id}')
        row = group.iloc[0]
        halves = []
        for item in group.itertuples():
            halves.extend([clean(item.first), clean(item.second)])
        records.append(dict(poem_id=poem_id, poet=row.poet, poet_key=key(row.poet),
                            era=row.era, source=row.source, halves=halves,
                            text=' '.join(halves)))
    poems = pd.DataFrame(records)
    parts = split_poems(poems, sorted(labels))
    cls, gen = build_examples(parts)
    # نتحقق بعد الاستخراج أيضًا؛ قد تزيل إزالة التكرار أمثلة كاملة لشاعر.
    summaries = []
    for split in cls:
        table = cls[split].groupby('era').agg(examples=('text','size'), poets=('poet','nunique'))
        if set(table.index) != set(labels) or table.poets.min() < 3:
            raise ValueError(f'{split}: fewer than 3 retained poets in an era; review the data.')
        print('\n', split, '\n', table.to_string())
        summaries.append(table.reset_index().assign(split=split))
    save_prepared(parts, cls, gen, new_root/'data', sorted(labels))
    pd.concat(summaries).to_csv(new_root/'data/data_summary.csv',index=False,encoding='utf-8-sig')
    metadata = dict(source='already reviewed Ashaar pairs from previous notebook 00',
        split='one fixed seed; grouped by normalized poet separately within each era',
        seed=42, reused_old_pairs=True, data_signature=data_signature(new_root/'data'),
        note='Retains only pairs present in old prepared files, not the entire raw corpus. Old model is not reused.')
    (new_root/'data/preparation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2), encoding='utf-8')
    print('\nSaved corrected project to',new_root)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--old-root',required=True)
    parser.add_argument('--new-root',required=True)
    args=parser.parse_args()
    prepare(args.old_root,args.new_root)
