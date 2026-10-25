"""Data preparation; no model downloads are needed to import this module."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd

CLASSICAL_METERS = {'بحر '+m for m in ['الطويل','المديد','البسيط','الوافر','الكامل','الهزج','الرجز','الرمل','السريع','المنسرح','الخفيف','المضارع','المقتضب','المجتث','المتقارب','المتدارك']}

MARKS = re.compile(r'[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]')

def clean(text):
    """Keep spelling; strip diacritics/tatweel consistently for both tasks."""
    text = unicodedata.normalize('NFC', str(text or ''))
    return re.sub(r'\s+', ' ', MARKS.sub('', text).replace('ـ', '')).strip()

def key(text):
    return re.sub(r'[^\u0621-\u063a\u0641-\u064a\u066e-\u06d3a-zA-Z0-9]', '', clean(text)).lower()

def prepare_poems(raw):
    required = {'poem verses', 'poet name', 'poet era', 'poem url'}
    if not required <= set(raw.columns):
        raise ValueError(f'Unexpected dataset schema. Missing {required - set(raw.columns)}')
    records = []
    for row in raw.to_dict('records'):
        verses = row['poem verses']
        if verses is None or isinstance(verses, str):
            continue  # Never guess how a flat string is divided into hemistichs.
        verses = list(verses)
        if not verses or any(not isinstance(x, str) or not clean(x) for x in verses):
            continue  # Do not drop individual blank elements and shift pair alignment.
        halves = [clean(x) for x in verses]
        poet, era = clean(row['poet name']), clean(row['poet era'])
        meter=clean(row.get('poem meter',''))
        if meter not in CLASSICAL_METERS:
            continue  # Conservative first release: skip free verse and unverified forms.
        if poet in ('', 'nan', 'None') or era in ('', 'nan', 'None'):
            continue
        text = ' '.join(halves)
        if len(key(text)) < 20:
            continue
        ident = hashlib.sha256(key(text).encode()).hexdigest()
        records.append(dict(poem_id=ident, poet=poet, poet_key=key(poet), era=era,
                            halves=halves, source=str(row['poem url'] or ''),
                            text=text))
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError('No usable poems found. Inspect the raw records first.')
    # Remove conflicting era labels assigned to the same normalized author.
    conflicts = df.groupby('poet_key').era.nunique()
    df = df[~df.poet_key.isin(conflicts[conflicts > 1].index)]
    # Remove duplicate poems even when attributed to multiple poets.
    return df.drop_duplicates('poem_id').reset_index(drop=True)

def split_poems(df, eras, seed=42):
    # نقسم شعراء كل عصر على حدة، وليس الأبيات عشوائيًا.
    if len(set(eras)) < 2:
        raise ValueError('Choose at least two eras.')
    df = df[df.era.isin(eras)].copy()
    if set(df.era) != set(eras):
        raise ValueError('Missing selected era.')
    if df.groupby('poet_key').era.nunique().max() > 1:
        raise ValueError('A normalized poet has conflicting era labels; review metadata.')
    groups = {'train': [], 'validation': [], 'test': []}
    for era in sorted(eras):
        era_data = df[df.era == era]
        poets = sorted(era_data.poet_key.unique())
        poets = pd.Series(poets).sample(frac=1, random_state=seed).tolist()
        if len(poets) < 9:
            raise ValueError(f'{era}: need at least 9 poets; found {len(poets)}')
        count = max(3, round(len(poets) * .15))
        names = {'validation': poets[:count], 'test': poets[count:2*count],
                 'train': poets[2*count:]}
        for split, authors in names.items():
            groups[split].append(era_data[era_data.poet_key.isin(authors)])
    parts = {}
    for split, frames in groups.items():
        parts[split] = pd.concat(frames, ignore_index=True)
        parts[split]['split'] = split
    assert_disjoint(parts, 'poet_key')
    assert_disjoint(parts, 'poem_id')
    return parts

def assert_disjoint(parts, field):
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            assert not set(parts[a][field]) & set(parts[b][field]), (a, b, field)

def build_examples(parts, max_per_poem=8):
    """Pairing is enabled only by an explicit reviewed choice in notebook 00."""
    cls, gen = {}, {}
    seen_halves = set()
    for split in ('train', 'validation', 'test'):
        cs, gs, this_split = [], [], set()
        for row in parts[split].to_dict('records'):
            halves = row['halves']
            if len(halves) % 2:
                continue  # Odd records are excluded, rather than silently mispaired.
            added = 0
            for i in range(0, len(halves), 2):
                first, second = halves[i:i+2]
                k1, k2 = key(first), key(second)
                if min(len(k1), len(k2)) < 6 or k1 == k2:
                    continue
                if k1 in seen_halves or k2 in seen_halves:
                    continue  # Earlier splits win; held-out repeated content is removed.
                if k1 in this_split or k2 in this_split:
                    continue
                common = dict(poem_id=row['poem_id'], poet=row['poet'], era=row['era'],
                              source=row['source'])
                cs.append(dict(**common, text=first+' '+second))
                gs.append(dict(**common, first=first, second=second))
                this_split.update((k1, k2))
                added += 1
                if added >= max_per_poem:
                    break
        cls[split], gen[split] = pd.DataFrame(cs), pd.DataFrame(gs)
        if not cs:
            raise ValueError(f'No examples in {split}; inspect pair formatting and filtering.')
        seen_halves.update(this_split)
    return cls, gen

def cap_by_era(df, cap, seed=42):
    # نعطي أولوية لتنوع الشعراء: مثال لكل شاعر، ثم مثال ثان، وهكذا.
    samples = []
    for era, group in df.groupby('era'):
        group = group.sample(frac=1, random_state=seed).copy()
        group['_author'] = group['poet'].map(key)
        group['_round'] = group.groupby('_author').cumcount()
        group = group.sort_values('_round', kind='stable').head(cap)
        samples.append(group.drop(columns=['_author', '_round']))
    return pd.concat(samples, ignore_index=True)

def save_prepared(parts, cls, gen, out, eras):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = {e: i for i, e in enumerate(sorted(eras))}
    for split in parts:
        c = cls[split].copy()
        c['label'] = c.era.map(labels)
        c.to_json(out/f'cls_{split}.jsonl', orient='records', lines=True, force_ascii=False)
        gen[split].to_json(out/f'gen_{split}.jsonl', orient='records', lines=True, force_ascii=False)
    manifest = pd.concat(parts.values()).drop(columns=['halves', 'text'])
    manifest.to_csv(out/'split_manifest.csv', index=False)
    (out/'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')


def data_signature(data_dir):
    # نربط كل نموذج بالبيانات التي تدرب عليها، حتى لا نخلط التجارب.
    data_dir = Path(data_dir)
    digest = hashlib.sha256()
    for name in ['labels.json'] + [f'{task}_{split}.jsonl'
                for task in ['cls','gen'] for split in ['train','validation','test']]:
        digest.update(name.encode())
        with (data_dir/name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()

def write_training_info(path, data_dir, model_name):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    info = dict(data_signature=data_signature(data_dir), model=model_name,
                sampling='equal era caps; author round robin', seed=42)
    (path/'training_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

def verify_training_info(path, data_dir):
    info_file = Path(path)/'training_info.json'
    if not info_file.exists():
        raise ValueError('Model has no data signature. Retrain it in this project version.')
    info = json.loads(info_file.read_text(encoding='utf-8'))
    if info['data_signature'] != data_signature(data_dir):
        raise ValueError('Data changed after training. Use matching data or retrain the model.')


def guard_run(path, data_dir, model, arguments):
    """نمنع استئناف تدريب ببيانات أو إعدادات مختلفة داخل المجلد نفسه."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    expected = dict(data_signature=data_signature(data_dir), model=model,
                    arguments=arguments)
    # التحويل مرتين يجعل مقارنة القوائم والقيم المكتبية ثابتة بعد الحفظ.
    expected = json.loads(json.dumps(expected, default=str))
    filename = path/'run_manifest.json'
    if filename.exists() and json.loads(filename.read_text(encoding='utf-8')) != expected:
        raise ValueError('الإعدادات أو البيانات تغيرت. اختاري مجلد تجربة جديدًا بدل استئناف هذه التجربة.')
    filename.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')


def verify_classifier_info(path, data_dir):
    """ندعم النموذج القديم من سجله الأصلي دون اختراع سجل تدريب جديد له."""
    path, data_dir = Path(path), Path(data_dir)
    if (path/'training_info.json').exists():
        verify_training_info(path, data_dir)
        return
    legacy_file = path/'experiment.json'
    if not legacy_file.exists():
        raise ValueError('لا يوجد سجل يربط المصنف بالبيانات.')
    info = json.loads(legacy_file.read_text(encoding='utf-8'))
    required = {'cls_train.jsonl', 'cls_validation.jsonl', 'labels.json'}
    hashes = info.get('data_sha256', {})
    if not required <= set(hashes):
        raise ValueError('سجل التجربة القديمة لا يحتوي تواقيع التدريب والتحقق والتسميات.')
    for name in sorted(required):
        actual = hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
        if actual != hashes[name]:
            raise ValueError('بيانات المصنف القديم غير مطابقة: ' + name)
