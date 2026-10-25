"""تلخيص تقييم بشري حقيقي؛ لا نملأ أي تقييم ناقص بقيم افتراضية."""
import argparse
from pathlib import Path
import pandas as pd

CRITERIA = ['language', 'meaning', 'poetry', 'no_repetition']


def summarize(ratings, key):
    """يمكن جمع ملف أكثر من مقيم بعد إضافة عمود rater لكل واحدة."""
    ratings, key = ratings.copy(), key.copy()
    if 'rater' not in ratings:
        ratings['rater'] = 'rater_1'
    if ratings[['id', 'version', 'rater']].duplicated().any():
        raise ValueError('هناك تقييم مكرر لنفس النص والنسخة والمقيم.')
    if key[['id', 'version']].duplicated().any():
        raise ValueError('مفتاح التقييم يحتوي أسطرًا مكررة.')
    if set(key.actual_version) != {'before', 'after'}:
        raise ValueError('المفتاح يجب أن يربط النسخ بقبل التدريب وبعده.')
    for _, frame in key.groupby('id'):
        if len(frame) != 2 or set(frame.version) != {'A', 'B'} or set(frame.actual_version) != {'before', 'after'}:
            raise ValueError('كل مدخل يحتاج نسختي A/B، واحدة قبل التدريب والأخرى بعده.')
    for field in CRITERIA:
        ratings[field] = pd.to_numeric(ratings[field], errors='coerce')
        if ratings[field].isna().any() or not ratings[field].isin([1, 2, 3, 4, 5]).all():
            raise ValueError(f'أكملي {field} بدرجات من 1 إلى 5، ولا تتركي قيما ناقصة.')
    expected = set(zip(key.id, key.version))
    for _, frame in ratings.groupby('rater'):
        if set(zip(frame.id, frame.version)) != expected:
            raise ValueError('كل مقيم يجب أن يقيم نفس مجموعة A/B كاملة.')
    merged = ratings.merge(key, on=['id', 'version'], validate='many_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('التقييم لا يطابق المفتاح.')
    # متوسطات المقيمين وصفية؛ لا نسميها دلالة إحصائية أو إثباتًا لجودة الشعر.
    means = merged.groupby(['rater', 'actual_version'])[CRITERIA].mean().reset_index()
    overview = means.groupby('actual_version')[CRITERIA].mean()
    overview['raters'] = means.groupby('actual_version').rater.nunique()
    return overview, means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    overview, raters = summarize(pd.read_csv(args.ratings), pd.read_csv(args.key))
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    overview.to_csv(dest/'human_summary.csv', encoding='utf-8-sig')
    raters.to_csv(dest/'human_per_rater.csv', index=False, encoding='utf-8-sig')
    print(overview.to_string())


if __name__ == '__main__':
    main()
