"""نحفظ البيانات والتقارير والملفات الصغيرة؛ لا نملأ Drive بنقاط التوقف."""
import argparse
import json
import shutil
from pathlib import Path


def sync(root, dest, include_data=True):
    root, dest = Path(root), Path(dest)
    if root.resolve() == dest.resolve():
        raise ValueError('مجلد الحفظ يجب أن يختلف عن مجلد التشغيل.')
    skipped = []
    for section in ['data', 'outputs', 'report']:
        if section == 'data' and not include_data:
            continue
        source = root/section
        if not source.exists():
            continue
        for path in source.rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            # أوزان AraBERT وcheckpoints كبيرة؛ ننزلها بأرشيف مستقل.
            if 'checkpoints' in relative.parts or path.stat().st_size > 50*1024*1024:
                skipped.append(str(relative))
                continue
            target = dest/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'storage_manifest.json').write_text(json.dumps(dict(
        skipped_large_or_checkpoint_files=skipped,
        note='Reports are not model weights. Export the trained model before ending Colab.'), indent=2), encoding='utf-8')
    print('Saved data and small results:', dest)
    print('Skipped checkpoint/large files:', len(skipped))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--skip-data', action='store_true')
    args = p.parse_args()
    sync(args.root, args.dest, include_data=not args.skip_data)
