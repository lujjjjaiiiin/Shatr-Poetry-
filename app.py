"""واجهة شطر. لا نعرض توقعات أو تكملات وهمية عند غياب النماذج."""
import html
import os
import secrets
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import clean, key, cap_by_era, verify_training_info, verify_classifier_info
from classification_config import selected_folder

CSS = """
.gradio-container {
  --body-background-fill:#f7f4ee; --body-text-color:#233c36;
  --body-text-color-subdued:#607169; --background-fill-primary:#fffdf8;
  --background-fill-secondary:#f0ece3; --block-background-fill:#fffdf8;
  --block-title-text-color:#233c36; --block-label-text-color:#233c36;
  --block-label-background-fill:#eee8dc; --input-background-fill:#ffffff;
  --input-text-color:#233c36; --input-placeholder-color:#78847c;
  --border-color-primary:#e4dccd; --border-color-accent:#b39860;
  --button-secondary-background-fill:#eee8dc;
  --button-secondary-background-fill-hover:#e4dccd;
  --button-secondary-text-color:#233c36;
}
.gradio-container {max-width:1120px!important; direction:rtl; background:#f7f4ee!important;
  color:#233c36!important; font-family:Tahoma,Arial,sans-serif!important;}
.hero {padding:44px 32px;border-radius:26px;background:#173f36;color:#fff8e9;
  margin:8px 0 22px;position:relative;overflow:hidden;}
.hero:after {content:'ش';position:absolute;left:28px;top:-65px;font-size:260px;
  color:#ffffff0c;line-height:1.4;pointer-events:none;}
.eyebrow {color:#d2b77e;font-size:13px;letter-spacing:2px;}
.hero h1 {font-size:64px!important;color:#fff8e9!important;margin:5px 0!important;}
.hero h2 {font-size:25px!important;color:#fff8e9!important;font-weight:400!important;}
.hero p {color:#e1e7de;max-width:650px;line-height:2;font-size:15px;}
.tiny {color:#6a786d;font-size:13px;line-height:1.9;}
.section {background:white;border:1px solid #e6e0d4;border-radius:20px;padding:22px;}
.section h3,.section p {color:#233c36!important;}
.gradio-container .prose,.gradio-container .prose p,.gradio-container .prose h3 {
  direction:rtl!important;text-align:right!important;
}
.result-card {background:#fffdf8;border:1px solid #e4dccd;border-radius:16px;padding:18px;
  margin:12px 0;direction:rtl;}
.verse {font-size:19px;line-height:2;color:#183f36;margin:0 0 10px;}
.meta {font-size:12px;color:#6a786d;line-height:1.9;}
.meta a {color:#80632c;text-decoration:underline;}
.hint {background:#eee9de;border-radius:12px;padding:13px 16px;line-height:1.9;color:#5c665c;}
textarea {direction:rtl!important;text-align:right!important;font-size:18px!important;line-height:1.9!important;}
button {font-family:inherit!important;}
footer {display:none!important;}
@media(max-width:600px) {.hero {padding:28px 20px;}.hero h1 {font-size:48px!important;}}
"""


def model_status(folder, data, kind):
    """نتأكد من ارتباط الأوزان بنفس البيانات قبل السماح باستخدامها."""
    try:
        if kind == 'classification':
            verify_classifier_info(folder, data)
        else:
            verify_training_info(folder, data)
        needed = 'adapter_config.json' if kind == 'completion' else 'config.json'
        if not (folder / needed).exists():
            raise ValueError('ملفات النموذج غير مكتملة.')
        patterns = ['adapter_model.safetensors', 'adapter_model.bin'] if kind == 'completion' else ['model*.safetensors', 'pytorch_model*.bin']
        if not any(list(folder.glob(pattern)) for pattern in patterns):
            raise ValueError('ملف الأوزان غير موجود.')
        return True, 'جاهز'
    except (ValueError, OSError, KeyError) as error:
        return False, str(error)


def safe_source(value):
    """روابط المصدر للقراءة فقط؛ نرفض بروتوكولات مثل javascript."""
    value = str(value)
    parsed = urlparse(value)
    return value if parsed.scheme in ('http', 'https') and parsed.netloc else ''


def cards(rows):
    if not rows:
        return '<div class="hint">لم نجد تشابهًا لفظيًا مناسبًا في فهرس التدريب.</div>'
    result = []
    for row in rows:
        source = safe_source(row['source'])
        link = f'<a href="{html.escape(source, quote=True)}" target="_blank" rel="noopener noreferrer">اقرئي المصدر ↗</a>' if source else 'المصدر غير متاح'
        result.append('<article class="result-card"><p class="verse">' + html.escape(row['text']) +
                      '</p><div class="meta">' + html.escape(row['poet']) + ' · ' + html.escape(row['era']) +
                      ' · ' + link + '</div></article>')
    return ''.join(result)


class ShatrService:
    """نفصل عمل النماذج عن الواجهة لتبقى الأكواد سهلة الفهم والاختبار."""
    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / 'data'
        self.classifier_dir = selected_folder(self.root)
        self.adapter_dir = self.root / 'outputs/shatr_adapter'
        self.classifier_ready, self.classifier_note = model_status(self.classifier_dir, self.data, 'classification')
        self.completion_ready, self.completion_note = model_status(self.adapter_dir, self.data, 'completion')
        self.corpus = None
        path = self.data / 'cls_train.jsonl'
        if path.exists():
            corpus = pd.read_json(path, lines=True)
            corpus = cap_by_era(corpus, 2500).drop_duplicates('text').reset_index(drop=True)
            if len(corpus):
                self.corpus = corpus
                self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=60000)
                self.matrix = self.vectorizer.fit_transform(corpus.text.map(clean))

    def search(self, text):
        text = clean(text)
        if len(text.split()) < 4:
            raise gr.Error('اكتبي بيتًا أو مقطعًا من أربع كلمات على الأقل.')
        if len(text) > 3000:
            raise gr.Error('اختاري مقطعًا أقصر للبحث.')
        if self.corpus is None:
            raise gr.Error('فهرس الأبيات غير جاهز. جهزي بيانات المشروع أولًا.')
        similarities = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        found, poets = [], set()
        for index in similarities.argsort()[::-1]:
            row = self.corpus.iloc[int(index)]
            # تنويع النتائج، مع استبعاد التطابق الحرفي حتى لا نسميه اكتشافًا.
            if similarities[index] <= 0:
                break
            if key(row.text) == key(text) or key(row.poet) in poets:
                continue
            poets.add(key(row.poet))
            found.append(row.to_dict())
            if len(found) == 3:
                break
        return cards(found)

    @lru_cache(maxsize=1)
    def load_classifier(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        verify_classifier_info(self.classifier_dir, self.data)
        tok = AutoTokenizer.from_pretrained(self.classifier_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(self.classifier_dir, local_files_only=True).eval()
        return tok, model

    def analyze(self, text):
        if not self.classifier_ready:
            raise gr.Error('ميزة التصنيف غير متاحة حاليًا؛ جربي البحث اللفظي.')
        import torch
        text = clean(text)
        if len(text.split()) < 4:
            raise gr.Error('اكتبي بيتًا أو مقطعًا من أربع كلمات على الأقل.')
        tok, model = self.load_classifier()
        if len(tok(text)['input_ids']) > 192:
            raise gr.Error('المقطع أطول من حد النموذج؛ اختاري مقطعًا أقصر.')
        with torch.inference_mode():
            scores = model(**tok(text, return_tensors='pt')).logits.softmax(-1)[0].tolist()
        labels = {model.config.id2label[i]: float(score) for i, score in enumerate(scores)}
        return labels, self.search(text)

    @lru_cache(maxsize=1)
    def load_generator(self):
        import torch
        from peft import PeftModel
        from train_completion import load_base
        verify_training_info(self.adapter_dir, self.data)
        tok, base = load_base()
        model = PeftModel.from_pretrained(base, self.adapter_dir)
        model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        return tok, model

    def complete(self, first):
        if not self.completion_ready:
            raise gr.Error('ميزة التكملة قيد التجهيز وغير متاحة حاليًا.')
        first = clean(first)
        if len(first.split()) < 2:
            raise gr.Error('اكتبي شطرًا أولًا من كلمتين على الأقل.')
        if len(first) > 1000:
            raise gr.Error('اكتبي شطرًا واحدًا قصيرًا.')
        from train_completion import generate
        tok, model = self.load_generator()
        try:
            result = generate(model, tok, first, seed=secrets.randbelow(2**31))
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return result or 'لم ينتج النموذج تكملة هذه المرة؛ جربي اقتراحًا جديدًا.'


def build_app(root=None):
    service = ShatrService(root or os.environ.get('SHATR_ROOT', '.'))
    classifier_name = 'AraPoemBERT' if service.classifier_dir.name == 'arapoembert_model' else 'AraBERT'
    cls_status = 'جاهز' if service.classifier_ready else 'قيد التجهيز'
    gen_status = 'جاهزة' if service.completion_ready else 'قيد التجهيز'
    with gr.Blocks(title='شطر | لكل بيت بصمة، ولكل شطر بداية', css=CSS,
                   theme=gr.themes.Soft(primary_hue='emerald', neutral_hue='stone')) as demo:
        gr.HTML('<header class="hero" dir="rtl"><div class="eyebrow">مساحة عربية للشعر والفضول</div>'
                '<h1>شطر</h1><h2>لكل بيت بصمة… ولكل شطر بداية</h2>'
                '<p>اقتربي من ملامح الشعر العربي، واستكشفي أثر الكلمات، وامنحي شطرك بداية جديدة.</p></header>')
        with gr.Tab('اكتشفي البصمة'):
            with gr.Row():
                with gr.Column(scale=6, elem_classes=['section']):
                    gr.Markdown('### أيّ عصر تلمح كلماتك؟\nاكتبي بيتًا أو مقطعًا فصيحًا لنستكشف أقرب عصر ضمن فئات المشروع.')
                    text = gr.Textbox(label='النص الشعري', placeholder='اكتبي كلماتك هنا…', lines=4, max_lines=8)
                    with gr.Row():
                        analyze_button = gr.Button('اكتشفي البصمة ←', variant='primary', interactive=service.classifier_ready)
                        search_button = gr.Button('استكشفي التشابه اللفظي', interactive=service.corpus is not None)
                    gr.Markdown(f'حالة نموذج العصر: **{cls_status}**', elem_classes=['tiny'])
                with gr.Column(scale=4, elem_classes=['section']):
                    prediction = gr.Label(label='ترجيحات العصور', num_top_classes=4)
                    gr.Markdown('الجاهلي · الأندلسي · العثماني · الحديث', elem_classes=['tiny'])
                    gr.HTML('<div class="hint">التصنيف لا يثبت تاريخ النص أو هوية مؤلفه. درجات النموذج غير معايرة، وقد يخطئ خصوصًا مع النصوص خارج هذه الفئات.</div>')
            gr.Markdown('### صدى الكلمات\nأبيات من بيانات التدريب تتشابه لفظيًا مع نصك؛ التشابه لا يثبت الاقتباس أو التأثر.')
            similar = gr.HTML('<div class="hint">ستظهر هنا أبيات مع أسماء شعرائها وروابط مصادرها.</div>')
            analyze_button.click(service.analyze, text, [prediction, similar], concurrency_limit=1)
            search_button.click(service.search, text, similar, concurrency_limit=1)
        with gr.Tab('أكملي الشطر'):
            with gr.Row():
                with gr.Column(elem_classes=['section']):
                    gr.Markdown('### ابدئي… واتركي للكلمات امتدادًا\nاكتبي شطرًا واحدًا، ثم جربي تكملة أو اقتراحًا آخر.')
                    first = gr.Textbox(label='الشطر الأول', placeholder='هنا تبدأ الحكاية…', lines=3)
                    generate_button = gr.Button('اقترحي شطرًا ←', variant='primary', interactive=service.completion_ready)
                with gr.Column(elem_classes=['section']):
                    completion = gr.Textbox(label='الشطر المقترح', lines=3, interactive=False)
                    another = gr.Button('اقتراح جديد ↻', interactive=service.completion_ready)
                    gr.Markdown(f'حالة التكملة: **{gen_status}**', elem_classes=['tiny'])
            gr.HTML('<div class="hint">هذه مساعدة إبداعية تجريبية؛ لا نفترض صحة الوزن أو القافية أو أصالة النص دون تحقق مستقل.</div>')
            generate_button.click(service.complete, first, completion, concurrency_limit=1)
            another.click(service.complete, first, completion, concurrency_limit=1)
        with gr.Tab('عن شطر'):
            gr.Markdown('''### من قراءة الشعر إلى المشاركة في كتابته
مشروع تخرج لطالبتين، للطلاب ومحبي الأدب وكتّاب الشعر المبتدئين.

**اكتشفي البصمة:** نقارن AraBERT وAraPoemBERT بخط أساس، ونعتمد الأفضل على التحقق لتصنيف العصر.

**أكملي الشطر:** نموذج Qwen3-0.6B يجري تخصيصه باستخدام LoRA، ويحسب التدريب الخسارة على التكملة فقط.

**صدى الكلمات:** استرجاع لفظي باستخدام TF-IDF من قسم التدريب فقط، مع الشاعر والمصدر.

نفصل الشعراء والقصائد والأشطر المتطابقة بين الأقسام. لا نعرض نتائج تدريب قبل تنفيذها، ولا نضمن تحسن الأداء.
''')
            gr.Markdown(f'المصنف الحالي في هذه الجلسة: **{classifier_name}**.')
    return demo.queue(default_concurrency_limit=1, api_open=False)


if __name__ == '__main__':
    build_app().launch(share=False)
