import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from db.database import save_comment, save_classification

# MODEL_PATH = Path("artifacts/tfidf_logreg_pipeline.joblib")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "tfidf_logreg_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found: {MODEL_PATH.resolve()}\n"
        "Make sure the trained pipeline is saved before starting the app."
    )

model = joblib.load(MODEL_PATH)

tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_feature_contributions(text: str, top_n: int = 10) -> pd.DataFrame:
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    row = X.toarray()[0]
    nonzero_idx = np.where(row > 0)[0]

    if len(nonzero_idx) == 0:
        return pd.DataFrame(columns=["feature", "direction", "contribution"])

    contributions = row[nonzero_idx] * coefs[nonzero_idx]
    features = feature_names[nonzero_idx]

    contrib_df = pd.DataFrame({
        "feature": features,
        "contribution_raw": contributions
    })

    contrib_df["direction"] = np.where(
        contrib_df["contribution_raw"] >= 0,
        "spam",
        "ham"
    )
    contrib_df["contribution"] = contrib_df["contribution_raw"].abs()

    contrib_df = (
        contrib_df[["feature", "direction", "contribution"]]
        .sort_values("contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return contrib_df


def build_confidence_bar(spam_proba: float, ham_proba: float, label: str) -> str:
    spam_width = max(2, int(spam_proba * 100))
    ham_width = max(2, int(ham_proba * 100))

    dominant_class = "SPAM" if label == "SPAM" else "HAM"
    dominant_text = f"Найбільш імовірний клас: {dominant_class}"

    return f"""
    <div class="confidence-card">
        <div class="confidence-header">
            <div class="confidence-title">Індикатор впевненості</div>
            <div class="confidence-caption">{dominant_text}</div>
        </div>

        <div class="bar-wrapper">
            <div class="bar-row">
                <div class="bar-label spam-label">SPAM</div>
                <div class="bar-track">
                    <div class="bar-fill spam-fill" style="width: {spam_width}%"></div>
                </div>
                <div class="bar-value">{spam_proba:.2%}</div>
            </div>

            <div class="bar-row">
                <div class="bar-label ham-label">HAM</div>
                <div class="bar-track">
                    <div class="bar-fill ham-fill" style="width: {ham_width}%"></div>
                </div>
                <div class="bar-value">{ham_proba:.2%}</div>
            </div>
        </div>
    </div>
    """


def build_summary_html(label: str, spam_proba: float, ham_proba: float, text: str) -> str:
    tone_class = "badge-spam" if label == "SPAM" else "badge-ham"
    verdict_text = "Ймовірний спам" if label == "SPAM" else "Схоже на звичайне повідомлення"

    chars = len(text)
    words = len(text.split()) if text.strip() else 0

    html = f"""
    <div class="result-card">
        <div class="result-header">
            <div>
                <div class="eyebrow">Результат класифікації</div>
                <div class="verdict">{verdict_text}</div>
            </div>
            <div class="verdict-badge {tone_class}">{label}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Ймовірність spam</div>
                <div class="stat-value">{spam_proba:.2%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Ймовірність ham</div>
                <div class="stat-value">{ham_proba:.2%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Символів</div>
                <div class="stat-value">{chars}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Слів</div>
                <div class="stat-value">{words}</div>
            </div>
        </div>
    </div>
    """
    return html


def predict_message(text: str):
    text = clean_text(text)

    if not text:
        empty_df = pd.DataFrame(columns=["feature", "direction", "contribution"])
        empty_result = "<div class='result-card'><div class='verdict'>Введіть текст для аналізу.</div></div>"
        empty_bar = """
        <div class="confidence-card">
            <div class="confidence-title">Індикатор впевненості</div>
            <div class="confidence-caption">Після аналізу тут з’явиться розподіл ймовірностей моделі.</div>
        </div>
        """
        return empty_result, empty_bar, empty_df

    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]

    ham_proba = float(proba[0])
    spam_proba = float(proba[1])

    print(ham_proba)

    label = "SPAM" if pred == 1 else "HAM"

    save_comm = save_comment(text, label) # збереження інформації про коментар в БД
    save_classification(save_comm, spam_proba * 100, ham_proba * 100, label)

    explanation_df = extract_feature_contributions(text, top_n=10)
    summary_html = build_summary_html(label, spam_proba, ham_proba, text)
    confidence_html = build_confidence_bar(spam_proba, ham_proba, label)

    return summary_html, confidence_html, explanation_df


EXAMPLES = [
    ["Вітаю! Ви виграли приз. Напишіть у приватні повідомлення просто зараз."],
    ["Дякую за відповідь, усе зрозуміло і працює добре."],
    ["Переходьте за посиланням, підписуйтесь на канал і отримуйте бонус."],
    ["Підкажіть, будь ласка, коли буде відправка замовлення?"],
    ["Терміново! Лише сьогодні унікальна пропозиція для всіх охочих."],
    ["Чи можна уточнити характеристики товару перед оплатою?"],
]

CUSTOM_CSS = """
.gradio-container {
    max-width: 1600px !important;
    margin: auto !important;
    padding-top: 20px !important;
}

.app-shell {
    background: linear-gradient(135deg, #eef2ff 0%, #e0f2fe 100%);
    border-radius: 28px;
    padding: 32px;
    color: #0f172a;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 36px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 16px;
    color: #334155;
    line-height: 1.7;
    max-width: 1100px;
}

.pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.pill {
    background: white;
    border: 1px solid #dbeafe;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    color: #1e293b;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
}

.result-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 20px;
    color: #0f172a;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
}

.eyebrow {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 6px;
}

.verdict {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
}

.verdict-badge {
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    white-space: nowrap;
}

.badge-spam {
    background: #fee2e2;
    color: #b91c1c;
}

.badge-ham {
    background: #dcfce7;
    color: #166534;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.stat-box {
    background: white;
    border-radius: 18px;
    padding: 14px;
    border: 1px solid #e2e8f0;
}

.stat-label {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 6px;
}

.stat-value {
    color: #0f172a;
    font-weight: 800;
    font-size: 22px;
}

.confidence-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    height: 100%;
}

.confidence-header {
    margin-bottom: 16px;
}

.confidence-title {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
}

.confidence-caption {
    font-size: 14px;
    color: #64748b;
}

.bar-wrapper {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.bar-row {
    display: grid;
    grid-template-columns: 70px 1fr 80px;
    gap: 12px;
    align-items: center;
}

.bar-label {
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
}

.spam-label {
    color: #b91c1c;
}

.ham-label {
    color: #166534;
}

.bar-track {
    width: 100%;
    height: 16px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
    position: relative;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.spam-fill {
    background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
}

.ham-fill {
    background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
}

.bar-value {
    text-align: right;
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
}

.soft-card {
    border-radius: 22px !important;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08) !important;
    border: 1px solid #e2e8f0 !important;
    overflow: visible !important;
}

.soft-card .label-wrap,
.soft-card .block-title,
.soft-card label,
.soft-card h1,
.soft-card h2,
.soft-card h3,
.soft-card p,
.soft-card span {
    line-height: 1.5 !important;
}

.soft-card .wrap,
.soft-card .prose,
.soft-card .gr-markdown,
.soft-card .gr-dataframe {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}

.center-block {
    max-width: 1400px;
    margin: 0 auto;
}
"""

with gr.Blocks(
    title="Spam Filter Demo",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
    ),
) as demo:
    gr.HTML(
        """
        <div class="app-shell">
            <div class="hero-title">Система автоматичного розпізнавання спаму в коментарях</div>
            <div class="hero-subtitle">
                Демонстраційний застосунок для дипломної роботи. Модель: TF-IDF + Logistic Regression.
                Додаток показує підсумковий клас, індикатор впевненості та найвпливовіші слова, які вплинули на рішення моделі.
            </div>
            <div class="pill-row">
                <div class="pill">Україномовні тексти</div>
                <div class="pill">Пояснювана модель</div>
                <div class="pill">Інтерактивна демонстрація</div>
                <div class="pill">TF-IDF + Logistic Regression</div>
            </div>
        </div>
        """
    )

    with gr.Row(elem_classes=["center-block"]):
        with gr.Column(scale=7):
            text_input = gr.Textbox(
                label="Текст повідомлення",
                placeholder="Введіть коментар або повідомлення для перевірки...",
                lines=10,
                elem_classes=["soft-card"],
            )

            with gr.Row():
                analyze_btn = gr.Button("Проаналізувати", variant="primary", size="lg")
                clear_btn = gr.Button("Очистити", variant="secondary")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input],
                label="Приклади для швидкої перевірки",
            )

        with gr.Column(scale=5):
            confidence_html = gr.HTML(
                value="""
                <div class="confidence-card">
                    <div class="confidence-title">Індикатор впевненості</div>
                    <div class="confidence-caption">Після аналізу тут з’явиться розподіл ймовірностей моделі.</div>
                </div>
                """
            )

    with gr.Column(elem_classes=["center-block"]):
        result_html = gr.HTML(
            value="<div class='result-card'><div class='verdict'>Результат з’явиться тут після аналізу.</div></div>"
        )

        explanation_table = gr.Dataframe(
            headers=["feature", "direction", "contribution"],
            datatype=["str", "str", "number"],
            label="Найвпливовіші слова та n-грамі",
            interactive=False,
            wrap=True,
            elem_classes=["soft-card"],
        )

        gr.Markdown(
            """
            ### Як інтерпретувати результат
            - **SPAM / HAM** — підсумковий клас, який обрала модель.
            - **Індикатор впевненості** — наочне порівняння сили двох класів.
            - **Найвпливовіші слова та n-грамі** — ознаки, які найбільше вплинули на рішення Logistic Regression.
              Позначка `spam` означає, що ознака штовхає модель у бік спаму, `ham` — у бік звичайного повідомлення.
            """,
            elem_classes=["soft-card"],
        )

    analyze_btn.click(
        fn=predict_message,
        inputs=[text_input],
        outputs=[result_html, confidence_html, explanation_table],
    )

    text_input.submit(
        fn=predict_message,
        inputs=[text_input],
        outputs=[result_html, confidence_html, explanation_table],
    )

    clear_btn.click(
        lambda: (
            "",
            """
            <div class='result-card'>
                <div class='verdict'>Результат з’явиться тут після аналізу.</div>
            </div>
            """,
            """
            <div class="confidence-card">
                <div class="confidence-title">Індикатор впевненості</div>
                <div class="confidence-caption">Після аналізу тут з’явиться розподіл ймовірностей моделі.</div>
            </div>
            """,
            pd.DataFrame(columns=["feature", "direction", "contribution"]),
        ),
        outputs=[text_input, result_html, confidence_html, explanation_table],
    )

# if __name__ == "__main__":
#     demo.launch()
    
def run():
    demo.launch()