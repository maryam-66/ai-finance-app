# نسخه نهایی با پشتیبانی از فارسی واقعی در PDF (fpdf2 + arabic_reshaper + bidi)

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import streamlit as st
import requests
from fpdf import FPDF
from io import BytesIO
import os
import arabic_reshaper
from bidi.algorithm import get_display

# بارگذاری واژگان VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# تنظیم صفحه
st.set_page_config(page_title="تحلیل بازار مالی", layout="centered")
st.title("📊 تحلیل مالی با هوش مصنوعی و تحلیل احساسات")

# ورودی‌ها
symbols = st.multiselect("نمادها:", ["AAPL", "GOOGL", "MSFT", "BTC", "ETH", "GOLD", "SILVER"], default=["AAPL"])
start_date = st.date_input("تاریخ شروع", pd.to_datetime("2021-01-01"))
end_date = st.date_input("تاریخ پایان", pd.to_datetime("2024-12-31"))

# تحلیل احساسات
st.subheader("🧠 تحلیل اخبار و احساسات")
newsapi_key = "1fbbb3b298474644b2187f4a534484d4"
run_analysis = st.button("تحلیل اخبار و تولید گزارش")

if run_analysis:
    all_data = []
    analyzer = SentimentIntensityAnalyzer()
    for symbol in symbols:
        url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&pageSize=5&apiKey={newsapi_key}"
        r = requests.get(url)
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            for article in articles:
                text = f"{article.get('title', '')}. {article.get('description', '')}"
                score = analyzer.polarity_scores(text)
                all_data.append({
                    "symbol": symbol,
                    "date": article.get("publishedAt", "")[:10],
                    "title": article.get("title", ""),
                    "compound": score['compound'],
                    "pos": score['pos'],
                    "neg": score['neg'],
                    "neu": score['neu']
                })
        else:
            st.warning(f"❌ خطا برای {symbol}: {r.status_code}")

    if all_data:
        df = pd.DataFrame(all_data)
        st.success("✅ تحلیل انجام شد")
        st.dataframe(df)

        # خلاصه آماری کلی احساسات
        st.subheader("📊 خلاصه کلی احساسات")
        avg_scores = df[['pos', 'neg', 'neu', 'compound']].mean()
        st.write("میانگین احساسات:")
        st.dataframe(avg_scores.to_frame(name='میانگین'))

        # 📈 نمودار میله‌ای احساسات کلی
        st.subheader("📊 نمودار میله‌ای احساسات کلی")
        fig_bar, ax_bar = plt.subplots()
        avg_scores[['pos', 'neg', 'neu']].plot(kind='bar', ax=ax_bar, color=['green', 'red', 'gray'])
        ax_bar.set_ylabel("میانگین نمره")
        ax_bar.set_title("میانگین احساسات مثبت، منفی، خنثی")
        st.pyplot(fig_bar)

        # خروجی اکسل
        excel_buf = BytesIO()
        df.to_excel(excel_buf, index=False)
        st.download_button("⬇️ دانلود Excel", data=excel_buf.getvalue(), file_name="sentiment.xlsx")

        # PDF با راست‌چین و پشتیبانی کامل از فارسی
        from fpdf import FPDF

        class PDF(FPDF):
            pass

        pdf = PDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", fname="DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", fname="DejaVuSans-Bold.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
        title = get_display(arabic_reshaper.reshape("تحلیل احساسات اخبار مالی"))
        pdf.cell(200, 10, txt=title, ln=True, align="C")

        for symbol, group in df.groupby("symbol"):
            pdf.set_font("DejaVu", style="B", size=12)
            pdf.ln(10)
            sym_txt = get_display(arabic_reshaper.reshape(f"نماد: {symbol}"))
            pdf.cell(0, 10, txt=sym_txt, ln=True)
            pdf.set_font("DejaVu", size=11)
            for _, row in group.iterrows():
                fa_title = get_display(arabic_reshaper.reshape(row['title']))
                date = row['date']
                compound = row['compound']
                sentiment_txt = get_display(arabic_reshaper.reshape(f"احساس کلی: {compound:.2f}"))
                pdf.multi_cell(0, 10, txt=f"{date}\n{fa_title}\n{sentiment_txt}\n")

            # نمودار احساسات
            fig, ax = plt.subplots()
            sentiments = group[['pos', 'neg', 'neu']].mean()
            ax.pie(sentiments, labels=['مثبت', 'منفی', 'خنثی'], autopct='%1.1f%%')
            ax.axis('equal')
            chart_buf = BytesIO()
            plt.savefig(chart_buf, format='png')
            plt.close(fig)
            chart_path = f"{symbol}_chart.png"
            with open(chart_path, "wb") as f:
                f.write(chart_buf.getvalue())
            pdf.image(chart_path, w=100)
            os.remove(chart_path)

        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            pdf_bytes = f.read()
        st.download_button("⬇️ دانلود PDF", data=pdf_bytes, file_name="sentiment_report.pdf")
    else:
        st.error("هیچ خبری یافت نشد.")
