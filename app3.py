import streamlit as st
import pandas as pd
from supabase import create_client
import plotly.express as px
import openai
from datetime import datetime

# --- ページ設定 ---
st.set_page_config(page_title="財務データ統合アプリ", layout="wide")
st.title("📊 財務データアップロード & 可視化アプリ")

# --- Supabase 接続 ---
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase()
openai.api_key = st.secrets["openai"]["api_key"]

# --- 固定テーブル名（Supabaseのpublic.monthly_pl）---
table_name = "monthly_pl"
DATE_COLUMN = "date"
SALES_COLUMN = "sales"

# --- GPT による売上分析関数 ---
def generate_sales_advice(df: pd.DataFrame, sales_col: str):
    if df.empty:
        return "データが存在しないため、分析できません。"
    
    csv_data = df[['year_month', sales_col]].to_csv(index=False)
    prompt = f"""
あなたは財務分析の専門家です。
以下のCSVデータは、企業の3年分の月次売上データです。
売上の傾向、注意点、改善アドバイスを500文字以内で日本語で要約してください。

データ:
{csv_data}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT APIエラー: {e}"

# --- Supabaseからデータ取得 ---
@st.cache_data(ttl=600)
def fetch_supabase_data():
    try:
        response = supabase.table(table_name).select("*").order(DATE_COLUMN, desc=False).execute()
        data = response.data
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ データ取得エラー: {e}")
        return pd.DataFrame()

# --- データ処理 ---
def process_data(df):
    if df.empty:
        return pd.DataFrame()
    try:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        df[SALES_COLUMN] = pd.to_numeric(df[SALES_COLUMN], errors='coerce')
        df.dropna(subset=[DATE_COLUMN, SALES_COLUMN], inplace=True)
        df["year"] = df[DATE_COLUMN].dt.year
        df["year_month"] = df[DATE_COLUMN].dt.strftime('%Y-%m')
        latest_year = df["year"].max()
        df_filtered = df[df["year"].isin([latest_year, latest_year - 1, latest_year - 2])].copy()
        df_filtered.sort_values(by=DATE_COLUMN, inplace=True)
        return df_filtered
    except Exception as e:
        st.error(f"❌ データ処理エラー: {e}")
        return pd.DataFrame()

# --- CSV アップロード UI ---
uploaded_file = st.file_uploader("📤 CSVファイルをアップロードしてください", type=["csv"])
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    st.success("✅ CSVを読み込みました。以下のデータを Supabase テーブル `monthly_pl` に保存します。")
    st.dataframe(df_upload)
    st.write("カラム構造:", df_upload.dtypes)

    if st.button("📥 Supabaseに保存する"):
        try:
            data_list = df_upload.to_dict(orient="records")
            success_count = 0
            failed_rows = []

            for row in data_list:
                response = supabase.table(table_name).insert(row).execute()
                if hasattr(response, 'data') and response.data:
                    success_count += 1
                else:
                    failed_rows.append(row)

            if success_count == len(data_list):
                st.success(f"🎉 {success_count} 件すべてのデータを Supabase に保存しました！")
            else:
                st.warning(f"成功: {success_count} 件 / 失敗: {len(failed_rows)} 件")
                with st.expander("失敗データの詳細"):
                    st.write(failed_rows)

            fetch_supabase_data.clear()  # キャッシュ削除
        except Exception as e:
            st.error("❌ Supabase保存中のエラー:")
            st.code(str(e), language="json")

# --- Supabaseデータの読み込み & 可視化 ---
df_supabase = fetch_supabase_data()
df_processed = process_data(df_supabase)

if not df_processed.empty:
    st.subheader("📊 売上推移グラフ")
    fig = px.bar(
        df_processed,
        x='year_month',
        y=SALES_COLUMN,
        title="月次売上高の推移",
        labels={'year_month': '年月', SALES_COLUMN: '売上高 (円)'},
        color='year',
        hover_data={DATE_COLUMN: '|%Y年%m月%d日', SALES_COLUMN: ':,.0f 円'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 表形式で表示")
    st.dataframe(
        df_processed[[DATE_COLUMN, "year_month", SALES_COLUMN]]
        .rename(columns={DATE_COLUMN: "日付", "year_month": "年月", SALES_COLUMN: "売上高"})
        .sort_values("日付", ascending=False)
        .style.format({"売上高": "{:,.0f} 円"}),
        use_container_width=True
    )

    with st.expander("💡 GPTによる売上分析アドバイス", expanded=True):
        with st.spinner("ChatGPTが分析中..."):
            advice = generate_sales_advice(df_processed, SALES_COLUMN)
            st.success("✅ 分析コメント:")
            st.markdown(advice)
else:
    st.info("📭 Supabase テーブル `monthly_pl` に保存されたデータがまだありません。")
