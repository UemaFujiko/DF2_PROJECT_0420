import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from openai import OpenAI
from datetime import datetime
import os

# --- OpenAI 初期化 ---
client = OpenAI(api_key=os.environ.get("API_KEY"))

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="財務データ統合アプリ", layout="wide")

# --- サイドバー ---
st.sidebar.title("📂 ナビゲーション")
page = st.sidebar.radio(
    "機能を選択してください",
    ("📤 Supabaseアップロード", "📈 Supabase可視化", "🧮 ローカルCSVダッシュボード")
)

# --- Supabase 接続 ---
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("環境変数 SUPABASE_URL または SUPABASE_KEY が設定されていません。")
    return create_client(url, key)

supabase = init_supabase()

# --- 定数 ---
table_name = "monthly_pl"
DATE_COLUMN = "date"
SALES_COLUMN = "sales"

# --- GPT 売上分析 ---
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT APIエラー: {e}"
# --- Supabaseデータ取得 ---
@st.cache_data(ttl=600)
def fetch_supabase_data():
    try:
        response = supabase.table(table_name).select("*").order(DATE_COLUMN, desc=False).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"❌ データ取得エラー: {e}")
        return pd.DataFrame()

# --- データ整形 ---
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
        return df[df["year"].isin([latest_year, latest_year - 1, latest_year - 2])].sort_values(by=DATE_COLUMN)
    except Exception as e:
        st.error(f"❌ データ処理エラー: {e}")
        return pd.DataFrame()

# === 📤 Supabaseアップロード ===
if page == "📤 Supabaseアップロード":
    st.title("📤 財務データのSupabaseアップロード")

    uploaded_file = st.file_uploader("📤 CSVファイルをアップロードしてください", type=["csv"])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.success("✅ CSVを読み込みました。以下のデータを Supabase テーブル `monthly_pl` に保存します。")
        st.dataframe(df_upload)
        st.write("カラム構造:", df_upload.dtypes)

        if st.button("📥 Supabaseにアップロード"):
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

                fetch_supabase_data.clear()
            except Exception as e:
                st.error("❌ Supabase保存中のエラー:")
                st.code(str(e), language="json")
# === 📈 Supabase可視化 ===
elif page == "📈 Supabase可視化":
    st.title("📈 Supabase上の売上データ可視化")

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
# === 🧮 ローカルCSVダッシュボード ===
elif page == "🧮 ローカルCSVダッシュボード":
    st.title("🧮 月次 財務データダッシュボード（ローカルCSV）")

    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください（Supabase不要）", type="csv", key="local_csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df["date"] = pd.to_datetime(df["date"])
        df["月"] = df["date"].dt.strftime("%Y-%m")
        df["変動費"] = df.get("outsourcing_costs", 0) + df.get("commissions_fees", 0)

        固定費列 = [
            "executive_compensation", "salaries", "bonuses", "rent_payments",
            "welfare_expenses", "employee_welfare", "supplies",
            "utilities", "communication", "transportation", "communication_expenses",
            "advertising", "entertainment", "training_expenses", "miscellaneous_expenses"
        ]
        存在する列 = [col for col in 固定費列 if col in df.columns]
        df["固定費"] = df[存在する列].sum(axis=1)

        df["粗利益"] = df.get("gross_profit", df["sales"] - df.get("cost_of_sales", df["変動費"] + df["固定費"]))
        df["経常利益"] = df.get("ordinary_profit", 0)

        df_summary = df[["月", "sales", "変動費", "固定費", "粗利益", "経常利益"]].copy()
        df_summary = df_summary.rename(columns={"sales": "売上高"})

        st.subheader("📅 月次財務項目一覧")
        st.dataframe(df_summary)

        # 人件費と指標
        personnel_columns = ["executive_compensation", "salaries", "bonuses", "welfare_expenses", "employee_welfare"]
        existing_personnel = [col for col in personnel_columns if col in df.columns]
        df["人件費"] = df[existing_personnel].sum(axis=1)

        df_summary["人件費"] = df["人件費"]
        df_summary["限界利益率"] = df_summary["粗利益"] / df_summary["売上高"]
        df_summary["売上高経常利益率"] = df_summary["経常利益"] / df_summary["売上高"]
        df_summary["損益分岐点比率"] = df_summary["固定費"] / df_summary["粗利益"]
        df_summary["生産性"] = df_summary["粗利益"] / df_summary["固定費"]
        df_summary["労働生産性"] = df_summary["粗利益"] / df_summary["人件費"]
        df_summary["労働分配率"] = df_summary["人件費"] / df_summary["粗利益"]

        st.subheader("📐 経営指標の一覧")
        st.dataframe(df_summary[[ "月", "限界利益率", "売上高経常利益率", "損益分岐点比率", "生産性", "労働生産性", "労働分配率" ]])

        # --- 利益構造ツリーマップ ---
        st.subheader("🧩 月次の利益構造（PLツリーマップ）")
        selected_month = st.selectbox("表示する月を選んでください", df_summary["月"].unique(), index=len(df_summary)-1)
        df_selected = df_summary[df_summary["月"] == selected_month].iloc[0]

        labels = ["売上高", "変動費", "固定費", "粗利益", "経常利益"]
        values = [df_selected["売上高"], df_selected["変動費"], df_selected["固定費"], df_selected["粗利益"], df_selected["経常利益"]]
        parents = ["", "売上高", "売上高", "売上高", "粗利益"]

        fig = px.treemap(names=labels, values=values, parents=parents, title=f"{selected_month} の利益構造")
        fig.update_layout(margin=dict(t=40, l=25, r=25, b=25))
        st.plotly_chart(fig, use_container_width=True)

        # --- 前月・前年同月比較グラフ ---
        st.subheader("📊 利益構造の比較（図表3風）")
        all_months = df_summary["月"].tolist()
        current_index = all_months.index(selected_month)
        prev_month = all_months[current_index - 1] if current_index > 0 else None
        prev_year = f"{int(selected_month[:4]) - 1}-{selected_month[5:]}"
        prev_year = prev_year if prev_year in all_months else None

        def show_comparison(before_month, after_month, label):
            if not before_month or not after_month:
                st.info(f"{label}比較できるデータがありません")
                return
            before = df_summary[df_summary["月"] == before_month].iloc[0]
            after = df_summary[df_summary["月"] == after_month].iloc[0]
            delta = after["経常利益"] - before["経常利益"]
            ratio = (delta / before["経常利益"]) * 100 if before["経常利益"] != 0 else 0

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=[before[label] / 1_000_000 for label in labels],
                name=before_month,
                marker_color="lightgray"
            ))
            fig.add_trace(go.Bar(
                x=labels,
                y=[after[label] / 1_000_000 for label in labels],
                name=after_month,
                marker_color="teal"
            ))
            fig.update_layout(
                title=f"{label}比較: {before_month} → {after_month}（経常利益差: {delta/1_000_000:.1f}百万円, {ratio:.1f}%）",
                barmode="group",
                yaxis_title="金額（百万円）"
            )
            st.plotly_chart(fig, use_container_width=True)

        show_comparison(prev_month, selected_month, "前月")
        show_comparison(prev_year, selected_month, "前年同月")
