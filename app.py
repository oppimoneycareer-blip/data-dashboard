import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from itertools import combinations

# ---------------------------------------------------------
# 設定とユーティリティ関数
# ---------------------------------------------------------
st.set_page_config(
    page_title="自動データ分析ダッシュボード",
    page_icon="📊",
    layout="wide"
)

# 日本語フォント対応（Plotlyはデフォルトで対応している場合が多いが、念のため設定）
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_sample_data():
    """サンプルデータ（売上データ）を生成する関数"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    categories = ["Electronics", "Clothing", "Home", "Toys"]
    regions = ["North", "South", "East", "West"]
    
    data = {
        "Date": dates,
        "Category": np.random.choice(categories, 100),
        "Region": np.random.choice(regions, 100),
        "Sales": np.random.randint(1000, 50000, 100),
        "Profit": np.random.randint(-5000, 15000, 100),
        "Quantity": np.random.randint(1, 50, 100),
        "Customer_Satisfaction": np.random.uniform(1.0, 5.0, 100)
    }
    return pd.DataFrame(data)

def analyze_numeric_column(df, col):
    """数値列の統計的考察を生成"""
    desc = df[col].describe()
    return f"""
    - **最大値**: {desc['max']:,.2f}
    - **最小値**: {desc['min']:,.2f}
    - **平均値**: {desc['mean']:,.2f}
    - **中央値**: {desc['50%']:,.2f}
    - **標準偏差**: {desc['std']:,.2f} (データのばらつき)
    """

def analyze_categorical_column(df, col):
    """カテゴリ列の統計的考察を生成"""
    counts = df[col].value_counts()
    top_cat = counts.index[0]
    top_val = counts.iloc[0]
    ratio = (top_val / len(df)) * 100
    return f"""
    - **ユニーク数**: {len(counts)} 種類
    - **最頻値**: {top_cat} ({top_val} レコード)
    - **構成比**: 全体の {ratio:.1f}% を占めています。
    """

def analyze_correlation(df, col1, col2):
    """2変数の相関考察を生成"""
    corr = df[col1].corr(df[col2])
    evaluation = "相関なし"
    if abs(corr) > 0.7: evaluation = "強い相関あり"
    elif abs(corr) > 0.4: evaluation = "中程度の相関あり"
    elif abs(corr) > 0.2: evaluation = "弱い相関あり"
    
    return f"""
    - **相関係数**: {corr:.4f}
    - **判定**: {evaluation}
    - ({col1}が増えると、{col2}は{'増える' if corr > 0 else '減る'}傾向にあります)
    """

# ---------------------------------------------------------
# サイドバー：データアップロード
# ---------------------------------------------------------
st.sidebar.header("📂 データ入力")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("ファイルを読み込みました！")
    except Exception as e:
        st.sidebar.error(f"エラーが発生しました: {e}")
        df = generate_sample_data()
else:
    st.sidebar.info("ファイルが未選択のため、サンプルデータを表示します。")
    df = generate_sample_data()

# データ型変換（日付らしきものを変換）
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            pass

# ---------------------------------------------------------
# メイン画面：データ概要
# ---------------------------------------------------------
st.title("📊 自動データ分析ダッシュボード")

st.header("1. データ概要")
col1, col2, col3 = st.columns(3)
col1.metric("行数", df.shape[0])
col2.metric("列数", df.shape[1])
col3.metric("欠損値の合計", df.isnull().sum().sum())

with st.expander("データフレームの中身を確認（最初の5行）", expanded=True):
    st.dataframe(df.head())

with st.expander("基本統計量（Describe）"):
    st.dataframe(df.describe())

# 列の分類
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

st.divider()

# ---------------------------------------------------------
# 自動可視化セクション
# ---------------------------------------------------------
st.header("2. 自動生成グラフと考察")

chart_count = 0

# --- A. 相関ヒートマップ (数値列が2つ以上ある場合) ---
if len(numeric_cols) >= 2:
    st.subheader(f"Graph {chart_count + 1}: 数値項目の相関ヒートマップ")
    corr_matrix = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=True, 
        color_continuous_scale='RdBu_r', 
        aspect="auto",
        range_color=[-1, 1]
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 考察
    max_corr = corr_matrix.replace(1.0, 0).abs().max().max()
    st.info(f"💡 **考察**: 最も強い相関の絶対値は **{max_corr:.2f}** です。色が濃い部分（赤または青）は項目間に関連性が強いことを示します。")
    chart_count += 1

# --- B. 数値変数の分布 (ヒストグラム & 箱ひげ図) ---
st.subheader("数値データの分布確認")
cols = st.columns(2)
for i, col in enumerate(numeric_cols[:6]): # 表示数制限（最大6つ）
    with cols[i % 2]:
        st.markdown(f"#### {col} の分布")
        
        # ヒストグラム
        fig_hist = px.histogram(df, x=col, marginal="box", title=f"Graph {chart_count + 1}: {col} Histogram")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 考察エリア
        insight = analyze_numeric_column(df, col)
        st.success(f"📈 **考察 ({col})**:\n{insight}")
        chart_count += 1

# --- C. カテゴリデータのカウント (棒グラフ) ---
if categorical_cols:
    st.subheader("カテゴリデータの構成比")
    cols_cat = st.columns(2)
    for i, col in enumerate(categorical_cols[:4]): # 表示数制限
        with cols_cat[i % 2]:
            st.markdown(f"#### {col} の件数")
            
            # 棒グラフ
            counts_df = df[col].value_counts().reset_index()
            counts_df.columns = [col, 'Count']
            fig_bar = px.bar(counts_df, x=col, y='Count', color='Count', title=f"Graph {chart_count + 1}: {col} Bar Chart")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 考察エリア
            insight = analyze_categorical_column(df, col)
            st.warning(f"📊 **考察 ({col})**:\n{insight}")
            chart_count += 1

# --- D. 2変数の関係性 (散布図) ---
if len(numeric_cols) >= 2:
    st.subheader("2変数の関係性（散布図）")
    # 相関が高い、または重要そうな組み合わせをいくつかピックアップ
    pairs = list(combinations(numeric_cols, 2))
    
    # 最大4つの組み合わせを表示
    cols_scatter = st.columns(2)
    for i, (col1, col2) in enumerate(pairs[:4]):
        with cols_scatter[i % 2]:
            st.markdown(f"#### {col1} vs {col2}")
            
            # カテゴリ変数があれば色分けに使用
            color_col = categorical_cols[0] if categorical_cols else None
            
            fig_scatter = px.scatter(
                df, x=col1, y=col2, 
                color=color_col, 
                trendline="ols", # 回帰直線
                title=f"Graph {chart_count + 1}: Scatter Plot"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 考察エリア
            insight = analyze_correlation(df, col1, col2)
            st.info(f"🔍 **考察 ({col1} vs {col2})**:\n{insight}")
            chart_count += 1

# --- E. 時系列推移 (日付データがある場合) ---
if datetime_cols and numeric_cols:
    st.subheader("時系列推移")
    date_col = datetime_cols[0] # 最初の日付列を使用
    
    # 日付でソート
    df_sorted = df.sort_values(by=date_col)
    
    cols_time = st.columns(2)
    for i, num_col in enumerate(numeric_cols[:2]): # 最初の2つの数値列を表示
        with cols_time[i % 2]:
            st.markdown(f"#### {num_col} の推移")
            
            fig_line = px.line(df_sorted, x=date_col, y=num_col, title=f"Graph {chart_count + 1}: Time Series of {num_col}")
            st.plotly_chart(fig_line, use_container_width=True)
            
            # 簡単なトレンド考察
            start_val = df_sorted[num_col].iloc[0]
            end_val = df_sorted[num_col].iloc[-1]
            diff = end_val - start_val
            trend = "増加" if diff > 0 else "減少"
            
            st.success(f"""
            📅 **考察**:
            - 期間中の変化量: {diff:,.2f}
            - 全体的な傾向: **{trend}** 傾向が見られます（始点と終点の比較）。
            """)
            chart_count += 1

# --- F. 箱ひげ図によるカテゴリ別分布 ---
if categorical_cols and numeric_cols:
    st.subheader("カテゴリ別の数値分布（箱ひげ図）")
    cat_col = categorical_cols[0]
    cols_box = st.columns(2)
    
    for i, num_col in enumerate(numeric_cols[:2]):
        with cols_box[i % 2]:
            st.markdown(f"#### {cat_col} 別の {num_col}")
            fig_box = px.box(df, x=cat_col, y=num_col, color=cat_col, title=f"Graph {chart_count + 1}: Box Plot by {cat_col}")
            st.plotly_chart(fig_box, use_container_width=True)
            
            # グループごとの平均値計算
            means = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            top_group = means.index[0]
            
            st.info(f"""
            📦 **考察**:
            - 平均値が最も高いグループ: **{top_group}** ({means[top_group]:,.2f})
            - カテゴリによる数値の違いを確認してください。
            """)
            chart_count += 1


st.divider()
st.write(f"合計 {chart_count} 個のグラフを生成しました。")
