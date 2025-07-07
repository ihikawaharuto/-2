
# 必要なパッケージ: streamlit numpy pandas transformers accelerate sentencepiece fugashi unidic-lite ipadic torch torchvision torchaudio annoy
import streamlit as st
import torch
import numpy as np
import time
import base64
import pickle
import os
import logging
from transformers import BertJapaneseTokenizer, BertModel
from annoy import AnnoyIndex

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 定数 ---
class Config:
    VECTOR_CACHE_PATH = "vectors_cache.npy"
    TEXT_SOURCES_CACHE_PATH = "text_sources_cache.pkl"
    ANNOY_INDEX_PATH = "annoy_index.ann"
    SAFE_FILE = "safe.txt"
    OUT_FILE = "out.txt"
    F_DIM = 768
    VIDEO_PATH = "fire2.mp4"
    CSS_PATH = "style.css"
# --- 動画再生用関数 ---
@st.cache_data
def get_video_base64(video_path):
    """動画ファイルをBase64エンコードして返す"""
    if not os.path.exists(video_path):
        st.warning(f"動画ファイルが見つかりません: {video_path}")
        return None
    with open(video_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

def get_video_html(video_path="fire2.mp4", width=150):
    """動画を再生するためのHTML文字列を生成する"""
    video_base64 = get_video_base64(video_path)
    if video_base64:
        return f"""
            <video autoplay muted loop width="{width}" style="border-radius: 10px; margin-top: 20px; mix-blend-mode: add;">            
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>"""
    return ""


# --- モデルとトークナイザの読み込み (キャッシュ付き) ---
@st.cache_resource
def load_model_and_tokenizer():
    """BERTモデルとトークナイザーをロードしてキャッシュします。"""
    try:
        logging.info("Loading BERT model and tokenizer...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        model.to(device)
        model.eval()
        logging.info("Model and tokenizer loaded successfully.")
        return tokenizer, model, device
    except ImportError as e:
        logging.error(f"Import error during model loading: {e}")
        st.error(f"モデルの読み込みに必要なライブラリが見つかりません: {e}")
        st.error("pip install -r requirements.txt などで、必要なライブラリ (transformers, torch, fugashi, unidic-liteなど) が正しくインストールされているか確認してください。")
        st.stop()
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading: {e}")
        st.error(f"モデルの読み込み中に予期せぬエラーが発生しました: {e}")
        st.stop()

tokenizer, model, device = load_model_and_tokenizer()

# --- エンゲージメント計算クラス ---
class EngagementCalculator:
    def calculate(self, followers, buzz_score, post_type):
        """
        エンゲージメント（インプレッション、リポスト、いいね）を計算します。
        :param followers: フォロワー数
        :param buzz_score: バズスコア (0-100)
        :param post_type: 'SAFE' または 'OUT'
        :return: (インプレッション数, リポスト数, いいね数)
        """
        p = 1 if post_type == 'OUT' else 0
        i = int(followers * 0.3 + followers**0.1 * (1 + 210970 * (buzz_score / 100)**3.2 * (1 + 0.5 * (buzz_score / 100)**5 * p)))
        r = int(i * 0.01 * (1 + 2 * (buzz_score / 100)**2) * (1 + p))
        l = int(i * 0.03 * (1 + 0.5 * (buzz_score / 100)**0.7) * (1 + 0.1 * p))
        return i, r, l

# --- メインの分類器クラス ---
class TextClassifier:
    def __init__(self, config):
        self.config = config
        self.vec = []
        self.text_sources = []
        self.index = AnnoyIndex(self.config.F_DIM, 'angular')
        self._ensure_data_files_exist()

    def _ensure_data_files_exist(self):
        """データファイルが存在しない場合に空のファイルを作成します。"""
        for fname in [self.config.SAFE_FILE, self.config.OUT_FILE]:
            if not os.path.exists(fname):
                logging.warning(f"Data file '{fname}' not found. Creating an empty file.")
                open(fname, 'w', encoding='utf-8').close()

    def get_vector(self, text):
        """テキストをベクトルに変換します。"""
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text, truncation=True, return_tensors='pt').to(device)
            outputs = model(**inputs)
            return outputs.pooler_output.detach().cpu().numpy()[0]

    def build_index(self):
        """Annoyインデックスを構築します。"""
        logging.info("Building Annoy index...")
        self.index = AnnoyIndex(self.config.F_DIM, 'angular')
        for i, v in enumerate(self.vec):
            self.index.add_item(i, v)
        self.index.build(10) # 10 is the number of trees
        logging.info(f"Annoy index built with {len(self.vec)} items.")


    def save_caches(self):
        """ベクトルとテキストソースのキャッシュ、Annoyインデックスを保存します。"""
        logging.info("Saving caches and index...")
        np.save(self.config.VECTOR_CACHE_PATH, np.array(self.vec))
        with open(self.config.TEXT_SOURCES_CACHE_PATH, 'wb') as f:
            pickle.dump(self.text_sources, f)
        self.index.save(self.config.ANNOY_INDEX_PATH)
        logging.info("Caches and index saved.")

    def load_from_cache(self):
        """キャッシュからデータをロードします。"""
        cache_files = [self.config.VECTOR_CACHE_PATH, self.config.TEXT_SOURCES_CACHE_PATH, self.config.ANNOY_INDEX_PATH]
        if all(os.path.exists(p) for p in cache_files):
            try:
                logging.info("Loading data from cache...")
                self.vec = np.load(self.config.VECTOR_CACHE_PATH).tolist()
                with open(self.config.TEXT_SOURCES_CACHE_PATH, 'rb') as f:
                    self.text_sources = pickle.load(f)
                self.index.load(self.config.ANNOY_INDEX_PATH)
                logging.info("Data loaded from cache successfully.")
                return True
            except (IOError, pickle.UnpicklingError, ValueError) as e:
                logging.error(f"Failed to load from cache: {e}. Rebuilding from source.")
                return False
        return False

    def load_from_source(self):
        """ソーステキストファイルからデータをロードし、インデックスを構築します。"""
        logging.info("Loading data from source text files...")
        self.vec, self.text_sources = [], []
        try:
            with open(self.config.SAFE_FILE, "r", encoding="utf-8") as f:
                texts_safe = [line.strip() for line in f if line.strip()]
            with open(self.config.OUT_FILE, "r", encoding="utf-8") as f:
                texts_out = [line.strip() for line in f if line.strip()]
        except FileNotFoundError as e:
            logging.error(f"Source file not found: {e}")
            st.error(f"ソースファイルが見つかりません: {e.filename}。ファイルが存在するか確認してください。")
            st.stop()

        # Process texts and create vectors
        all_texts = [(text, "safe") for text in texts_safe] + [(text, "out") for text in texts_out]
        if not all_texts:
            logging.warning("Source files are empty. Index will be empty.")
            return

        with st.spinner("テキストをベクトルに変換中..."):
            for text, label in all_texts:
                self.vec.append(self.get_vector(text))
                self.text_sources.append((text, label))        
        
        self.build_index()
        self.save_caches()

    def add_text(self, text, label):
        """
        新しいテキストを追加し、インデックスを更新します。
        注意: この実装ではテキストを追加するたびにインデックス全体を再構築するため、
              データ量が多い場合はパフォーマンスが低下する可能性があります。
        """
        file_path = self.config.SAFE_FILE if label == "safe" else self.config.OUT_FILE
        try:
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(f"{text}\n")
        except IOError as e:
            logging.error(f"Failed to write to {file_path}: {e}")
            st.error(f"ファイルへの書き込みに失敗しました: {file_path}")
            return None        
        self.vec.append(self.get_vector(text))
        self.text_sources.append((text, label))
        
        self.build_index()
        self.save_caches()
        
        color = "#1e90ff" if label == "safe" else "#ff4500"
        return f"<span style='color: {color};'>{label.upper()} に追加されました</span>"

    def judge(self, text):
        """入力されたテキストを判定します。"""
        if not self.vec or not self.index.get_n_items():
            return "<span style='color: black;'>判定不可 (学習データがありません)</span>"
        try:
            vec_x = self.get_vector(text)
            indices, distances = self.index.get_nns_by_vector(vec_x, 1, include_distances=True)
            
            if not indices:
                return "<span style='color: black;'>判定不可 (類似データが見つかりません)</span>"
            # Annoyのangular距離は sqrt(2(1-cos(angle))) なので、cos(angle) = 1 - (dist^2 / 2)
            sim_score = 1 - (distances[0]**2 / 2)
            
            if sim_score < 0.75:
                return f"<span style='color: black;'>判定保留 (類似度が低いため: {sim_score:.2f})</span>"

            label = self.text_sources[indices[0]][1]
            color = "#1e90ff" if label == "safe" else "#ff4500"
            label_upper = label.upper()
            
            return f"<span style='color: {color}; font-weight: bold;'>{label_upper} (類似度: {sim_score:.2f})</span>"
        except Exception as e:
            logging.error(f"Judgement error: {e}", exc_info=True)
            return f"<span style='color: black;'>判定中にエラーが発生しました</span>"
# --- CSSファイルを読み込む関数 ---
def load_css(file_name):
    """外部CSSファイルを読み込んで適用する"""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"{file_name} が見つかりません。デザインが適用されません。")

# --- UIヘルパー関数 ---
def handle_add_text(classifier, text, label):
    """テキスト追加のロジックを処理する"""
    if text.strip():
        with st.spinner(f"{label.upper()} に追加中..."):
            msg = classifier.add_text(text, label)
        if msg:
            icon = "✅" if label == "safe" else "🔥"
            st.markdown(f"{icon} {msg}", unsafe_allow_html=True)
            st.session_state.judgement_result = None # 結果表示をリセット
    else:
        st.warning("文章を入力してください。")
# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="リアルタイム炎上判定システム", layout="centered")

    # --- セッション状態の初期化 ---
    if 'judgement_result' not in st.session_state:
        st.session_state.judgement_result = None
    if 'judgement_text' not in st.session_state:
        st.session_state.judgement_text = ""

    # --- 外部CSSファイルの読み込み ---
    load_css(Config.CSS_PATH)

    st.markdown("<h1>炎上判定システム</h1>", unsafe_allow_html=True)

    # --- アプリケーションの初期化 ---
    @st.cache_resource
    def get_classifier():
        classifier = TextClassifier(Config())
        if not classifier.load_from_cache():
            classifier.load_from_source()
        return classifier

    try:
        classifier = get_classifier()
    except Exception as e:
        logging.critical(f"Classifier initialization failed: {e}", exc_info=True)
        st.error("アプリケーションの初期化に失敗しました。ログを確認してください。")
        st.stop()

    calculator = EngagementCalculator()

    # --- タブの作成 ---
    tab1, tab2 = st.tabs(["炎上判定", "エンゲージメント計算"])

    # --- 炎上判定タブ ---
    with tab1:
        input_text = st.text_area("文章を入力してください", height=150, key="text_judge")

        if st.button("判定", key="judge_button"):
            if input_text.strip():
                with st.spinner("判定中..."):
                    result = classifier.judge(input_text)
                st.session_state.judgement_text = result
                if "OUT" in result:
                    st.session_state.judgement_result = "OUT"
                elif "SAFE" in result:
                    st.session_state.judgement_result = "SAFE"
                else:
                    st.session_state.judgement_result = "OTHER"
            else:
                st.warning("文章を入力してください。")
                st.session_state.judgement_result = None

        # --- 判定結果の表示 ---
        if st.session_state.get('judgement_result'):
            st.markdown("---")
            st.markdown("### 判定結果")
            if st.session_state.judgement_result == "OUT":
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(st.session_state.judgement_text, unsafe_allow_html=True)
                with col2:
                    st.markdown(get_video_html(), unsafe_allow_html=True)
            else:
                st.markdown(st.session_state.judgement_text, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("判定結果を学習データに追加")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("SAFE に追加", key="safe_add"):
                handle_add_text(classifier, input_text, "safe")
        with col2:
            if st.button("OUT に追加", key="out_add"):
                handle_add_text(classifier, input_text, "out")

    # --- エンゲージメント計算タブ ---
    with tab2:
        if 'judgement_result' in st.session_state:
            st.session_state.judgement_result = None # タブを切り替えたら結果表示をリセット
        st.header("エンゲージメント予測")
        followers = st.number_input("フォロワー数", min_value=0, value=10000)
        buzz_score = st.slider("バズスコア", min_value=0, max_value=100, value=50)
        post_type = st.radio("投稿タイプ", ('SAFE', 'OUT'), horizontal=True)

        if st.button("計算", key="calc_button"):
            impressions, reposts, likes = calculator.calculate(followers, buzz_score, post_type)
            st.markdown("## 計算結果")
            st.markdown(f"""
            <div class="result-box">
                <p><strong>インプレッション数:</strong> {impressions:,}</p>
                <p><strong>リポスト数:</strong> {reposts:,}</p>
                <p><strong>いいね数:</strong> {likes:,}</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
''