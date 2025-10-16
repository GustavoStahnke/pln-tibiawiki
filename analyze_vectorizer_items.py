"""
Analisa os resultados da vetorização TF-IDF gerada pelo vectorize_items.py.

Entradas:
  - data/items_info_preprocessed.csv
  - data/data_vectorizer/tfidf_{FIELD}.npz
  - data/data_vectorizer/tfidf_vocab_{FIELD}.json

Saídas (por padrão):
  - data/data_analyze_vectorizer/top_terms_global_{FIELD}.png
  - data/data_analyze_vectorizer/top_terms_por_categoria_{FIELD}.png
  - data/data_analyze_vectorizer/wordcloud_global_{FIELD}.png
  - data/data_analyze_vectorizer/wordcloud_por_categoria_<CATEGORIA>_{FIELD}.png
  - data/data_analyze_vectorizer/similaridade_exemplo_{FIELD}.csv
  - data/data_analyze_vectorizer/categorias_frequencia_{FIELD}.png

Requisitos:
  pip install matplotlib wordcloud scikit-learn scipy pandas numpy
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------- Utils -------------------------

def safe_filename(s: str) -> str:
    """Sanitiza um nome para virar parte de filename."""
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:100] or "NA"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------- Load -------------------------

def load_data(input_csv: str, vec_dir: str, field: str):
    tfidf_path = os.path.join(vec_dir, f"tfidf_{field}.npz")
    vocab_path = os.path.join(vec_dir, f"tfidf_vocab_{field}.json")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV pré-processado não encontrado: {input_csv} (rode preprocess_csv.py)")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"Matriz TF-IDF não encontrada: {tfidf_path} (rode vectorize_items.py com --text-field {field})")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulário não encontrado: {vocab_path} (rode vectorize_items.py com --text-field {field})")

    df = pd.read_csv(input_csv, dtype=str).fillna("")
    X = sparse.load_npz(tfidf_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
        # vectorize_items.py salva um dict termo->índice direto
        vocab = vocab_json if isinstance(vocab_json, dict) else vocab_json.get("vocab", {})
        if not isinstance(vocab, dict):
            raise ValueError(f"Formato inesperado em {vocab_path}")

    inv_vocab = {i: t for t, i in vocab.items()}
    print(f"[INFO] Carregado: {X.shape[0]} docs, {X.shape[1]} termos | campo: {field}")
    return df, X, inv_vocab


# ------------------------- Plots -------------------------

def plot_top_terms_global(X, inv_vocab, out_png: str, top_n: int = 20):
    term_sums = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(term_sums)[::-1][:top_n]
    termos = [inv_vocab[i] for i in idx]
    valores = term_sums[idx]

    plt.figure(figsize=(11, 6))
    plt.barh(range(top_n)[::-1], valores[::-1])
    plt.yticks(range(top_n)[::-1], termos[::-1])
    plt.title("Top termos globais (TF-IDF total)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Gráfico global salvo em {out_png}")

    return pd.DataFrame({"termo": termos, "tfidf_sum": valores})


def wordcloud_from_tfidf(X, inv_vocab, out_path):
    term_sums = np.asarray(X.sum(axis=0)).ravel()
    nz = np.where(term_sums > 0)[0]
    freqs = {inv_vocab[i]: float(term_sums[i]) for i in nz}
    wc = WordCloud(width=1400, height=900, background_color="white", max_words=400)
    wc.generate_from_frequencies(freqs)
    wc.to_file(out_path)
    print(f"[OK] Wordcloud salva em {out_path}")


def analyze_by_category(df, X, inv_vocab, out_dir: str, category_field: str, top_n: int = 10, top_k_categories: int = 5):
    if category_field not in df.columns:
        print(f"[WARN] Coluna '{category_field}' não encontrada; pulando análise por categoria.")
        return

    cats = df[category_field].fillna("N/A").astype(str)
    cat_counts = cats.value_counts()

    # gráfico geral das categorias
    plot_cats = os.path.join(out_dir, f"categorias_frequencia_{category_field}.png")
    plt.figure(figsize=(12, 6))
    cat_counts.head(20).plot(kind="bar")
    plt.title(f"Categorias mais frequentes (top {min(20, len(cat_counts))}) — {category_field}")
    plt.ylabel("Quantidade de itens")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_cats, dpi=150)
    plt.close()
    print(f"[OK] Frequência de categorias salva em {plot_cats}")

    # top categorias para detalhar
    top_cats = cat_counts.head(top_k_categories).index.tolist()
    if not top_cats:
        return

    plot_top_cat = os.path.join(out_dir, f"top_terms_por_categoria_{category_field}.png")
    fig, axes = plt.subplots(len(top_cats), 1, figsize=(10, 3.5 * len(top_cats)))
    if len(top_cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, top_cats):
        mask = (cats == cat).values
        Xc = X[mask]
        term_sums = np.asarray(Xc.sum(axis=0)).ravel()
        idx = np.argsort(term_sums)[::-1][:top_n]
        termos = [inv_vocab[i] for i in idx]
        valores = term_sums[idx]
        ax.barh(range(top_n)[::-1], valores[::-1])
        ax.set_yticks(range(top_n)[::-1])
        ax.set_yticklabels(termos[::-1])
        ax.set_title(f"Top termos — {category_field} = {cat}")
    plt.tight_layout()
    plt.savefig(plot_top_cat, dpi=150)
    plt.close()
    print(f"[OK] Gráfico por categoria salvo em {plot_top_cat}")

    # wordclouds por categoria
    for cat in top_cats:
        mask = (cats == cat).values
        Xc = X[mask]
        term_sums = np.asarray(Xc.sum(axis=0)).ravel()
        nz = np.where(term_sums > 0)[0]
        freqs = {inv_vocab[i]: float(term_sums[i]) for i in nz}
        wc = WordCloud(width=1200, height=700, background_color="white", max_words=300)
        wc.generate_from_frequencies(freqs)
        out_path = os.path.join(out_dir, f"wordcloud_por_categoria_{safe_filename(str(cat))}.png")
        wc.to_file(out_path)
        print(f"[OK] Wordcloud da categoria '{cat}' salva em {out_path}")


def exemplo_similaridade(df, X, out_dir: str, field: str, doc_idx: int = 0, top_k: int = 5):
    if X.shape[0] < 2 or doc_idx >= X.shape[0]:
        return
    sims = cosine_similarity(X[doc_idx], X).ravel()
    idx = np.argsort(sims)[::-1][1:top_k+1]

    # Escolhe colunas amigáveis se existirem
    cols_pref = [c for c in ["Nome", "nome", "Voc", "categoria", "Atributos", "Notas", "Descrição", "descricao"] if c in df.columns]
    subset = df.loc[idx, cols_pref].copy() if cols_pref else df.loc[idx].copy()

    subset.insert(0, "doc_idx", idx)
    subset["similaridade"] = sims[idx]
    out_path = os.path.join(out_dir, f"similaridade_exemplo_{field}.csv")
    subset.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Similaridade (doc {doc_idx}) salva em {out_path}")


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Análise dos itens com base na matriz TF-IDF.")
    ap.add_argument("--input-csv", default=os.path.join("data", "items_info_preprocessed.csv"),
                    help="CSV pré-processado gerado pelo preprocess_csv.py")
    ap.add_argument("--vec-dir", default=os.path.join("data", "data_vectorizer"),
                    help="Diretório onde estão tfidf_{FIELD}.npz e tfidf_vocab_{FIELD}.json (default: data/data_vectorizer)")
    ap.add_argument("--out-dir", default=os.path.join("data", "data_analyze_vectorizer"),
                    help="Diretório de SAÍDA para gráficos/CSVs da análise (default: data/data_analyze_vectorizer)")
    ap.add_argument("--text-field", default="lemmas",
                    help="Campo textual usado na vetorização (ex.: lemmas, stems, tokens_sem_stopwords, texto_normalizado)")
    ap.add_argument("--category-field", default="Voc",
                    help="Coluna categórica para análise por categoria (ex.: Voc, Tipo de Dano, etc.)")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N termos globais (default: 20)")
    ap.add_argument("--top-n-cat", type=int, default=10, help="Top-N termos por categoria (default: 10)")
    ap.add_argument("--top-k-categories", type=int, default=5, help="Qtde de categorias para detalhar (default: 5)")
    ap.add_argument("--doc-idx", type=int, default=0, help="Índice do documento para exemplo de similaridade (default: 0)")
    ap.add_argument("--similar-k", type=int, default=5, help="Top-K documentos similares (default: 5)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    df, X, inv_vocab = load_data(args.input_csv, args.vec_dir, args.text_field)

    # Caminhos de saída (nomeados por field) — SEMPRE em out_dir
    plot_top_global = os.path.join(args.out_dir, f"top_terms_global_{args.text_field}.png")
    wordcloud_global = os.path.join(args.out_dir, f"wordcloud_global_{args.text_field}.png")

    # Top termos globais + wordcloud global
    plot_top_terms_global(X, inv_vocab, out_png=plot_top_global, top_n=args.top_n)
    wordcloud_from_tfidf(X, inv_vocab, out_path=wordcloud_global)

    # Análises por categoria (com gráfico e nuvens)
    analyze_by_category(
        df=df,
        X=X,
        inv_vocab=inv_vocab,
        out_dir=args.out_dir,
        category_field=args.category_field,
        top_n=args.top_n_cat,
        top_k_categories=args.top_k_categories
    )

    # Similaridade de exemplo
    exemplo_similaridade(df, X, out_dir=args.out_dir, field=args.text_field,
                         doc_idx=args.doc_idx, top_k=args.similar_k)

    print("[OK] Análise concluída.")


if __name__ == "__main__":
    main()
