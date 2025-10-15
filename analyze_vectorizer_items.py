"""
analyze_items.py
----------------
Analisa os resultados da vetorização TF-IDF gerada pelo vectorizer_items.py.
Produz estatísticas, gráficos e nuvens de palavras dos itens do TibiaWiki.

Entrada esperada:
  - data/items_info.csv
  - data/tfidf_items.npz
  - data/tfidf_vocab.json

Saídas geradas:
  - data/top_terms_global.png
  - data/top_terms_por_categoria.png
  - data/wordcloud_global.png
  - data/wordcloud_por_categoria_<categoria>.png
  - data/similaridade_exemplo.csv
  - data/categorias_frequencia.png

Requisitos:
  pip install matplotlib wordcloud scikit-learn scipy pandas numpy
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "items_info.csv")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_items.npz")
VOCAB_PATH = os.path.join(DATA_DIR, "tfidf_vocab.json")

# arquivos de saída
PLOT_TOP_GLOBAL = os.path.join(DATA_DIR, "top_terms_global.png")
PLOT_TOP_CAT = os.path.join(DATA_DIR, "top_terms_por_categoria.png")
WORDCLOUD_GLOBAL = os.path.join(DATA_DIR, "wordcloud_global.png")
PLOT_CATS_FREQ = os.path.join(DATA_DIR, "categorias_frequencia.png")


def load_data():
    if not (os.path.exists(CSV_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(VOCAB_PATH)):
        raise FileNotFoundError("Faltam arquivos. Rode antes o vectorizer_items.py")

    df = pd.read_csv(CSV_PATH)
    X = sparse.load_npz(TFIDF_PATH)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
        vocab = vocab_json.get("vocab", vocab_json)
        if isinstance(vocab, dict):
            inv_vocab = {i: t for t, i in vocab.items()}
        else:
            raise ValueError("Formato inesperado em tfidf_vocab.json")

    print(f"[INFO] Dados carregados: {X.shape[0]} docs, {X.shape[1]} termos")
    return df, X, inv_vocab


def top_terms_global(X, inv_vocab, top_n=20):
    term_sums = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(term_sums)[::-1][:top_n]
    termos = [inv_vocab[i] for i in idx]
    valores = term_sums[idx]

    plt.figure(figsize=(10, 5))
    plt.barh(range(top_n)[::-1], valores[::-1], color="#4B8BBE")
    plt.yticks(range(top_n)[::-1], termos[::-1])
    plt.title("Top termos globais (TF-IDF total)")
    plt.tight_layout()
    plt.savefig(PLOT_TOP_GLOBAL)
    plt.close()
    print(f"[OK] Gráfico global salvo em {PLOT_TOP_GLOBAL}")

    return pd.DataFrame({"termo": termos, "tfidf_sum": valores})


def wordcloud_from_tfidf(X, inv_vocab, out_path):
    term_sums = np.asarray(X.sum(axis=0)).ravel()
    freqs = {inv_vocab[i]: float(term_sums[i]) for i in np.where(term_sums > 0)[0]}
    wc = WordCloud(width=1200, height=800, background_color="white", max_words=300)
    wc.generate_from_frequencies(freqs)
    wc.to_file(out_path)
    print(f"[OK] Wordcloud salva em {out_path}")


def analyze_by_category(df, X, inv_vocab, top_n=10):
    if "categoria" not in df.columns:
        print("[WARN] Coluna 'categoria' não encontrada, pulando análise por categoria.")
        return

    cats = df["categoria"].fillna("N/A").astype(str)
    cat_counts = cats.value_counts()
    top_cats = cat_counts.head(5).index  # top 5 categorias mais frequentes

    # ---- gráfico geral das categorias
    plt.figure(figsize=(10, 5))
    cat_counts.head(15).plot(kind="bar", color="#306998")
    plt.title("Categorias mais frequentes (top 15)")
    plt.ylabel("Quantidade de itens")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_CATS_FREQ)
    plt.close()
    print(f"[OK] Frequência de categorias salva em {PLOT_CATS_FREQ}")

    # ---- top termos por categoria
    fig, axes = plt.subplots(len(top_cats), 1, figsize=(8, 12))
    if len(top_cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, top_cats):
        mask = (cats == cat).values
        Xc = X[mask]
        term_sums = np.asarray(Xc.sum(axis=0)).ravel()
        idx = np.argsort(term_sums)[::-1][:top_n]
        termos = [inv_vocab[i] for i in idx]
        valores = term_sums[idx]
        ax.barh(range(top_n)[::-1], valores[::-1], color="#FFD43B")
        ax.set_yticks(range(top_n)[::-1])
        ax.set_yticklabels(termos[::-1])
        ax.set_title(f"Top termos - {cat}")
    plt.tight_layout()
    plt.savefig(PLOT_TOP_CAT)
    plt.close()
    print(f"[OK] Gráfico por categoria salvo em {PLOT_TOP_CAT}")

    # ---- wordclouds por categoria
    for cat in top_cats:
        mask = (cats == cat).values
        Xc = X[mask]
        term_sums = np.asarray(Xc.sum(axis=0)).ravel()
        freqs = {inv_vocab[i]: float(term_sums[i]) for i in np.where(term_sums > 0)[0]}
        wc = WordCloud(width=1000, height=600, background_color="white", max_words=200)
        wc.generate_from_frequencies(freqs)
        out_path = os.path.join(DATA_DIR, f"wordcloud_por_categoria_{cat.replace('/', '_')}.png")
        wc.to_file(out_path)
        print(f"[OK] Wordcloud da categoria '{cat}' salva em {out_path}")


def exemplo_similaridade(df, X, inv_vocab, doc_idx=0, top_k=5):
    if X.shape[0] < 2:
        return
    sims = cosine_similarity(X[doc_idx], X).ravel()
    idx = np.argsort(sims)[::-1][1:top_k+1]
    subset = df.loc[idx, ["nome", "categoria", "descricao"]].copy() if "nome" in df.columns else df.loc[idx].copy()
    subset["similaridade"] = sims[idx]
    out_path = os.path.join(DATA_DIR, "similaridade_exemplo.csv")
    subset.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Similaridade (doc {doc_idx}) salva em {out_path}")


def main():
    df, X, inv_vocab = load_data()

    # Top termos globais + wordcloud global
    top_terms_global(X, inv_vocab, top_n=20)
    wordcloud_from_tfidf(X, inv_vocab, WORDCLOUD_GLOBAL)

    # Análises por categoria (com gráfico e nuvens)
    analyze_by_category(df, X, inv_vocab, top_n=10)

    # Similaridade de exemplo
    exemplo_similaridade(df, X, inv_vocab, doc_idx=0, top_k=5)

    print("[OK] Análise concluída.")


if __name__ == "__main__":
    main()
