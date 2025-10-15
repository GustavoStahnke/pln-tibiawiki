#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vectorizer_items.py
----------------
Lê data/items_info.csv e realiza vetorização (TF-IDF) sobre a coluna `descricao_limpa`.
Também calcula estatísticas rápidas: top termos globais e por categoria.
Salva a matriz TF-IDF em .npz e o vocabulário em .json para uso futuro.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

INPUT_CSV = os.path.join("data", "items_info.csv")
OUT_DIR   = "data"
TFIDF_MTX = os.path.join(OUT_DIR, "tfidf_items.npz")
TFIDF_VOC = os.path.join(OUT_DIR, "tfidf_vocab.json")
TOP_TERMS = os.path.join(OUT_DIR, "top_terms_global.csv")
TOP_CAT   = os.path.join(OUT_DIR, "top_terms_por_categoria.csv")

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Não encontrei {INPUT_CSV}. Rode antes o get_tibiaitens_info.py")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if "descricao_limpa" not in df.columns:
        raise ValueError("Coluna 'descricao_limpa' não encontrada. Garanta que o pré-processamento foi gerado.")

    texts = df["descricao_limpa"].fillna("").astype(str).tolist()

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=30000,     # limite opcional
        ngram_range=(1,2),      # unigrama + bigrama
        min_df=2,               # ignora termos muito raros
        max_df=0.95,            # ignora termos muito frequentes
        lowercase=False,        # já recebemos em minúsculas
        norm="l2",
    )
    X = vectorizer.fit_transform(texts)  # scipy.sparse matrix
    vocab = vectorizer.vocabulary_       # dict termo -> índice

    # Salva artefatos
    sparse.save_npz(TFIDF_MTX, X)
    with open(TFIDF_VOC, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Estatística global: top termos por soma de TF-IDF
    term_sums = np.asarray(X.sum(axis=0)).ravel()  # soma por coluna
    inv_vocab = {i:t for t,i in vocab.items()}
    top_idx = np.argsort(term_sums)[::-1][:100]
    top_terms = [(inv_vocab[i], float(term_sums[i])) for i in top_idx]
    pd.DataFrame(top_terms, columns=["termo","tfidf_sum"]).to_csv(TOP_TERMS, index=False, encoding="utf-8")

    # Estatística por categoria (se existir)
    if "categoria" in df.columns:
        rows = []
        cats = df["categoria"].fillna("N/A").astype(str)
        for cat in cats.unique():
            mask = (cats == cat).values
            if mask.sum() < 3:
                continue
            Xc = X[mask]
            sums = np.asarray(Xc.sum(axis=0)).ravel()
            idx = np.argsort(sums)[::-1][:30]
            rows += [(cat, inv_vocab[i], float(sums[i])) for i in idx]
        if rows:
            pd.DataFrame(rows, columns=["categoria","termo","tfidf_sum"]).to_csv(TOP_CAT, index=False, encoding="utf-8")

    print("[OK] TF-IDF salvo em:", TFIDF_MTX)
    print("[OK] Vocabulário salvo em:", TFIDF_VOC)
    print("[OK] Top termos global em:", TOP_TERMS)
    if os.path.exists(TOP_CAT):
        print("[OK] Top termos por categoria em:", TOP_CAT)

if __name__ == "__main__":
    main()
