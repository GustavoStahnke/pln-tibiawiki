"""
Vectorização TF-IDF para o CSV pré-processado do TibiaWiki.

Lê:   data/items_info_preprocessed.csv
Usa:  coluna de texto escolhida (ex.: 'lemmas', 'stems', 'tokens_sem_stopwords', 'texto_normalizado')
Gera: 
  - data/data_vectorizer/tfidf_{field}.npz
  - data/data_vectorizer/tfidf_vocab_{field}.json
  - data/data_vectorizer/top_terms_global_{field}.csv
  - data/data_vectorizer/top_terms_by_{group}_{field}.csv (se --group-by)
  - data/data_vectorizer/feature_names_{field}.csv

Uso:
  python vectorize_items.py --text-field lemmas --group-by Voc
"""

import os
import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# --------- Caminhos padrão ---------
INPUT_CSV = os.path.join("data", "items_info_preprocessed.csv")
OUT_DIR   = os.path.join("data", "data_vectorizer")


def build_vectorizer(
    ngram_range: Tuple[int, int],
    min_df,
    max_df,
    max_features: Optional[int],
):
    """Cria o TfidfVectorizer configurado para textos já normalizados."""
    token_pattern = r"(?u)\b[\w\%\+\-]{2,}\b"  # mantém %, +, -
    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        lowercase=False,  # já está minúsculo
        norm="l2",
        token_pattern=token_pattern,
    )
    return vect


def save_artifacts(
    X,
    vocab: dict,
    field: str,
    feature_names: List[str],
    group_by: Optional[str],
    df: pd.DataFrame,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    # caminhos de saída
    tfidf_mtx = os.path.join(OUT_DIR, f"tfidf_{field}.npz")
    tfidf_voc = os.path.join(OUT_DIR, f"tfidf_vocab_{field}.json")
    top_glob  = os.path.join(OUT_DIR, f"top_terms_global_{field}.csv")
    feat_map  = os.path.join(OUT_DIR, f"feature_names_{field}.csv")

    # matriz
    sparse.save_npz(tfidf_mtx, X)

    # vocabulário
    with open(tfidf_voc, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # top termos global
    term_sums = np.asarray(X.sum(axis=0)).ravel()
    order = np.argsort(term_sums)[::-1]
    top_terms = pd.DataFrame({
        "termo": [feature_names[i] for i in order],
        "tfidf_sum": [float(term_sums[i]) for i in order],
    })
    top_terms.head(1000).to_csv(top_glob, index=False, encoding="utf-8")

    # mapa de índices -> termos
    pd.DataFrame({
        "index": list(range(len(feature_names))),
        "termo": feature_names
    }).to_csv(feat_map, index=False, encoding="utf-8")

    # top termos por grupo
    if group_by and group_by in df.columns:
        groups = df[group_by].fillna("N/A").astype(str).values
        rows = []
        for g in np.unique(groups):
            mask = (groups == g)
            if mask.sum() < 2:
                continue
            Xg = X[mask]
            sums = np.asarray(Xg.sum(axis=0)).ravel()
            idx = np.argsort(sums)[::-1][:200]
            rows += [(g, feature_names[i], float(sums[i])) for i in idx]
        if rows:
            out_grp = os.path.join(OUT_DIR, f"top_terms_by_{group_by}_{field}.csv")
            pd.DataFrame(rows, columns=[group_by, "termo", "tfidf_sum"]).to_csv(
                out_grp, index=False, encoding="utf-8"
            )
            print(f"[OK] Top termos por {group_by} salvo em: {out_grp}")

    print(f"[OK] TF-IDF salvo em: {tfidf_mtx}")
    print(f"[OK] Vocabulário salvo em: {tfidf_voc}")
    print(f"[OK] Top termos global salvo em: {top_glob}")
    print(f"[OK] Mapa de termos salvo em: {feat_map}")


def main():
    ap = argparse.ArgumentParser(description="Vectorização TF-IDF para itens do TibiaWiki (CSV pré-processado).")
    ap.add_argument("--input", default=INPUT_CSV, help="Caminho do CSV pré-processado (default: data/items_info_preprocessed.csv)")
    ap.add_argument("--text-field", default="lemmas", help="Coluna de texto a vetorizar (ex.: lemmas, stems, tokens_sem_stopwords, texto_normalizado)")
    ap.add_argument("--group-by", default=None, help="Coluna para agrupar e extrair top termos por grupo (ex.: Voc, Tipo de Dano). Opcional.")
    ap.add_argument("--ngram-min", type=int, default=1, help="n em n-gram mínimo (default: 1)")
    ap.add_argument("--ngram-max", type=int, default=2, help="n em n-gram máximo (default: 2)")
    ap.add_argument("--min-df", default=2, type=float, help="min_df (int ou float em proporção). Default: 2")
    ap.add_argument("--max-df", default=0.95, type=float, help="max_df (float em proporção ou int). Default: 0.95")
    ap.add_argument("--max-features", default=None, type=int, help="Limite de features. Opcional.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Não encontrei {args.input}. Rode antes o preprocess_csv.py")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str).fillna("")

    if args.text_field not in df.columns:
        text_like = [c for c in ["lemmas", "stems", "tokens_sem_stopwords", "texto_normalizado", "tokens", "texto_unificado"] if c in df.columns]
        raise ValueError(
            f"Coluna '{args.text_field}' não encontrada no CSV. "
            f"Tente uma destas: {text_like}"
        )

    texts = df[args.text_field].astype(str).tolist()

    vect = build_vectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )

    X = vect.fit_transform(texts)
    vocab = vect.vocabulary_
    feature_names = list(vect.get_feature_names_out())

    save_artifacts(
        X=X,
        vocab=vocab,
        field=args.text_field,
        feature_names=feature_names,
        group_by=args.group_by,
        df=df,
    )

    print(f"[OK] Linhas: {X.shape[0]} | Features: {X.shape[1]}")
    if args.group_by and args.group_by not in df.columns:
        print(f"[WARN] Coluna '{args.group_by}' não existe no CSV. Estatística por grupo não foi gerada.")


if __name__ == "__main__":
    main()
