"""
Vectorização para itens do TibiaWiki
-----------------------------------------------------------------
- create_bow_vectors
- create_tfidf_vectors
- create_sbert_embeddings
- calculate_similarity_matrix
- search_similar_documents
- save_vectors

Adaptações ao dataset:
- Lê um CSV com os itens (padrão: items_info_preprocessed.csv)
- Usa a coluna "lemmas" por padrão para o corpus textual (fallback: "texto_normalizado")
- Salva arquivos em um diretório de saída (padrão: data/data_analyze_vectorizer)

Requisitos:
  pip install numpy pandas scikit-learn sentence-transformers
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SBERT é importado sob demanda para permitir uso só quando necessário
def _load_sbert(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

# -----------------------------
# Funções de vetorizaçao
# -----------------------------

def _prepare_corpus(series: pd.Series) -> List[str]:
    """Converte série em lista de strings limpando NaNs e listas."""
    series = series.fillna("")
    corpus = []
    for val in series.tolist():
        if isinstance(val, list):
            corpus.append(" ".join(map(str, val)))
        else:
            corpus.append(str(val))
    return corpus

def create_bow_vectors(corpus: List[str]):
    vectorizer = CountVectorizer(lowercase=True, min_df=2, max_df=0.95)
    bow_matrix = vectorizer.fit_transform(corpus)
    bow_features = vectorizer.get_feature_names_out()
    return bow_matrix, bow_features

def create_tfidf_vectors(corpus: List[str]):
    vectorizer = TfidfVectorizer(lowercase=True, min_df=2, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_features = vectorizer.get_feature_names_out()
    return tfidf_matrix, tfidf_features

def create_sbert_embeddings(corpus: List[str], model=None, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    if model is None:
        model = _load_sbert(model_name)
    embeddings = model.encode(corpus, normalize_embeddings=True)
    return embeddings, model

# -----------------------------
# Similaridade e busca
# -----------------------------

def calculate_similarity_matrix(method: str,
                                bow_matrix=None,
                                tfidf_matrix=None,
                                sbert_embeddings=None):
    method = method.lower()
    if method == "bow" and bow_matrix is not None:
        return cosine_similarity(bow_matrix)
    elif method == "tfidf" and tfidf_matrix is not None:
        return cosine_similarity(tfidf_matrix)
    elif method == "sbert" and sbert_embeddings is not None:
        return cosine_similarity(sbert_embeddings)
    else:
        raise ValueError(f"Método '{method}' não disponível ou matriz não calculada.")

def search_similar_documents(query: str, embeddings: np.ndarray, model) -> np.ndarray:
    """Retorna vetor de similaridade do 'query' contra todos os documentos."""
    qv = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(qv, embeddings)[0]
    return sims

# -----------------------------
# Salvamento
# -----------------------------

def save_vectors(output_dir: Path,
                 bow_matrix=None, bow_features: Optional[np.ndarray]=None,
                 tfidf_matrix=None, tfidf_features: Optional[np.ndarray]=None,
                 sbert_embeddings=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    if bow_matrix is not None:
        bow_df = pd.DataFrame.sparse.from_spmatrix(bow_matrix, columns=bow_features)
        bow_df.to_csv(output_dir / "bow_matrix.csv", index=False, sep=";")

    if tfidf_matrix is not None:
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=tfidf_features)
        tfidf_df.to_csv(output_dir / "tfidf_matrix.csv", index=False, sep=";")

    if sbert_embeddings is not None:
        np.save(output_dir / "sbert_embeddings.npy", sbert_embeddings)
        sbert_df = pd.DataFrame(sbert_embeddings)
        sbert_df.to_csv(output_dir / "sbert_embeddings.csv", index=False, sep=";")

# -----------------------------
# CLI / Execução
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Vectorizador funcional para itens do TibiaWiki.")
    parser.add_argument("--dataset", type=str, default="data/items_info_preprocessed.csv",
                        help="Caminho do CSV de itens (padrão: data/items_info_preprocessed.csv)")
    parser.add_argument("--text-col", type=str, default="lemmas",
                        help="Coluna de texto a usar (padrão: lemmas; fallback automático para texto_normalizado)")
    parser.add_argument("--output-dir", type=str, default="data/data_vectorizer",
                        help="Diretório de saída (padrão: data/data_vectorizer)")
    parser.add_argument("--run", nargs="+", choices=["bow", "tfidf", "sbert"], default=["bow","tfidf","sbert"],
                        help="Quais representações calcular (padrão: todas)")
    parser.add_argument("--similarity", choices=["bow","tfidf","sbert"], default=None,
                        help="Se definido, calcula e salva a matriz de similaridade do método escolhido em CSV")
    parser.add_argument("--query", type=str, default=None,
                        help="Se definido, procura itens semelhantes usando SBERT e imprime top-k")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Quantidade de resultados para a busca (padrão: 5)")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Nome do modelo SBERT (padrão: paraphrase-multilingual-MiniLM-L12-v2)")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    if not dataset_path.exists():
        # Tenta no diretório atual e no /mnt/data
        alt1 = Path("/mnt/data") / dataset_path.name
        if alt1.exists():
            dataset_path = alt1
        else:
            raise FileNotFoundError(f"Dataset não encontrado: {args.dataset}")

    df = pd.read_csv(dataset_path)

    # Escolha da coluna de texto
    text_col = args.text_col if args.text_col in df.columns else None
    if text_col is None:
        if "lemmas" in df.columns:
            text_col = "lemmas"
        elif "texto_normalizado" in df.columns:
            text_col = "texto_normalizado"
        else:
            raise ValueError("Não encontrei as colunas 'lemmas' ou 'texto_normalizado' no dataset.")

    corpus = _prepare_corpus(df[text_col])

    # Vetores
    bow_matrix = tfidf_matrix = sbert_embeddings = None
    bow_features = tfidf_features = None
    sbert_model = None

    if "bow" in args.run:
        bow_matrix, bow_features = create_bow_vectors(corpus)

    if "tfidf" in args.run:
        tfidf_matrix, tfidf_features = create_tfidf_vectors(corpus)

    if "sbert" in args.run or args.query is not None or args.similarity == "sbert":
        sbert_embeddings, sbert_model = create_sbert_embeddings(corpus, model=None, model_name=args.model_name)

    # Similaridade opcional
    if args.similarity:
        sim = calculate_similarity_matrix(method=args.similarity,
                                          bow_matrix=bow_matrix,
                                          tfidf_matrix=tfidf_matrix,
                                          sbert_embeddings=sbert_embeddings)
        # Salva matriz de similaridade como CSV (pode ser grande)
        sim_df = pd.DataFrame(sim)
        output_dir.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(output_dir / f"similarity_{args.similarity}.csv", index=False, sep=";")

    # Busca opcional (SBERT)
    if args.query is not None:
        if sbert_embeddings is None or sbert_model is None:
            sbert_embeddings, sbert_model = create_sbert_embeddings(corpus, model=None, model_name=args.model_name)
        sims = search_similar_documents(args.query, sbert_embeddings, sbert_model)
        top_idx = np.argsort(sims)[::-1][:args.top_k]
        results = [{"idx": int(i),
                    "similarity": float(sims[i]),
                    "Nome": df["Nome"].iloc[i] if "Nome" in df.columns else None,
                    "url_origem": df["url_origem"].iloc[i] if "url_origem" in df.columns else None} for i in top_idx]
        print(json.dumps(results, ensure_ascii=False, indent=2))

    # Salvamento dos vetores
    save_vectors(output_dir=output_dir,
                 bow_matrix=bow_matrix, bow_features=bow_features,
                 tfidf_matrix=tfidf_matrix, tfidf_features=tfidf_features,
                 sbert_embeddings=sbert_embeddings)

    # Também salva metadados úteis
    meta = {
        "dataset": dataset_path.as_posix(),
        "text_col": text_col,
        "rows": len(corpus),
        "columns": list(df.columns),
        "computed": args.run,
        "similarity_saved_for": args.similarity,
        "model_name": args.model_name
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Vetores salvos em: {output_dir}")

if __name__ == "__main__":
    main()
