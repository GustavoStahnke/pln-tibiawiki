#!/usr/bin/env python3
"""Sistema de recomendação simples para itens do TibiaWiki (versão funcional, sem POO)."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from vectorizer_items import (
    create_bow_vectors,
    create_tfidf_vectors,
    create_sbert_embeddings,
    calculate_similarity_matrix,
    search_similar_documents,
)

# ==============================
# CLUSTERIZAÇÃO E PROJEÇÃO PCA
# ==============================
def perform_clustering(df, vectors, method="sbert", n_clusters=None):
    """Executa KMeans para agrupar os itens com base nos vetores."""
    if n_clusters is None:
        n_samples = len(df)
        if n_samples <= 10:
            n_clusters = 3
        elif n_samples <= 30:
            n_clusters = 4
        elif n_samples <= 60:
            n_clusters = 5
        else:
            n_clusters = 6

    if method == "sbert":
        X = vectors["sbert_embeddings"]
    elif method == "tfidf":
        X = vectors["tfidf_matrix"].toarray()
    else:
        X = vectors["bow_matrix"].toarray()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X)

    cluster_df = pd.DataFrame({
        "Nome": df["Nome"],
        "cluster": clusters
    })
    return cluster_df


def create_pca_projection(df, vectors, clusters=None, method="sbert"):
    """Reduz a dimensionalidade dos vetores para 2 componentes principais (PCA)."""
    if method == "sbert":
        X = vectors["sbert_embeddings"]
    elif method == "tfidf":
        X = vectors["tfidf_matrix"].toarray()
    else:
        X = vectors["bow_matrix"].toarray()

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    pca_df = pd.DataFrame(coords, columns=["pc1", "pc2"])
    pca_df.insert(0, "Nome", df["Nome"])
    if clusters is not None:
        pca_df.insert(1, "cluster", clusters["cluster"])

    return pca_df


# ==============================
# RECOMENDAÇÃO E BUSCA
# ==============================
def recommend_by_title(df, vectors, title, method="sbert", top_k=5):
    """Recomenda itens semelhantes a partir do nome do item."""
    if "Nome" not in df.columns:
        raise ValueError("A coluna 'Nome' não foi encontrada no dataset.")

    try:
        item_idx = df["Nome"].tolist().index(title)
    except ValueError:
        raise ValueError(f"Item '{title}' não encontrado no dataset.")

    similarity_matrix = calculate_similarity_matrix(
        method=method,
        bow_matrix=vectors.get("bow_matrix"),
        tfidf_matrix=vectors.get("tfidf_matrix"),
        sbert_embeddings=vectors.get("sbert_embeddings"),
    )

    similarities = similarity_matrix[item_idx]
    sim_scores = [(i, sim) for i, sim in enumerate(similarities) if i != item_idx]
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, (idx, sim) in enumerate(sim_scores[:top_k]):
        row = df.iloc[idx]
        recommendations.append({
            "rank": i + 1,
            "Nome": row["Nome"],
            "Voc": row.get("Voc", "N/A"),
            "Lvl": row.get("Lvl", "N/A"),
            "similaridade": float(sim),
            "url_origem": row.get("url_origem", "N/A")
        })

    return recommendations


def recommend_by_query(df, vectors, query, top_k=5):
    """Busca itens semanticamente similares a uma consulta textual."""
    if vectors.get("sbert_embeddings") is None:
        raise ValueError("Embeddings SBERT não disponíveis. Gere-os antes de usar a busca.")

    # Criar embeddings para a query e calcular similaridades
    from vectorizer_items import _load_sbert
    model = _load_sbert()
    embeddings = vectors["sbert_embeddings"]

    sims = search_similar_documents(query, embeddings, model)
    top_idx = np.argsort(sims)[::-1][:top_k]

    recommendations = []
    for i, idx in enumerate(top_idx):
        row = df.iloc[idx]
        recommendations.append({
            "rank": i + 1,
            "Nome": row["Nome"],
            "Voc": row.get("Voc", "N/A"),
            "Lvl": row.get("Lvl", "N/A"),
            "similaridade": float(sims[idx]),
            "url_origem": row.get("url_origem", "N/A")
        })
    return recommendations

def recommend_substitutes(df, vectors, title, method="sbert", top_k=10, lvl_tolerance=20):
    """
    Recomenda itens substitutos (parecidos) levando em conta:
      - Similaridade semântica
      - Mesma vocação (Voc)
      - Nível próximo (Lvl)
    """
    if "Nome" not in df.columns:
        raise ValueError("A coluna 'Nome' não foi encontrada no dataset.")

    # Localizar item base
    try:
        base_item = df[df["Nome"] == title].iloc[0]
    except IndexError:
        raise ValueError(f"Item '{title}' não encontrado no dataset.")

    base_voc = str(base_item.get("Voc", "")).strip()
    base_lvl = base_item.get("Lvl", 0)

    # Obter recomendações semânticas
    recs = recommend_by_title(df, vectors, title, method=method, top_k=top_k * 3)

    # Filtrar por vocação e tolerância de nível
    filtered = []
    for r in recs:
        voc_match = (str(r["Voc"]).strip() == base_voc) if base_voc else True
        lvl_match = (
            abs(float(r["Lvl"]) - float(base_lvl)) <= lvl_tolerance
            if not pd.isna(r["Lvl"]) and not pd.isna(base_lvl)
            else True
        )
        if voc_match and lvl_match:
            filtered.append(r)
        if len(filtered) >= top_k:
            break

    # Retorno formatado
    return filtered


# ==============================
# SALVAMENTO
# ==============================
def save_results(output_dir, cluster_df=None, pca_df=None, similarity_matrix=None, df=None, method="sbert"):
    """Salva resultados em CSV no diretório especificado."""
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if cluster_df is not None:
        cluster_df.to_csv(output_path / "cluster_labels.csv", index=False, sep=";")

    if pca_df is not None:
        pca_df.to_csv(output_path / "cluster_pca2d.csv", index=False, sep=";")

    if similarity_matrix is not None and df is not None:
        sim_df = pd.DataFrame(similarity_matrix, index=df["Nome"], columns=df["Nome"])
        sim_df.to_csv(output_path / f"similarity_{method}.csv", sep=";")
