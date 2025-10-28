"""
- Lê VETORES já gerados de: data/data_vectorizer/
    - bow_matrix.csv
    - tfidf_matrix.csv
    - sbert_embeddings.npy
    - metadata.json
- Gera e SALVA resultados de análise em: data/data_analyze_vectorizer/
    - cluster_labels.csv
    - cluster_pca2d.csv
    - similarity_sbert.csv

Se os vetores não existirem, o script calcula BoW/TF-IDF/SBERT na hora
(usando vectorizer_items.py), porém NÃO salva os vetores na pasta antiga;
apenas usa para análise e salva os artefatos em data/data_analyze_vectorizer/.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from vectorizer_items import (
    create_bow_vectors,
    create_tfidf_vectors,
    create_sbert_embeddings,
    calculate_similarity_matrix,
)
from recommendation_items import (
    perform_clustering,
    create_pca_projection,
    recommend_by_title,
    recommend_by_query,
    save_results,
)

# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path("data")
INPUT_VECTORS_DIR = DATA_DIR / "data_vectorizer"        
OUTPUT_ANALYZE_DIR = DATA_DIR / "data_analyze_vectorizer"
DATASET_PATH = DATA_DIR / "items_info_preprocessed.csv"


# -----------------------------
# Utilitários
# -----------------------------

def vetores_prontos(input_dir: Path = INPUT_VECTORS_DIR) -> bool:
    """Verifica se os vetores já foram criados (na pasta data/data_vectorizer)."""
    arquivos_necessarios = [
        "bow_matrix.csv",
        "tfidf_matrix.csv",
        "sbert_embeddings.npy",
        "metadata.json",
    ]
    todos_existem = all((input_dir / nome).exists() for nome in arquivos_necessarios)
    if todos_existem:
        print("✅ Vetores existentes encontrados em:", input_dir)
    else:
        print("❌ Vetores não encontrados em:", input_dir, "\n   → O script calculará os vetores em memória.")
    return todos_existem


def load_data(data_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carrega o dataset pré-processado de itens."""
    df = pd.read_csv(data_path)
    print(f"✅ {len(df)} itens carregados de {data_path}.")
    return df


# -----------------------------
# Pipeline principal
# -----------------------------

def analyze_items():
    """Executa todo o pipeline de vetorização/clusterização/recomendação e salva os resultados na pasta de análise."""
    print("🔍 Iniciando análise de itens do TibiaWiki...")

    df = load_data()

    # 1) Carregar ou calcular vetores
    if vetores_prontos(INPUT_VECTORS_DIR):
        print("📂 Carregando vetores existentes (data/data_vectorizer)...")
        bow_df = pd.read_csv(INPUT_VECTORS_DIR / "bow_matrix.csv", sep=";")
        tfidf_df = pd.read_csv(INPUT_VECTORS_DIR / "tfidf_matrix.csv", sep=";")
        sbert_embeddings = np.load(INPUT_VECTORS_DIR / "sbert_embeddings.npy")

        bow_matrix = bow_df.values
        tfidf_matrix = tfidf_df.values
        bow_features = bow_df.columns.tolist()
        tfidf_features = tfidf_df.columns.tolist()

        print("   ✅ Vetores carregados com sucesso!")
    else:
        print("🧮 Gerando vetores em memória (BoW/TF-IDF/SBERT)...")
        corpus = df["lemmas"].fillna("").tolist()

        bow_matrix, bow_features = create_bow_vectors(corpus)
        tfidf_matrix, tfidf_features = create_tfidf_vectors(corpus)

        # Alguns vectorizers retornam apenas embeddings; outros (seu) podem retornar (embeddings, model).
        sbert_result = create_sbert_embeddings(corpus)
        if isinstance(sbert_result, tuple) and len(sbert_result) >= 1:
            sbert_embeddings = sbert_result[0]
        else:
            sbert_embeddings = sbert_result

    vectors = {
        "bow_matrix": bow_matrix,
        "bow_features": bow_features,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_features": tfidf_features,
        "sbert_embeddings": sbert_embeddings,
    }

    # 2) Clusterização e projeção PCA
    print("📊 Criando clusters e projeção PCA...")
    cluster_df = perform_clustering(df, vectors, method="sbert")
    pca_df = create_pca_projection(df, vectors, cluster_df, method="sbert")

    # 3) Similaridade (SBERT)
    print("🔗 Calculando matriz de similaridade (SBERT)...")
    similarity_matrix = calculate_similarity_matrix(
        method="sbert",
        bow_matrix=bow_matrix,
        tfidf_matrix=tfidf_matrix,
        sbert_embeddings=sbert_embeddings
    )

    # 4) Salvar análise em data/data_analyze_vectorizer
    OUTPUT_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    save_results(OUTPUT_ANALYZE_DIR, cluster_df, pca_df, similarity_matrix, df, method="sbert")
    print(f"📂 Resultados de análise salvos em: {OUTPUT_ANALYZE_DIR}/")

    return df, vectors


# -----------------------------
# Helpers de exibição
# -----------------------------

def recommend_items_by_name(title, df, vectors, top_k=5):
    """Mostra recomendações por nome de item."""
    recs = recommend_by_title(df, vectors, title, method="sbert", top_k=top_k)
    print(f"\n🎯 Itens semelhantes a '{title}':")
    print("=" * 60)
    for r in recs:
        print(f"{r['rank']}. {r['Nome']} (similaridade: {r['similaridade']:.3f})")
        print(f"   Vocação: {r['Voc']} | Nível: {r['Lvl']}")
        print(f"   Link: {r['url_origem']}\n")


def search_items(query, df, vectors, top_k=5):
    """Mostra resultados de busca semântica."""
    recs = recommend_by_query(df, vectors, query, top_k=top_k)
    print(f"\n🔎 Resultados para '{query}':")
    print("=" * 60)
    for r in recs:
        print(f"{r['rank']}. {r['Nome']} (similaridade: {r['similaridade']:.3f})")
        print(f"   Vocação: {r['Voc']} | Nível: {r['Lvl']}")
        print(f"   Link: {r['url_origem']}\n")


def recommend_substitute_items(title, df, vectors, top_k=5):
    """Mostra itens substitutos considerando vocação e nível (usa função do recommendation_items)."""
    from recommendation_items import recommend_substitutes
    recs = recommend_substitutes(df, vectors, title, top_k=top_k)
    print(f"\n🪄 Itens substitutos para '{title}':")
    print("=" * 60)
    for r in recs:
        print(f"{r['rank']}. {r['Nome']} (sim: {r['similaridade']:.3f})")
        print(f"   Vocação: {r['Voc']} | Nível: {r['Lvl']}")
        print(f"   Link: {r['url_origem']}\n")


# -----------------------------
# Main
# -----------------------------

def main():
    """Executa o pipeline completo e mostra exemplos."""
    df, vectors = analyze_items()

    print("\n" + "=" * 60)
    print("🧭 EXEMPLOS DE USO")
    print("=" * 60)

    # Recomendação por nome
    recommend_items_by_name("Alicorn Quiver", df, vectors)

    # Busca por query
    search_items("arco mágico para paladin", df, vectors)

    # Itens substitutos (classe/nível)
    recommend_substitute_items("Alicorn Quiver", df, vectors)


if __name__ == "__main__":
    main()
