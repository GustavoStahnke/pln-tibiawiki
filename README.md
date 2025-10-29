# 🧙‍♂️ Projeto de PLN — TibiaWiki Itens

Repositório do trabalho da disciplina de **Processamento de Linguagem Natural (PLN)** aplicando **vetorização** (BoW, TF-IDF, SBERT), **similaridade**, **clusterização (K-Means)**, **PCA** e **recomendação** para itens do Tibia, usando o **TibiaWiki** como fonte.

**Fonte de dados:** https://www.tibiawiki.com.br/wiki/Itens

---

## 🎯 Objetivo

- Coletar e estruturar itens do Tibia (links → páginas → atributos).  
- Pré-processar texto (normalização, stopwords, lemas/tokens).  
- Vetorizar (BoW/TF-IDF/SBERT) e calcular **similaridade** (cosseno).  
- Clusterizar (K-Means) e projetar (PCA) para análise exploratória.  
- Recomendar itens:  
  - por **nome** (itens similares);  
  - por **consulta** (busca semântica);  
  - **substitutos** (mesma *Voc* e tolerância de *Lvl*).

---

## 🗂️ Estrutura do Projeto

```
pln-tibiawiki/
│   analyze_vectorizer_items.py     # Lê vetores e gera clusters/PCA/similaridade -> data/data_analyze_vectorizer
│   get_tibiaitens_info.py          # Extrai atributos detalhados de cada link
│   get_tibiaitens_links.py         # Coleta links de itens no TibiaWiki
│   preprocess_csv.py               # Normalização, stopwords, lemas/tokens
│   recommendation_items.py         # Clustering, PCA, recomendação, substitutos
│   vectorizer_items.py             # BoW/TF-IDF/SBERT + similaridade do cosseno
│   requirements.txt
│   README.md
│
└── data/
    │   items_info.csv
    │   items_info_preprocessed.csv
    │
    ├── data_vectorizer/            # ENTRADA (vetores)
    │   ├── bow_matrix.csv
    │   ├── tfidf_matrix.csv
    │   ├── sbert_embeddings.npy
    │   └── metadata.json
    │
    └── data_analyze_vectorizer/    # SAÍDA (análises)
        ├── cluster_labels.csv
        ├── cluster_pca2d.csv
        └── similarity_sbert.csv
```

> **Atenção:** os CSVs de matrizes usam `sep=";"`. Mantenha esse separador ao ler/salvar.

---

## ⚙️ Ambiente & Instalação

**Requisitos**
- Python **3.10+**  
- Internet na **primeira execução do SBERT** (download automático do modelo)

**Instalação rápida**
```bash
# Criar e ativar ambiente
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# Baixar recursos NLTK usados no pré-processamento
python - << 'PY'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
PY
```

**Requisitos do Selenium (opcional no scraping)**  
Caso use Selenium, recomendo `webdriver-manager` (já no requirements) para gerenciar o driver automaticamente.

---

## ▶️ Como Executar (Passo a Passo)

> Você pode pular etapas se já tiver os artefatos gerados.

```bash
# (A) Coletar links de itens (gera data/items_links.csv)
python get_tibiaitens_links.py

# (B) Extrair informações de cada item (gera data/items_info.csv)
python get_tibiaitens_info.py

# (C) Pré-processar (gera data/items_info_preprocessed.csv)
python preprocess_csv.py

# (D) Vetorizar (gera BoW/TF-IDF/SBERT em data/data_vectorizer/)
python vectorizer_items.py

# (E) Analisar (clusters, PCA, similaridade → data/data_analyze_vectorizer/)
python analyze_vectorizer_items.py
```

### Exemplos de Uso (recomendações)
No `analyze_vectorizer_items.py`, ajuste as chamadas de exemplo:
- **Por nome:** `recommend_items_by_name("Alicorn Quiver", df, vectors, top_k=5)`  
- **Por consulta:** `search_items("arco mágico para paladin", df, vectors, top_k=10)`  
- **Substitutos (mesma Voc + |ΔLvl| <= tolerância):** `recommend_substitute_items("Alicorn Quiver", df, vectors, top_k=5)`

---

## 📦 Artefatos Gerados

- **Vetores (entrada da análise)** → `data/data_vectorizer/`  
  `bow_matrix.csv`, `tfidf_matrix.csv`, `sbert_embeddings.npy`, `metadata.json`

- **Análises** → `data/data_analyze_vectorizer/`  
  `cluster_labels.csv` (rótulos K-Means), `cluster_pca2d.csv` (PCA 2D), `similarity_sbert.csv` (matriz de similaridade)

---

## 🛠️ Troubleshooting

- **SBERT não roda / baixa modelo** → precisa de internet na 1ª execução.  
- **Erro no KMeans (`n_init="auto"`)** → troque para `n_init=10` ou garanta `scikit-learn >= 1.4`.  
- **CSV “tudo em uma coluna”** → faltou `sep=";"` no `read_csv`.  
- **Selenium/driver** → use `webdriver-manager` ou configure o ChromeDriver manualmente.

---

## 📚 Referências

- Sentence-Transformers (SBERT) — buscas semânticas multilíngues  
- scikit-learn — K-Means, PCA, vetorizadores (BoW/TF-IDF)  
- NLTK — tokenização, stopwords  
- BeautifulSoup / Selenium — scraping

---

## 📋 Requirements (para `requirements.txt`)

> Use estas versões mínimas (ou fixe com `==` se quiser reprodutibilidade forte).

```
# Core
numpy>=1.26
scipy>=1.11
pandas>=2.0
scikit-learn>=1.4

# NLP
nltk>=3.9
sentence-transformers>=2.6

# Scraping
requests>=2.32
beautifulsoup4>=4.12
selenium>=4.22
webdriver-manager>=4.0

# Visualização (se usar gráficos/wordcloud em EDA/relatório)
matplotlib>=3.8
wordcloud>=1.9

# Utilitários
tqdm>=4.66
```

**Instalação direta via pip (alternativa):**
```bash
pip install numpy scipy pandas scikit-learn nltk sentence-transformers             requests beautifulsoup4 selenium webdriver-manager             matplotlib wordcloud tqdm
```

**Recursos NLTK (rodar uma vez):**
```bash
python - << 'PY'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
PY
```
