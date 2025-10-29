# 🧙‍♂️ Projeto de Processamento de Linguagem Natural (PLN) - TibiaWiki Itens  

Este repositório contém o código e os experimentos desenvolvidos para o trabalho da disciplina de **Processamento de Linguagem Natural (PLN)**, utilizando como base de dados a página de **Itens do TibiaWiki** ([link](https://www.tibiawiki.com.br/wiki/Itens)).  

## 📌 Objetivo  
O projeto tem como objetivo **coletar, estruturar e analisar informações de itens do jogo Tibia** a partir do TibiaWiki, explorando tarefas típicas de PLN, como:  
- Classificação de itens por categoria (armas, armaduras, consumíveis, etc.);  
- Extração de atributos (peso, preço, voc, descrição, requisitos, uso, etc.);  
- Recuperação de informações por meio de buscas;  
- Monitoramento de mudanças, considerando que os itens podem ser atualizados conforme novas versões do jogo.  

## 🛠️ Tecnologias Utilizadas  
* [Python](https://www.python.org)
* [Selenium](https://selenium.dev/)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [pandas](https://pandas.pydata.org/)
* [requests](https://docs.python-requests.org/)
* [nltk](https://github.com/nltk/nltk)

## 🛠️ Resumo do pipeline  (Passo a Passo)
- Coleta de dados (Scraping): get_items_links.py 
- Extração: get_tibiaitens_info.py
- Pré-processamento: preprocess_csv.py
- Vetorização (BoW, TF-IDF, SBERT): vectorizer_items.py
- Análise/Exploração: analyze_items.py 
- Classificação por recomendação (Itens por vocação): recommendation_items.py

## 🗂️ Estrutura do Projeto
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
