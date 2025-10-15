# 🧙‍♂️ Projeto de Processamento de Linguagem Natural (PLN) - TibiaWiki Itens  

Este repositório contém o código e os experimentos desenvolvidos para o trabalho da disciplina de **Processamento de Linguagem Natural (PLN)**, utilizando como base de dados a página de **Itens do TibiaWiki** ([link](https://www.tibiawiki.com.br/wiki/Itens)).  

## 📌 Objetivo  
O projeto tem como objetivo **coletar, estruturar e analisar informações de itens do jogo Tibia** a partir do TibiaWiki, explorando tarefas típicas de PLN, como:  
- Classificação de itens por categoria (armas, armaduras, consumíveis, etc.);  
- Extração de atributos (peso, preço, descrição, requisitos, uso, etc.);  
- Recuperação de informações por meio de buscas;  
- Desenvolvimento de um agente conversacional (chatbot) para responder perguntas sobre os itens;  
- Monitoramento de mudanças, considerando que os itens podem ser atualizados conforme novas versões do jogo.  

## 🛠️ Tecnologias Utilizadas  
* [Python](https://www.python.org)
* [Selenium](https://selenium.dev/)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [pandas](https://pandas.pydata.org/)
* [requests](https://docs.python-requests.org/)
* [nltk](https://github.com/nltk/nltk)


## 🛠️ Resumo do pipeline  
- Coleta de dados (Scraping): get_items_links.py 
- Extração + Pré-processamento: get_tibiaitens_info.py
- Vetorização (TF-IDF): vectorizer_items.py
- Análise/Exploração: analyze_items.py