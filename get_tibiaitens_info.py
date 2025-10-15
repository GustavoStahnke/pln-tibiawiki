"""
----------------------
Lê data/items_links.csv, abre cada página de item do TibiaWiki usando Selenium e
extrai informações (nome, categoria, peso, preço, descrição, uso, requisitos).
Depois aplica PRÉ-PROCESSAMENTO na descrição: tokenização, remoção de stopwords,
normalização, lematização (se spaCy disponível) e stemming (Snowball, sem downloads).

Saída: data/items_info.csv com colunas extras:
  - descricao_limpa
  - descricao_lemmas
  - descricao_stem

Para lematização em PT-BR:
    python3 -m spacy download pt_core_news_sm
"""

import csv
import os
import re
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

# --- PLN (sem downloads do NLTK) ---
import string
try:
    import spacy
    nlp = spacy.load("pt_core_news_sm")
except Exception:
    nlp = None
from nltk.stem.snowball import SnowballStemmer
STEMMER = SnowballStemmer("portuguese")

# Pequena lista de stopwords de fallback (caso spaCy não esteja instalado)
FALLBACK_STOPS = {
    "a","à","às","ao","aos","as","o","os","um","uns","uma","umas",
    "de","da","do","das","dos","d","em","no","na","nos","nas",
    "por","para","pra","com","sem","entre","sobre","contra","desde",
    "e","ou","mas","também","tambem","como","que","se","sua","seu","suas","seus",
    "é","ser","são","está","em","essa","esse","isso","isto","aquele","aquela","aquilo",
}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

INPUT_CSV = os.path.join("data", "items_links.csv")
OUTPUT_DIR = os.path.join("data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "items_info.csv")

# Limite para testes (None = sem limite)
LIMIT = 10
# LIMIT = None

FIELD_ALIASES = {
    "categoria": {"categoria", "tipo", "classificação"},
    "peso": {"peso"},
    "preco": {"preço", "preco", "valor"},
    "uso": {"uso", "usos", "utilização", "utilizacao"},
    "requisitos": {"requisitos", "requerimentos"},
    "descricao": {"descrição", "descricao", "description"},
}

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extract_infobox_dict(soup: BeautifulSoup) -> Dict[str, str]:
    data = {}
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c.lower()) \
              or soup.select_one("table.infobox, table.tabela")
    if not infobox:
        return data
    for tr in infobox.find_all("tr"):
        if tr.find("th") and tr.find_all("td"):
            key = normalize_space(tr.find("th").get_text(separator=" ")).lower()
            val = normalize_space(" ".join(td.get_text(separator=" ") for td in tr.find_all("td")))
            if key and val:
                data[key] = val
    return data

def match_field(infobox: Dict[str, str], target_key: str) -> Optional[str]:
    aliases = FIELD_ALIASES.get(target_key, set())
    for k, v in infobox.items():
        k_clean = k.lower()
        for a in aliases:
            if a in k_clean:
                return v
    return None

def extract_title_and_description(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    title = None
    h1 = soup.find("h1", id="firstHeading")
    if h1:
        title = normalize_space(h1.get_text())
    desc = None
    content = soup.find(id="mw-content-text") or soup.find(id="content")
    if content:
        p = content.find("p")
        if p:
            desc = normalize_space(p.get_text(separator=" "))
    return title, desc

# ------------ Pré-processamento sem downloads NLTK ------------
def simple_tokenize_pt(text: str):
    # pega palavras com letras (inclui acentos) e números separados
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)

def preprocess_text(text: Optional[str]) -> Tuple[str, str, str]:
    """
    Retorna (descricao_limpa, descricao_lemmas, descricao_stem)
    - descricao_limpa: minúsculas, sem pontuação, sem stopwords, tokens unidos por espaço
    - descricao_lemmas: lemas em PT-BR (via spaCy, se disponível)
    - descricao_stem: radicais via Snowball (sem downloads)
    """
    if not text:
        return "", "", ""

    text_low = text.lower().translate(PUNCT_TABLE)
    tokens = simple_tokenize_pt(text_low)

    # stopwords: usa spaCy se disponível; senão fallback pequeno
    if nlp is not None:
        stops = nlp.Defaults.stop_words
    else:
        stops = FALLBACK_STOPS

    tokens_no_stop = [t for t in tokens if t not in stops]

    descricao_limpa = " ".join(tokens_no_stop)

    if nlp is not None:
        doc = nlp(" ".join(tokens_no_stop))
        lemmas = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
        descricao_lemmas = " ".join(lemmas)
    else:
        descricao_lemmas = ""  # se não houver modelo PT, deixa em branco

    stems = [STEMMER.stem(t) for t in tokens_no_stop]
    descricao_stem = " ".join(stems)

    return descricao_limpa, descricao_lemmas, descricao_stem
# -------------------------------------------------------------

def parse_item_page(html: str, url: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title, desc_from_p = extract_title_and_description(soup)
    infobox = extract_infobox_dict(soup)

    categoria = match_field(infobox, "categoria")
    peso = match_field(infobox, "peso")
    preco = match_field(infobox, "preco")
    uso = match_field(infobox, "uso")
    requisitos = match_field(infobox, "requisitos")
    descricao = match_field(infobox, "descricao") or desc_from_p

    descricao_limpa, descricao_lemmas, descricao_stem = preprocess_text(descricao)

    return {
        "url": url,
        "nome": title or "",
        "categoria": categoria or "",
        "peso": peso or "",
        "preco": preco or "",
        "uso": uso or "",
        "requisitos": requisitos or "",
        "descricao": descricao or "",
        "descricao_limpa": descricao_limpa,
        "descricao_lemmas": descricao_lemmas,
        "descricao_stem": descricao_stem,
    }

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {INPUT_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1280,1200")
    chrome_options.add_argument("--lang=pt-BR")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 20)

    rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        urls = [r["url"] for r in reader if r.get("url")]

    if LIMIT is not None:
        urls = urls[:LIMIT]

    total = len(urls)
    for i, url in enumerate(urls, 1):
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.ID, "content")))
            time.sleep(0.5)

            html = driver.page_source
            item = parse_item_page(html, url)
            rows.append(item)

            if i % 5 == 0:
                print(f"[{i}/{total}] Processados...")
        except Exception as e:
            print(f"[ERRO] {url}: {e}")
            continue
        time.sleep(0.3)

    cols = [
        "url", "nome", "categoria", "peso", "preco", "uso", "requisitos", "descricao",
        "descricao_limpa", "descricao_lemmas", "descricao_stem"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Salvo {len(df)} itens em: {OUTPUT_CSV}")

    driver.quit()

if __name__ == "__main__":
    main()