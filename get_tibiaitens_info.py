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

INPUT_CSV = os.path.join("data", "items_links.csv")
OUTPUT_DIR = os.path.join("data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "items_info.csv")

# 🔧 Limite de páginas para processar
LIMIT = 10
# LIMIT = None

# Labels possíveis encontrados na infobox do TibiaWiki (pt-BR)
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
    """
    Mapeia th->td da infobox em um dicionário {label_lower: value_text}.
    """
    data = {}
    # Infobox comum em wikis
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c.lower())
    if not infobox:
        # fallback: algumas wikis usam outras classes
        infobox = soup.select_one("table.infobox, table.tabela")
    if not infobox:
        return data

    for tr in infobox.find_all("tr"):
        th = tr.find(["th", "td"])
        tds = tr.find_all("td")
        # Caso de linhas chave/valor: TH (chave) + TD (valor)
        if tr.find("th") and tds:
            key = normalize_space(tr.find("th").get_text(separator=" ")).lower()
            val = normalize_space(" ".join(td.get_text(separator=" ") for td in tds))
            if key and val:
                data[key] = val
    return data

def match_field(infobox: Dict[str, str], target_key: str) -> Optional[str]:
    """
    Tenta obter do dicionário da infobox um campo de acordo com aliases.
    """
    aliases = FIELD_ALIASES.get(target_key, set())
    for k, v in infobox.items():
        k_clean = k.lower()
        for a in aliases:
            if a in k_clean:
                return v
    return None

def extract_title_and_description(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    # Título
    title = None
    h1 = soup.find("h1", id="firstHeading")
    if h1:
        title = normalize_space(h1.get_text())

    # Descrição: heurística — primeiro parágrafo de conteúdo após o título
    desc = None
    content = soup.find(id="mw-content-text") or soup.find(id="content")
    if content:
        p = content.find("p")
        if p:
            desc = normalize_space(p.get_text(separator=" "))
    return title, desc

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

    return {
        "url": url,
        "nome": title or "",
        "categoria": categoria or "",
        "peso": peso or "",
        "preco": preco or "",
        "uso": uso or "",
        "requisitos": requisitos or "",
        "descricao": descricao or "",
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

    # 🔧 aplica limite (se definido)
    if LIMIT is not None:
        urls = urls[:LIMIT]

    total = len(urls)
    for i, url in enumerate(urls, 1):
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.ID, "content")))
            time.sleep(0.5)  # gentileza com o servidor

            html = driver.page_source
            item = parse_item_page(html, url)
            rows.append(item)

            if i % 5 == 0:
                print(f"[{i}/{total}] Processados...")
        except Exception as e:
            print(f"[ERRO] {url}: {e}")
            continue
        time.sleep(0.3)

    # Salvar CSV final
    df = pd.DataFrame(rows, columns=["url", "nome", "categoria", "peso", "preco", "uso", "requisitos", "descricao"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Salvo {len(df)} itens em: {OUTPUT_CSV}")

    driver.quit()

if __name__ == "__main__":
    main()
