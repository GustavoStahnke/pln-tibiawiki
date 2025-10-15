import csv
import os
import re
import time
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

BASE_URL = "https://www.tibiawiki.com.br"
START_PATH = "/wiki/Itens"
START_URL = urljoin(BASE_URL, START_PATH)

OUTPUT_DIR = os.path.join("data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "items_links.csv")

# Namespaces do MediaWiki que NÃO são itens (evitar coletar)
EXCLUDED_PREFIXES = (
    "Categoria:", "Arquivo:", "Ficheiro:", "Fórum:", "Ajuda:", "Wikipédia:",
    "MediaWiki:", "Especial:", "Template:", "Modelo:", "Módulo:", "Discussão:",
    "Predefinição:", "Usuário:", "User:", "Portal:"
)

def is_item_link(href: str) -> bool:
    if not href:
        return False
    # Deve ser um caminho do /wiki/
    if not href.startswith("/wiki/"):
        return False

    # Ignora âncoras internas
    if "#" in href:
        return False

    # Ignora páginas de namespaces não desejados
    for prefix in EXCLUDED_PREFIXES:
        if href[len("/wiki/"):].startswith(prefix):
            return False

    return True

def collect_links_from_html(html: str) -> set:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if is_item_link(href):
            links.add(urljoin(BASE_URL, href))
    return links

def scroll_to_bottom(driver, pause=0.6, max_tries=8):
    """
    Tenta rolar a página algumas vezes (caso haja conteúdos preguiçosos),
    embora a maioria das páginas do TibiaWiki seja estática.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    tries = 0
    while tries < max_tries:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            tries += 1
        else:
            tries = 0
            last_height = new_height

def main():
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

    try:
        driver.get(START_URL)
        # Aguarda conteúdo principal carregar
        wait.until(EC.presence_of_element_located((By.ID, "content")))

        # (Opcional) rolar um pouco
        scroll_to_bottom(driver)

        html = driver.page_source
        collected = collect_links_from_html(html)

        # Também tentar seguir links de subcategorias/letras, se existirem na página inicial de Itens
        soup = BeautifulSoup(html, "html.parser")
        nav_candidates = []
        for a in soup.select("a[href^='/wiki/']"):
            text = (a.get_text() or "").strip().lower()
            href = a["href"]
            # Heurísticas: links que parecem "Categoria:Itens de X", "Armas", "Armaduras", etc.
            # Mantemos alguns agregadores para expandir a coleta.
            if any(k in text for k in ["arma", "armadura", "escudo", "poção", "anéis", "botas", "capacete", "amuleto", "itens"]):
                if is_item_link(href):
                    nav_candidates.append(urljoin(BASE_URL, href))

        nav_candidates = list(dict.fromkeys(nav_candidates))  # dedup na ordem

        # Visita alguns agregadores para coletar mais links de itens
        for url in nav_candidates[:25]:  # limite de segurança
            try:
                driver.get(url)
                wait.until(EC.presence_of_element_located((By.ID, "content")))
                time.sleep(0.6)
                html2 = driver.page_source
                collected |= collect_links_from_html(html2)
            except Exception:
                # Continua mesmo se algum sublink falhar
                continue
            time.sleep(0.4)

        # Deduplicar e ordenar
        links_sorted = sorted(collected)

        # Salvar CSV
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url"])
            for url in links_sorted:
                writer.writerow([url])

        print(f"[OK] Coletados {len(links_sorted)} links em: {OUTPUT_CSV}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()