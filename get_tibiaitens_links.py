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

ITENS_TIBIA = [
    "Capacetes",
    "Armaduras",
    "Escudos",
    "Calças",
    "Spellbooks",
    "Botas",
    "Aljavas",
    "Extra Slot",
    "Machados",
    "Clavas",
    "Espadas",
    "Rods",
    "Wands",
    "Antigas Wands e Rods",
    "Distância",
    "Munição",
    "Réplicas de Armas",
    "Punhos",
    "Livros",
    "Prêmios de Eventos",
    "Runas de Decoração",
    "Documentos e Papéis",
    "Dolls e Bears",
    "Decorações",
    "Instrumentos Musicais",
    "Troféus",
    "Itens de Fansites",
    "Comidas",
    "Líquidos",
    "Plantas e Ervas",
    "Produtos de Criaturas",
    "Amuletos e Colares",
    "Anéis",
    "Chaves",
    "Ferramentas",
    "Ferramentas de Cozinha",
    "Fontes de Luz",
    "Itens de Domar",
    "Recipientes",
    "Itens de Addons",
    "Itens de Imbuements",
    "Itens Encantados",
    "Jogos e Diversão",
    "Itens de Quest",
    "Cristais (Itens)",
    "Itens de Festa",
    "Valiosos",
    "Lixos",
    "Runas"
]



def is_item_link(href: str) -> bool:
    if not href:
        return False
    # Deve ser um caminho do /wiki/
    if not href.startswith("/wiki/"):
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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 20)

    try:
        driver.get(START_URL)
        # Aguarda conteúdo principal carregar
        wait.until(EC.presence_of_element_located((By.ID, "content")))

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        collected = set()

        # Coletar apenas links que possuem nome na lista ITENS_TIBIA
        for a in soup.select("a[href^='/wiki/']"):
            text = (a.get_text() or "").strip()
            href = a["href"]

            if any(item_name == text for item_name in ITENS_TIBIA):
                if is_item_link(href):
                    collected.add(urljoin(BASE_URL, href))

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