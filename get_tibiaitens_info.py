"""
----------------------
Driver para extrair dados de tabelas HTML do TibiaWiki usando Selenium WebDriver.
Lê data/items_links.csv, abre cada URL e extrai dados de todas as tabelas encontradas.
Cria novas colunas automaticamente conforme necessário (tabela_origem, url_origem, etc.).

Saída: data/items_info.csv com dados estruturados das tabelas
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

def get_item_urls() -> Dict:
    INPUT_CSV = os.path.join("data", "items_links.csv")
    df = pd.read_csv(INPUT_CSV)
    return df["url"].tolist()


def process_single_url(url: str) -> pd.DataFrame:
    """
    Processa uma única URL com seu próprio driver Selenium.
    Retorna DataFrame com os dados extraídos da tabela tabelaDPL.
    """
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
        print(f"[INFO] Processando: {url}")
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Encontra a tabela tabelaDPL
        try:
            table = driver.find_element(By.ID, "tabelaDPL")
            print(f"[OK] Tabela tabelaDPL encontrada!")

            # Extrai cabeçalhos
            headers = []
            header_rows = table.find_elements(By.TAG_NAME, "th")
            if header_rows:
                headers = [th.text.strip() for th in header_rows]

            # Extrai linhas de dados
            rows = table.find_elements(By.TAG_NAME, "tr")
            table_data = []

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if cells:
                    row_data = [cell.text.strip() for cell in cells]
                    # Adiciona metadados
                    row_data.insert(0, "tabela_1")
                    row_data.insert(1, url)
                    table_data.append(row_data)

            if table_data:
                # Cria DataFrame
                if headers:
                    df_columns = ["tabela_origem", "url_origem"] + headers
                else:
                    max_cols = max(len(row) for row in table_data) if table_data else 2
                    df_columns = ["tabela_origem", "url_origem"] + [f"coluna_{i+1}" for i in range(max_cols - 2)]

                df_table = pd.DataFrame(table_data, columns=df_columns[:len(table_data[0])])
                print(f"[OK] Extraídos {len(df_table)} registros da tabela tabelaDPL")
                return df_table
            else:
                print(f"[WARN] Nenhum dado encontrado na tabela")
                return pd.DataFrame()

        except Exception as e:
            print(f"[ERROR] Erro ao encontrar/processar tabela tabelaDPL: {e}")
            return pd.DataFrame()

    finally:
        driver.quit()


def main():
    # Arquivo de saída para os dados extraídos das tabelas
    OUTPUT_CSV = os.path.join("data", "items_info.csv")

    # Cria diretório data se não existir
    os.makedirs("data", exist_ok=True)

    item_urls = get_item_urls()
    print(f"[INFO] Iniciando extração de {len(item_urls)} URLs com drivers individuais...")

    all_extracted_data = []

    for i, url in enumerate(item_urls, 1):
        try:
            # Processa cada URL com seu próprio driver
            df_extracted = process_single_url(url)

            if not df_extracted.empty:
                all_extracted_data.append(df_extracted)
            else:
                print(f"[WARN] Nenhum dado extraído da URL {i}")

            # Pequena pausa para não sobrecarregar o servidor
            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] Erro ao processar URL {i}: {e}")
            continue

    # Combina todos os dados extraídos
    if all_extracted_data:
        final_df = pd.concat(all_extracted_data, ignore_index=True)

        # Salva em CSV
        final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"[OK] Dados salvos em: {OUTPUT_CSV}")
        print(f"[INFO] Total de registros extraídos: {len(final_df)}")
        print(f"[INFO] Colunas criadas: {list(final_df.columns)}")
    else:
        print("[WARN] Nenhum dado foi extraído de nenhuma URL")

    print("[OK] Processo concluído.")



if __name__ == "__main__":
    main()