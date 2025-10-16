# preprocess_csv.py
"""
Pré-processamento de texto para o CSV do TibiaWiki.

Passos aplicados em texto_unificado:
1) Normalização (lowercase, remoção de pontuação/caracteres especiais; mantém %, +, -)
2) Tokenização (spaCy)
3) Remoção de stopwords (spaCy + extras)
4) Lematização (spaCy; fallback para tokens se modelo ausente)
5) Stemming (Snowball/Portuguese) — sem downloads do NLTK

Entrada:  data/items_info.csv
Saída:    data/items_info_preprocessed.csv
"""

import os
import re
import warnings
from typing import Iterable, List, Set

import pandas as pd

# ---------- spaCy ----------
import spacy
from spacy.language import Language
from spacy.lang.pt.stop_words import STOP_WORDS as SPACY_STOPWORDS

SPACY_MODEL = "pt_core_news_sm"

def load_nlp() -> Language:
    """Carrega o modelo PT do spaCy; se não houver, usa blank('pt') como fallback."""
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["ner"])
        print(f"[OK] spaCy model carregado: {SPACY_MODEL}")
        return nlp
    except Exception:
        warnings.warn(
            f"Não foi possível carregar '{SPACY_MODEL}'. "
            "Usando spacy.blank('pt') como fallback (lemmas serão os próprios tokens)."
        )
        nlp = spacy.blank("pt")  # tokeniza, mas não tem lemmatizer treinado
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

nlp = load_nlp()

# ---------- Stemming sem corpora (NLTK Snowball) ----------
from nltk.stem.snowball import SnowballStemmer
STEMMER = SnowballStemmer("portuguese")

# ---------- Configurações de IO ----------
INPUT_CSV = os.path.join("data", "items_info.csv")
OUTPUT_CSV = os.path.join("data", "items_info_preprocessed.csv")

# Colunas para não processar
EXCLUDE_COLS = {"tabela_origem", "url_origem"}

# ---------- Normalização ----------
try:
    import regex as re_u  # melhor suporte a Unicode, se disponível
    PAT_ALLOWED = re_u.compile(r"[^\p{L}\p{N}\s%\+\-]+", flags=re_u.UNICODE)
except Exception:
    PAT_ALLOWED = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s%\+\-]+")
SPACE_COLLAPSE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = PAT_ALLOWED.sub(" ", t)          # remove símbolos indesejados
    t = SPACE_COLLAPSE.sub(" ", t).strip()
    return t

# ---------- Stopwords ----------
def build_stopwords(extra: Iterable[str] = ()) -> Set[str]:
    sw = set(w.lower() for w in SPACY_STOPWORDS)
    # extras comuns do seu domínio
    sw.update({"nenhum", "nenhuma", "desconhecido", "ninguém", "min", "oz"})
    sw.update(w.lower() for w in extra)
    return sw

PT_STOPWORDS = build_stopwords()

# ---------- Utilidades ----------
def is_all_numeric(series: pd.Series) -> bool:
    pat = r"\s*[\+\-]?\d+(?:[.,]\d+)?\s*"
    return series.astype(str).str.fullmatch(pat).fillna(False).all()

def spacy_pipe(texts: List[str]):
    return nlp.pipe(texts, batch_size=256)

def preprocess_texts(texts: List[str]):
    """
    Retorna tupla:
      (texto_normalizado, tokens, tokens_sem_stop, lemmas, stems)
    """
    normalized = [normalize_text(t) for t in texts]

    tokens, tokens_ns, lemmas, stems = [], [], [], []

    # Checa se há lemmatizer de verdade
    has_lemmatizer = "lemmatizer" in nlp.pipe_names

    for doc in spacy_pipe(normalized):
        toks = [t.text for t in doc if t.text.strip()]
        tokens.append(toks)

        toks_ns = [t for t in toks if t not in PT_STOPWORDS]
        tokens_ns.append(toks_ns)

        if has_lemmatizer:
            lem = [t.lemma_.lower() for t in doc if t.text.strip() and t.text not in PT_STOPWORDS]
        else:
            lem = [t.lower() for t in toks_ns]  # fallback: sem lemmatizer
        lemmas.append(lem)

        stems.append([STEMMER.stem(t) for t in toks_ns])

    return normalized, tokens, tokens_ns, lemmas, stems

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"Arquivo não encontrado: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    # Detecta colunas textuais (exclui fixas e numéricas puras)
    candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    text_cols = [c for c in candidate_cols if not is_all_numeric(df[c])]

    if not text_cols:
        warnings.warn("Nenhuma coluna textual detectada; salvando cópia com colunas originais.")
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"[OK] Saída (sem alterações): {OUTPUT_CSV}")
        return

    # Campo unificado
    df["texto_unificado"] = df[text_cols].apply(
        lambda row: " ".join(v for v in row if isinstance(v, str) and v.strip()),
        axis=1
    )

    texts = df["texto_unificado"].tolist()
    norm, toks, toks_ns, lemmas, stems = preprocess_texts(texts)

    # Serialização (se preferir JSON, dá pra trocar aqui)
    df["texto_normalizado"] = norm
    df["tokens"] = [" ".join(x) for x in toks]
    df["tokens_sem_stopwords"] = [" ".join(x) for x in toks_ns]
    df["lemmas"] = [" ".join(x) for x in lemmas]
    df["stems"] = [" ".join(x) for x in stems]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("[OK] Pré-processamento concluído.")
    print(f"[INFO] Colunas textuais processadas: {text_cols}")
    print(f"[INFO] Linhas: {len(df)} | Saída: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
