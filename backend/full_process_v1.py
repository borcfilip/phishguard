"""
phishing_pipeline.py
====================
Uceleny pipeline: URL -> extrakcia 50 crt -> LightGBM predikcia -> LIME vysvetlenie

Pouzitie:
    python phishing_pipeline.py
    -> interaktivny rezim, zadaj URL

Poziadavky (pip):
    lightgbm, lime, scikit-learn, joblib, requests,
    beautifulsoup4, lxml, tldextract
"""

import requests
import joblib
import json
import os
import re
import socket
import time
import warnings
import numpy as np

from collections import OrderedDict, Counter
from datetime import datetime
from urllib.parse import urlparse, unquote

import tldextract
from bs4 import BeautifulSoup

import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')

# =====================================================================
# KONFIGURÁCIA – uprav podľa seba
# =====================================================================
WHOIS_API_KEY      = "at_xPSrF5c0VXbvO9laEt4eNPGBYrhoD"
PAGERANK_KEY       = "80cs088swsc04wsgckgcogkk0ccgg4sc4wc4gow4"
DATAFORSEO_LOGIN   = "20filipborc03@gmail.com"   # ← zmeň na svoje DataForSEO prihlasovacie meno
DATAFORSEO_PASSWORD= "00dcf5cbcb143076"      # ← zmeň na svoje DataForSEO heslo
CACHE_FILE         = "api_cache.json"
MODEL_PATH         = "lgbm_detector_full.pkl"
LIME_DATA_PATH     = "lime_training_data_full.pkl"

# =====================================================================
# PORADIE 50 ČŔRT – musí byť totožné s COLUMNS_TO_KEEP v train2.py
# =====================================================================
FEATURES = [
    'length_url', 'length_words_raw', 'nb_eq', 'nb_and', 'nb_qm', 'nb_dots',
    'ip', 'ratio_digits_url', 'longest_word_path', 'nb_slash', 'nb_underscore',
    'longest_words_raw', 'http_in_path', 'nb_com', 'char_repeat', 'tld_in_subdomain',
    'nb_subdomains', 'nb_semicolumn', 'avg_word_path', 'google_index', 'phish_hints',
    'nb_colon', 'tld_in_path', 'ratio_digits_host', 'shortest_words_raw',
    'brand_in_path', 'length_hostname', 'page_rank', 'nb_external_redirection',
    'nb_dslash', 'abnormal_subdomain', 'avg_words_raw', 'statistical_report',
    'prefix_suffix', 'nb_at', 'nb_hyphens', 'domain_in_title', 'nb_percent',
    'dns_record', 'nb_hyperlinks', 'external_favicon', 'ratio_extRedirection',
    'web_traffic', 'domain_age', 'domain_in_brand', 'nb_www', 'empty_title',
    'ratio_extMedia', 'brand_in_subdomain', 'whois_registered_domain'
]

# =====================================================================
# SLOVNÍKY
# =====================================================================
BRAND_LIST = [
    'google', 'facebook', 'microsoft', 'apple', 'amazon', 'netflix',
    'paypal', 'ebay', 'linkedin', 'instagram', 'wikipedia', 'outlook',
    'office', 'microsoftonline', 'tatrabanka', 'slsp', 'vub', 'csob',
    'revolut', 'binance', 'coinbase', 'blockchain', 'dropbox', 'adobe',
    'spotify', 'twitter', 'whatsapp', 'icloud', 'gmail', 'yahoo',
    '365bank', 'postovabanka', 'primabanka', 'unicredit', 'metamask',
    'trustwallet', 'ledger', 'kraken', 'crypto'
]

PHISH_HINTS = [
    'login', 'signin', 'verify', 'account', 'update', 'secure', 'webspace',
    'customer', 'support', 'billing', 'banking', 'confirm', 'password',
    'security', 'wallet', 'service', 'limited', 'help', 'portal',
    'verification', 're-verify', 'sign-in', 'authenticate', 'validation',
    'suspension', 'recover', 'restore', 'authorize', 'credential', 'token',
    'otp', '2fa', 'mfa', 'alert', 'notice', 'locked', 'suspended',
    'expired', 'urgent', 'immediate'
]

# =====================================================================
# GOOGLE INDEX CHECKER – DataForSEO API
# =====================================================================
class GoogleIndexChecker:
    """
    Overuje či je doména indexovaná v Google pomocou DataForSEO SERP API.
    Posiela dopyt "site:<domain>" a kontroluje počet organických výsledkov.

    DataForSEO dokumentácia:
        https://docs.dataforseo.com/v3/serp/google/organic/live/advanced/

    Cenník (orientačný): ~$0.0025 per task (Live Advanced endpoint).
    Pre úsporu credits je výsledok cachovaný v api_cache.json.
    """

    _API_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

    @classmethod
    def _call_api(cls, domain: str) -> int:
        """
        Zavolá DataForSEO SERP API s dopytom "site:<domain>".

        Vracia:
            1  – doména je indexovaná (>= 1 organický výsledok)
            0  – doména NIE je indexovaná alebo nastala chyba API
        """
        import base64

        credentials = base64.b64encode(
            f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}".encode()
        ).decode()

        payload = [
            {
                "keyword":       f"site:{domain}",
                "location_code": 2840,   # USA – najširší Google index
                "language_code": "en",
                "device":        "desktop",
                "os":            "windows",
                "depth":         10,     # 1 stránka výsledkov – stačí nám vedieť či vôbec existuje
                "se_domain":     "google.com",
            }
        ]

        try:
            resp = requests.post(
                cls._API_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type":  "application/json",
                },
                json=payload,
                timeout=15,
            )
            data = resp.json()

            if resp.status_code != 200:
                print(f"[google_index] DataForSEO HTTP {resp.status_code}: "
                      f"{data.get('status_message', '')}")
                return 0

            tasks = data.get("tasks", [])
            if not tasks:
                print(f"[google_index] DataForSEO: prázdna odpoveď pre {domain}")
                return 0

            task      = tasks[0]
            task_code = task.get("status_code", 0)

            if task_code != 20000:
                print(f"[google_index] DataForSEO task error {task_code}: "
                      f"{task.get('status_message', '')}")
                return 0

            results = task.get("result", []) or []
            if not results:
                print(f"[google_index] DataForSEO: žiadne výsledky pre site:{domain}")
                return 0

            result_item = results[0]
            items_count = result_item.get("items_count", 0) or 0
            se_results  = result_item.get("se_results_count", 0) or 0

            # Indexovaná = Google vrátil aspoň 1 výsledok
            indexed = 1 if (items_count > 0 or se_results > 0) else 0
            print(f"[google_index] DataForSEO: {domain} → {indexed} "
                  f"(items={items_count}, se_results={se_results})")
            return indexed

        except Exception as e:
            print(f"[google_index] DataForSEO chyba: {e}")
            return 0

    @classmethod
    def check(cls, domain: str, cache: dict) -> int:
        """
        Hlavná metóda – vracia 1 (indexovaná) alebo 0 (neindexovaná).
        Výsledok sa cachuje, API sa volá len raz na unikátnu doménu.
        """
        if domain in cache and 'google_index' in cache[domain]:
            print(f"[google_index] Cache hit: {domain}")
            return cache[domain]['google_index']

        return cls._call_api(domain)

    @classmethod
    def quit(cls):
        """Kompatibilita so zvyškom pipeline – DataForSEO nepotrebuje cleanup."""
        pass


# =====================================================================
# POMOCNÁ FUNKCIA – parsing WHOIS dátumu
# =====================================================================
def parse_whois_date(date_str):
    if not date_str:
        return None
    formats = [
        '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d', '%d-%b-%Y', '%Y/%m/%d', '%d.%m.%Y',
    ]
    clean = date_str.strip()
    for fmt in formats:
        try:
            return datetime.strptime(clean[:20].strip(), fmt)
        except ValueError:
            continue
    return None


# =====================================================================
# EXTRAKTOR – 50 ČŔRT
# =====================================================================
class PhishingExtractorComplete:

    def __init__(self, url):
        self.raw_url = url.strip()
        self.url = unquote(self.raw_url).lower()
        self.parsed_obj = urlparse(self.url)
        self.host = self.parsed_obj.netloc
        self.path = self.parsed_obj.path
        self.tld_info = tldextract.extract(self.url)
        self.main_reg_domain = self.tld_info.registered_domain
        self.features = {}
        self.extracted_tokens = []

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self, data):
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    # ------------------------------------------------------------------
    def extract_lexical(self):
        f = self.features
        u = self.url
        host = self.host

        f['length_url']      = len(self.raw_url)
        f['length_hostname'] = len(host)

        clean_host = host.split(':')[0]
        f['ip'] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", clean_host) else 0

        f['nb_dots']      = u.count('.')
        f['nb_hyphens']   = u.count('-')
        f['nb_at']        = u.count('@')
        f['nb_qm']        = u.count('?')
        f['nb_and']       = u.count('&')
        f['nb_eq']        = u.count('=')
        f['nb_underscore']= u.count('_')
        f['nb_percent']   = u.count('%')
        f['nb_slash']     = u.count('/')
        f['nb_colon']     = u.count(':')
        f['nb_semicolumn']= u.count(';')
        f['nb_www']       = host.lower().count('www')
        f['nb_com']       = u.count('.com')

        url_no_proto = re.sub(r'^https?://', '', u)
        f['nb_dslash']    = url_no_proto.count('//')
        f['http_in_path'] = 1 if re.search(r'https?://', self.path) else 0

        f['ratio_digits_url']  = sum(c.isdigit() for c in u) / len(u) if u else 0
        f['ratio_digits_host'] = sum(c.isdigit() for c in host) / len(host) if host else 0

        common_tlds = ['com','org','net','sk','gov','edu','info','biz','xyz','top']
        f['tld_in_path']      = 1 if any(f'.{t}' in self.path for t in common_tlds) else 0
        f['tld_in_subdomain'] = 1 if any(f'.{t}' in self.tld_info.subdomain for t in common_tlds) else 0

        sub = self.tld_info.subdomain
        parts = sub.split('.') if sub else []
        f['abnormal_subdomain'] = 1 if (
            len(sub) > 25 or len(parts) > 3 or any(len(p) > 15 for p in parts)
        ) else 0

        f['nb_subdomains'] = host.count('.')
        f['prefix_suffix'] = 1 if '-' in host else 0

        # Tokenizácia
        clean_text = re.sub(r'^https?://', '', u)
        tld = self.tld_info.suffix
        if tld:
            domain_core = host.replace(f".{tld}", "")
            clean_text = domain_core + self.path + self.parsed_obj.query
        self.extracted_tokens = [w for w in re.split(r'[./\-_?=&:%]', clean_text) if w]

        word_lengths = [len(w) for w in self.extracted_tokens] if self.extracted_tokens else [0]
        p_tokens     = [w for w in re.split(r'[./\-_?=&:%]', self.path) if w]
        path_lengths = [len(w) for w in p_tokens] if p_tokens else [0]

        f['length_words_raw']  = len(self.extracted_tokens)
        f['shortest_words_raw']= min(word_lengths)
        f['longest_words_raw'] = max(word_lengths)
        f['avg_words_raw']     = sum(word_lengths) / len(word_lengths)
        f['longest_word_path'] = max(path_lengths)
        f['avg_word_path']     = sum(path_lengths) / len(path_lengths)

        counts = Counter(u)
        f['char_repeat'] = max(counts.values()) if counts else 0

        f['phish_hints']       = sum(1 for h in PHISH_HINTS if h in u)
        f['domain_in_brand']   = 1 if any(b in self.tld_info.domain    for b in BRAND_LIST) else 0
        f['brand_in_subdomain']= 1 if any(b in self.tld_info.subdomain for b in BRAND_LIST) else 0
        f['brand_in_path']     = 1 if any(b in self.path               for b in BRAND_LIST) else 0

    # ------------------------------------------------------------------
    def extract_external(self):
        f = self.features
        f.update({'domain_age': 0, 'whois_registered_domain': 0,
                  'page_rank': 0, 'web_traffic': 0})

        cache  = self._load_cache()
        domain = self.main_reg_domain

        if domain in cache and 'domain_age' in cache[domain]:
            for k in ['domain_age', 'whois_registered_domain', 'page_rank', 'web_traffic']:
                f[k] = cache[domain].get(k, 0)
        else:
            whois_ok = False

            # WHOIS
            try:
                res = requests.get(
                    f"https://www.whoisxmlapi.com/whoisserver/WhoisService"
                    f"?apiKey={WHOIS_API_KEY}&domainName={domain}&outputFormat=JSON",
                    timeout=5
                ).json()
                c_date = res.get('WhoisRecord', {}).get('createdDate', '')
                if c_date:
                    dt = parse_whois_date(c_date)
                    if dt:
                        f['whois_registered_domain'] = 1
                        f['domain_age'] = (datetime.now() - dt).days
                        whois_ok = True
            except Exception as e:
                print(f"[WHOIS] Chyba: {e}")

            # PageRank
            try:
                r = requests.get(
                    f"https://openpagerank.com/api/v1.0/getPageRank?domains[]={domain}",
                    headers={'API-OPR': PAGERANK_KEY}, timeout=5
                ).json()
                if r.get('status_code') == 200:
                    entry = r['response'][0]
                    f['page_rank']   = entry.get('page_rank_integer', 0)
                    f['web_traffic'] = entry.get('rank', 0)
            except Exception as e:
                print(f"[PageRank] Chyba: {e}")

            cache.setdefault(domain, {})
            cache[domain].update({k: f[k] for k in
                ['domain_age', 'whois_registered_domain', 'page_rank', 'web_traffic']})
            self._save_cache(cache)

        # statistical_report – 0 ak API zlyhalo
        api_ok = (f['whois_registered_domain'] == 1 and f['page_rank'] > 0)
        f['statistical_report'] = 1 if (
            api_ok and f['domain_age'] < 180 and f['page_rank'] < 2
        ) else 0

        socket.setdefaulttimeout(3)
        try:
            socket.gethostbyname(self.host.split(':')[0])
            f['dns_record'] = 0
        except:
            f['dns_record'] = 1

    # ------------------------------------------------------------------
    def google_index_check(self):
        cache  = self._load_cache()
        domain = self.main_reg_domain
        result = GoogleIndexChecker.check(domain, cache)
        self.features['google_index'] = result
        cache.setdefault(domain, {})
        cache[domain]['google_index'] = result
        self._save_cache(cache)

    # ------------------------------------------------------------------
    def extract_content(self):
        f = self.features
        defaults = {
            'nb_hyperlinks': 0, 'ratio_extRedirection': 0,
            'ratio_extMedia': 0, 'empty_title': 1,
            'domain_in_title': 0, 'external_favicon': 0,
            'nb_external_redirection': 0,
        }
        try:
            res  = requests.get(self.raw_url, timeout=5,
                                headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(res.text, 'lxml')
        except:
            f.update(defaults)
            return

        title = (soup.title.string.lower()
                 if soup.title and soup.title.string else "").strip()
        f['empty_title']     = 1 if not title else 0
        f['domain_in_title'] = 1 if self.tld_info.domain.lower() in title else 0

        links     = [a['href'] for a in soup.find_all('a', href=True)]
        ext_links = [l for l in links
                     if l.startswith(('http', '//'))
                     and tldextract.extract(l).registered_domain != self.main_reg_domain]
        f['nb_hyperlinks']        = len(links)
        f['nb_external_redirection'] = len(ext_links)
        f['ratio_extRedirection'] = len(ext_links) / len(links) if links else 0

        media   = soup.find_all(['img', 'video', 'embed'], src=True)
        ext_m   = sum(1 for m in media
                      if m['src'].startswith(('http', '//'))
                      and tldextract.extract(m['src']).registered_domain != self.main_reg_domain)
        f['ratio_extMedia'] = ext_m / len(media) if media else 0

        fav = soup.find('link', rel=re.compile(r'icon', re.I))
        if fav and fav.get('href'):
            fav_domain = tldextract.extract(fav['href']).registered_domain
            f['external_favicon'] = 1 if (
                fav['href'].startswith(('http', '//'))
                and fav_domain and fav_domain != self.main_reg_domain
            ) else 0
        else:
            f['external_favicon'] = 0

    # ------------------------------------------------------------------
    def run(self):
        self.extract_lexical()
        self.extract_external()
        self.google_index_check()
        self.extract_content()

        # Vektor v poradí ktoré pozná model (FEATURES)
        vector = OrderedDict((k, self.features.get(k, 0)) for k in FEATURES)
        return vector, self.extracted_tokens


# =====================================================================
# LIME EXPLAINER – inicializovaný raz pri importe
# =====================================================================
class PhishingExplainer:
    """
    Obaluje LightGBM model a LIME explainer.
    LIME potrebuje trénovacie dáta – načíta ich z lime_training_data.pkl
    ktorý vytvoríš raz pomocou save_lime_data().
    """
    def __init__(self, model_path=MODEL_PATH, lime_data_path=LIME_DATA_PATH):
        # Načítanie modelu
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model nenájdený: {model_path}")
        self.model = joblib.load(model_path)
        print(f"[pipeline] Model načítaný: {model_path}")

        # Načítanie LIME trénovacích dát
        if not os.path.exists(lime_data_path):
            raise FileNotFoundError(
                f"LIME dáta nenájdené: {lime_data_path}\n"
                f"Spusti najprv save_lime_data() na vytvorenie tohto súboru."
            )
        X_train = joblib.load(lime_data_path)
        print(f"[pipeline] LIME trénovacie dáta načítané: {X_train.shape}")

        # Inicializácia LIME – stačí raz
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=FEATURES,
            class_names=['legitimate', 'phishing'],
            mode='classification',
            random_state=42
        )
        print("[pipeline] LIME explainer pripravený.")

    def predict(self, feature_vector: dict) -> dict:
        """
        Vstup:  feature_vector – OrderedDict z PhishingExtractorComplete.run()
        Výstup: dict s predikciou a LIME vysvetlením
        """
        # Numpy vektor v správnom poradí
        X = np.array([feature_vector[f] for f in FEATURES], dtype=float).reshape(1, -1)

        # Predikcia
        pred_label = int(self.model.predict(X)[0])
        pred_proba = self.model.predict_proba(X)[0]
        confidence = float(pred_proba[pred_label])

        # LIME lokálna analýza
        exp = self.lime_explainer.explain_instance(
            data_row=X[0],
            predict_fn=self.model.predict_proba,
            num_features=8,      # top 8 čŕt – dostatok pre popup
            num_samples=1000     # rýchlosť vs. presnosť (pre 50 URL stačí)
        )

        # LIME vracia podmienky ako "page_rank > 5.00" alebo
        # "0.00 < domain_age <= 180.00" – vyextrahujeme čistý názov črty
        def extract_feature_name(lime_label):
            for feat in FEATURES:
                if feat in lime_label:
                    return feat
            return lime_label  # fallback ak sa nenájde

        # Formátovanie výstupu
        explanation = [
            {
                "feature":   extract_feature_name(lime_label),
                "condition": lime_label,   # pôvodná podmienka – užitočná pre dokumentáciu
                "value":     round(float(feature_vector.get(extract_feature_name(lime_label), 0)), 4),
                "impact":    round(float(impact), 4)
            }
            for lime_label, impact in exp.as_list()
        ]

        # Zoradenie: najväčší vplyv (absolútna hodnota) na vrch
        explanation.sort(key=lambda x: abs(x['impact']), reverse=True)

        return {
            "prediction":  "phishing" if pred_label == 1 else "legitimate",
            "confidence":  round(confidence * 100, 1),  # napr. 94.3
            "explanation": explanation,
        }


# =====================================================================
# UTILITA – uloženie LIME trénovacích dát (spusti raz)
# =====================================================================
def save_lime_data(dataset_path: str, separator: str = ';',
                   n_samples: int = 1000,
                   output_path: str = LIME_DATA_PATH):
    """
    Načíta dataset, vyberie náhodných n_samples riadkov a uloží
    numpy array pre LIME inicializáciu.

    Spusti PRED prvým použitím pipeline:
        from phishing_pipeline import save_lime_data
        save_lime_data('C:/cesta/k/datasetu.csv')
    """
    import pandas as pd
    df = pd.read_csv(dataset_path, sep=separator)
    X  = df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    X_sample = X.sample(min(n_samples, len(X)), random_state=42)
    joblib.dump(X_sample.values, output_path)
    print(f"[save_lime_data] Ulozene {len(X_sample)} riadkov -> {output_path}")


# =====================================================================
# HLAVNÁ FUNKCIA – spustenie pre jednu URL
# =====================================================================
def analyze_url(url: str, explainer: PhishingExplainer,
                verbose: bool = True) -> dict:
    """
    Kompletná analýza jednej URL.
    Vracia dict s predikciou, confidence a LIME vysvetlením.
    """
    print(f"\n{'='*60}")
    print(f"Analyzujem: {url[:80]}{'...' if len(url)>80 else ''}")
    print('='*60)

    # 1. Extrakcia čŕt
    extractor = PhishingExtractorComplete(url)
    feature_vector, tokens = extractor.run()

    # 2. Predikcia + LIME
    result = explainer.predict(feature_vector)

    # 3. Výpis
    if verbose:
        label = result['prediction'].upper()
        conf  = result['confidence']

        print(f"\nVYSLEDOK : {label}")
        print(f"ISTOTA   : {conf}%")
        print(f"\nTOP VLASTNOSTI (LIME):")
        print(f"  {'Vlastnost':<25} {'Podmienka':<35} {'Hodnota':>8} {'Vplyv':>10}")
        print(f"  {'-'*80}")
        for e in result['explanation']:
            smer = '+' if e['impact'] > 0 else '-'
            print(f"  {e['feature']:<25} {e['condition']:<35} {e['value']:>8.3f} {smer}{abs(e['impact']):>9.4f}")

        print(f"\nTokeny ({len(tokens)}): {tokens[:8]}{'...' if len(tokens)>8 else ''}")

    result['features'] = dict(feature_vector)
    return result


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    # --- Krok 0: prvýkrát spusti save_lime_data() ---
    # save_lime_data('C:/Users/20fil/OneDrive/Desktop/bakalarka/dataset1_bodkociarka.csv')

    # --- Krok 1: inicializácia (raz na začiatku) ---
    try:
        explainer = PhishingExplainer()
    except FileNotFoundError as e:
        print(f"\nCHYBA: {e}")
        raise SystemExit(1)

    # --- Krok 2: interaktívna slučka ---
    test_urls = [
        # Pridaj URL ktoré chceš otestovať
        "https://www.highspeedinternet.com/ny/new-york",
    ]

    results = []
    for url in test_urls:
        try:
            r = analyze_url(url, explainer, verbose=True)
            results.append({"url": url, **r})
        except Exception as e:
            print(f"Chyba pre {url}: {e}")

    # Voliteľne: uložiť výsledky do JSON
    with open("pipeline_results.json", "w", encoding="utf-8") as f:
        # features sú veľké, vynechaj ich z JSON výstupu ak nepotrebuješ
        out = [{k: v for k, v in r.items() if k != 'features'} for r in results]
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nVysledky ulozene -> pipeline_results.json")

    # Vždy zatvoriť Chrome driver
    GoogleIndexChecker.quit()
