import argparse
import csv
import hashlib
import http.client
import io
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from fill_no_inhibition_from_pubchem import (
    DELAY_SECONDS,
    SESSION,
    fetch_chembl_molecule,
    fetch_chembl_activities,
    get_pubchem_compound_info,
)
from collect_structured_six_indicators_pubchem_chembl import (
    INDICATORS,
    assay_indicators,
    classify_activity,
)


USER_AGENT = "lgba-foodb-indicator-multisource/1.0"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

RAW_HEADER = "NO inhibition raw"
SOURCE_HEADER = "NO inhibition source"
BINARY_HEADER = "NO inhibition binary"
STRUCTURED_EVIDENCE_ANY = "NO structured evidence any"
STRUCTURED_ACTIVE_ANY = "NO structured active any"
PUBMED_EVIDENCE_ANY = "NO pubmed evidence any"
PUBMED_ACTIVE_ANY = "NO pubmed active any"
PUBMED_PMIDS = "NO pubmed pmids"
PUBMED_SUPPORTIVE_SENTENCE_COUNT = "NO pubmed supportive sentence count"
EVIDENCE_ANY_HEADER = "NO inhibition evidence any"
ACTIVE_ANY_HEADER = "NO inhibition active any"
SOURCE_METHODS_HEADER = "NO inhibition source methods"
QUERY_ROUTE_HEADER = "NO inhibition query route"
CURRENT_INDICATOR = "no"
CURRENT_LABEL = "NO inhibition"
CACHE_NAMESPACE_SUFFIX = "no"
PUBMED_QUERY_TERMS = [
    '"nitric oxide"',
    '"NO production"',
    "iNOS",
    "NOS2",
    '"nitric oxide scavenging"',
]

INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
NO_PATTERNS = [
    re.compile(r"\bnitric oxide\b", re.IGNORECASE),
    re.compile(r"\bNO production\b", re.IGNORECASE),
    re.compile(r"\biNOS\b", re.IGNORECASE),
    re.compile(r"\bNOS2\b", re.IGNORECASE),
    re.compile(r"\bnitrite accumulation\b", re.IGNORECASE),
    re.compile(r"\bnitric oxide scavenging\b", re.IGNORECASE),
]
INDICATOR_SIGNAL_PATTERNS = NO_PATTERNS
INDICATOR_CONFIG = {
    "no": {
        "label": "NO inhibition",
        "prefix": "NO",
        "cache": "no",
        "query_terms": [
            '"nitric oxide"',
            '"NO production"',
            "iNOS",
            "NOS2",
            '"nitric oxide scavenging"',
            '"nitrite accumulation"',
        ],
        "patterns": [
            r"\bnitric oxide\b",
            r"\bNO production\b",
            r"\biNOS\b",
            r"\bNOS2\b",
            r"\bnitrite accumulation\b",
            r"\bnitric oxide scavenging\b",
        ],
    },
    "tnf_alpha": {
        "label": "TNF-alpha inhibition",
        "prefix": "TNF-alpha",
        "cache": "tnf_alpha",
        "query_terms": [
            '"TNF-alpha"',
            '"TNF alpha"',
            '"TNF-a"',
            '"tumor necrosis factor"',
            '"tumour necrosis factor"',
        ],
        "patterns": [
            r"\bTNF[-\s]?alpha\b",
            r"\bTNFalpha\b",
            r"\bTNF[-\s]?a\b",
            r"\btumou?r necrosis factor\b",
        ],
    },
    "il6": {
        "label": "IL-6 inhibition",
        "prefix": "IL-6",
        "cache": "il6",
        "query_terms": ['"IL-6"', "IL6", '"interleukin-6"', '"interleukin 6"'],
        "patterns": [r"\bIL[-\s]?6\b", r"\bIL6\b", r"\binterleukin[-\s]?6\b"],
    },
    "il1_beta": {
        "label": "IL-1beta inhibition",
        "prefix": "IL-1beta",
        "cache": "il1_beta",
        "query_terms": [
            '"IL-1beta"',
            '"IL-1 beta"',
            "IL1B",
            '"interleukin-1beta"',
            '"interleukin 1 beta"',
        ],
        "patterns": [
            r"\bIL[-\s]?1[-\s]?beta\b",
            r"\bIL[-\s]?1beta\b",
            r"\bIL[-\s]?1B\b",
            r"\bIL1B\b",
            r"\binterleukin[-\s]?1[-\s]?beta\b",
        ],
    },
    "ros": {
        "label": "ROS reduction",
        "prefix": "ROS",
        "cache": "ros",
        "query_terms": ['"ROS"', '"reactive oxygen species"', '"oxidative stress"'],
        "patterns": [r"\bROS\b", r"\breactive oxygen species\b", r"\boxidative stress\b"],
    },
    "nfkb": {
        "label": "NF-kB pathway suppression",
        "prefix": "NF-kB",
        "cache": "nfkb",
        "query_terms": ['"NF-kB"', '"NF-kappaB"', "NFKB", '"nuclear factor kappa B"', "p65"],
        "patterns": [
            r"\bNF[-\s]?kB\b",
            r"\bNF[-\s]?kappa[-\s]?B\b",
            r"\bNFkb\b",
            r"\bNFKB\b",
            r"\bnuclear factor[-\s]?kappa[-\s]?B\b",
            r"\bp65\b",
        ],
    },
}
POSITIVE_PATTERNS = [
    re.compile(r"\binhibit", re.IGNORECASE),
    re.compile(r"\bsuppress", re.IGNORECASE),
    re.compile(r"\breduc", re.IGNORECASE),
    re.compile(r"\bdecreas", re.IGNORECASE),
    re.compile(r"\bdown[-\s]?regulat", re.IGNORECASE),
    re.compile(r"\bscavenge", re.IGNORECASE),
    re.compile(r"\battenuat", re.IGNORECASE),
    re.compile(r"\balleviat", re.IGNORECASE),
]
NEGATIVE_PATTERNS = [
    re.compile(r"\bincreas", re.IGNORECASE),
    re.compile(r"\binduc", re.IGNORECASE),
    re.compile(r"\bup[-\s]?regulat", re.IGNORECASE),
    re.compile(r"\bactivat", re.IGNORECASE),
]
TOXICITY_PATTERNS = [
    re.compile(r"\btoxicity\b", re.IGNORECASE),
    re.compile(r"\btoxic\b", re.IGNORECASE),
    re.compile(r"\bexposure\b", re.IGNORECASE),
    re.compile(r"\bcontaminant\b", re.IGNORECASE),
    re.compile(r"\bpollut", re.IGNORECASE),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Broad FooDB indicator annotation using PubChem, ChEMBL, and PubMed."
    )
    parser.add_argument(
        "--indicator",
        default="no",
        choices=sorted(INDICATOR_CONFIG),
        help="Inflammation indicator to annotate.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--inchikey-column", default="moldb_inchikey")
    parser.add_argument("--inchi-column", default="moldb_inchi")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retmax", type=int, default=10)
    parser.add_argument("--requests-per-second", type=float, default=3.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--email", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output CSV and skip already written input rows.",
    )
    return parser.parse_args()


def default_output_path(input_path, indicator):
    path = Path(input_path)
    return path.with_name(f"{path.stem}_{indicator}_multisource{path.suffix}")


def configure_indicator(indicator):
    global RAW_HEADER, SOURCE_HEADER, BINARY_HEADER
    global STRUCTURED_EVIDENCE_ANY, STRUCTURED_ACTIVE_ANY
    global PUBMED_EVIDENCE_ANY, PUBMED_ACTIVE_ANY, PUBMED_PMIDS
    global PUBMED_SUPPORTIVE_SENTENCE_COUNT, EVIDENCE_ANY_HEADER, ACTIVE_ANY_HEADER
    global SOURCE_METHODS_HEADER, QUERY_ROUTE_HEADER
    global CURRENT_INDICATOR, CURRENT_LABEL, CACHE_NAMESPACE_SUFFIX
    global PUBMED_QUERY_TERMS, INDICATOR_SIGNAL_PATTERNS

    config = INDICATOR_CONFIG[indicator]
    CURRENT_INDICATOR = indicator
    CURRENT_LABEL = config["label"]
    CACHE_NAMESPACE_SUFFIX = config["cache"]
    PUBMED_QUERY_TERMS = config["query_terms"]
    INDICATOR_SIGNAL_PATTERNS = [
        re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]
    ]

    label = config["label"]
    prefix = config["prefix"]
    RAW_HEADER = f"{label} raw"
    SOURCE_HEADER = f"{label} source"
    BINARY_HEADER = f"{label} binary"
    STRUCTURED_EVIDENCE_ANY = f"{prefix} structured evidence any"
    STRUCTURED_ACTIVE_ANY = f"{prefix} structured active any"
    PUBMED_EVIDENCE_ANY = f"{prefix} pubmed evidence any"
    PUBMED_ACTIVE_ANY = f"{prefix} pubmed active any"
    PUBMED_PMIDS = f"{prefix} pubmed pmids"
    PUBMED_SUPPORTIVE_SENTENCE_COUNT = f"{prefix} pubmed supportive sentence count"
    EVIDENCE_ANY_HEADER = f"{label} evidence any"
    ACTIVE_ANY_HEADER = f"{label} active any"
    SOURCE_METHODS_HEADER = f"{label} source methods"
    QUERY_ROUTE_HEADER = f"{label} query route"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def open_csv_reader_with_fallback(path):
    last_error = None
    raw_bytes = path.read_bytes()
    for encoding in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            text = raw_bytes.decode(encoding)
            input_file = io.StringIO(text)
            reader = csv.DictReader(input_file)
            _ = reader.fieldnames
            return input_file, reader, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def delay_from_rate(requests_per_second):
    if requests_per_second <= 0:
        raise ValueError("--requests-per-second must be greater than 0.")
    return 1.0 / requests_per_second


def request_text(url, requests_per_second, max_retries):
    last_error = None
    delay_seconds = delay_from_rate(requests_per_second)
    for attempt in range(1, max_retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=90) as response:
                text = response.read().decode("utf-8")
            time.sleep(delay_seconds)
            return text
        except (HTTPError, URLError, TimeoutError, http.client.IncompleteRead, http.client.RemoteDisconnected, OSError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in {400, 404}:
                return ""
            sleep_seconds = min(300, max(5.0, delay_seconds * (2 ** attempt)) + random.uniform(0, 3))
            log(f"Request failed, retrying in {sleep_seconds:.1f}s: {last_error}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Request failed after retries: {url}; last error: {last_error}")


def safe_filename(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80]
    return f"{prefix}_{digest}"


def cached_get(cache_dir, namespace, key, suffix, url, requests_per_second, max_retries):
    directory = Path(cache_dir) / namespace
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_filename(key)}.{suffix}"
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = request_text(url, requests_per_second, max_retries)
    path.write_text(text, encoding="utf-8")
    return text


def split_sentences(text):
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def has_no_signal(text):
    return any(pattern.search(text) for pattern in INDICATOR_SIGNAL_PATTERNS)


def choose_best_structured(records):
    if not records:
        return None
    ic50_records = [
        item for item in records
        if str(item.get("activity_name", "")).upper() == "IC50"
        and item.get("numeric_value") is not None
    ]
    if ic50_records:
        return min(ic50_records, key=lambda item: item["numeric_value"])
    active_records = [item for item in records if item.get("binary") == 1]
    if active_records:
        return active_records[0]
    binary_records = [item for item in records if item.get("binary") in {0, 1}]
    if binary_records:
        return binary_records[0]
    return records[0]


def parse_number(value):
    try:
        return float(str(value or "").strip())
    except (TypeError, ValueError):
        return None


def classify_assays(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON"
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    table = data.get("Table", {})
    columns = table.get("Columns", {}).get("Column", [])
    rows = table.get("Row", [])
    relevant = []
    for row in rows:
        cells = row.get("Cell", [])
        record = {column: cells[index] if index < len(cells) else "" for index, column in enumerate(columns)}
        assay_name = str(record.get("Assay Name", "")).strip()
        activity_name = str(record.get("Activity Name", "")).strip()
        if CURRENT_INDICATOR not in assay_indicators(assay_name, activity_name):
            continue
        binary, rule = classify_activity(
            activity_name,
            record.get("Activity Outcome", ""),
            record.get("Activity Value [uM]", ""),
            "uM",
        )
        if binary is None:
            binary = "Review"
            rule = "Relevant assay found but not classifiable"
        relevant.append(
            {
                "aid": str(record.get("AID", "")).strip(),
                "assay_name": assay_name,
                "activity_name": activity_name,
                "rule": rule,
                "binary": binary,
                "numeric_value": parse_number(record.get("Activity Value [uM]", "")),
            }
        )
    chosen = choose_best_structured(relevant)
    if not chosen:
        return None
    return {
        "raw": chosen.get("rule", ""),
        "source": f"PubChem AID {chosen.get('aid', '')}: {chosen.get('assay_name', '')}",
        "binary": chosen.get("binary", "Review"),
    }


def classify_chembl_activities(molecule_chembl_id):
    activities = fetch_chembl_activities(molecule_chembl_id)
    relevant = []
    for activity in activities:
        assay_name = str(activity.get("assay_description", "")).strip()
        activity_name = str(activity.get("standard_type", "")).strip()
        if CURRENT_INDICATOR not in assay_indicators(assay_name, activity_name):
            continue
        binary, rule = classify_activity(
            activity_name,
            activity.get("activity_comment", ""),
            activity.get("standard_value", ""),
            activity.get("standard_units", ""),
            activity.get("standard_relation", ""),
        )
        if binary is None:
            binary = "Review"
            rule = "Relevant assay found but not classifiable"
        relevant.append(
            {
                "assay_id": str(activity.get("assay_chembl_id", "")).strip(),
                "assay_name": assay_name,
                "activity_name": activity_name,
                "rule": rule,
                "binary": binary,
                "numeric_value": parse_number(activity.get("standard_value", "")),
            }
        )
    chosen = choose_best_structured(relevant)
    if not chosen:
        return None
    return {
        "raw": chosen.get("rule", ""),
        "source": f"ChEMBL {chosen.get('assay_id', '')}: {chosen.get('assay_name', '')}",
        "binary": chosen.get("binary", "Review"),
    }


def classify_pubmed_sentence(sentence, compound_name):
    text = str(sentence or "")
    lowered_name = str(compound_name or "").lower()
    has_name = lowered_name and lowered_name in text.lower()
    has_positive = any(pattern.search(text) for pattern in POSITIVE_PATTERNS)
    has_negative = any(pattern.search(text) for pattern in NEGATIVE_PATTERNS)
    has_toxicity = any(pattern.search(text) for pattern in TOXICITY_PATTERNS)

    if has_positive and not has_negative and has_name:
        if has_toxicity:
            return "supportive_but_toxicity_context"
        return "supportive_antiinflammatory"
    if has_positive and not has_negative:
        return "mixed"
    if has_negative:
        return "proinflammatory_or_activation"
    if has_toxicity:
        return "toxicity_or_exposure_context"
    return "mention_only"


def extract_identifiers(row, args):
    name = str(row.get(args.name_column, "") or "").strip()
    smiles = str(row.get(args.smiles_column, "") or "").strip()
    raw_a = str(row.get(args.inchikey_column, "") or "").strip()
    raw_b = str(row.get(args.inchi_column, "") or "").strip()

    inchikey = ""
    inchi = ""
    for value in [raw_a, raw_b]:
        if not value:
            continue
        if value.startswith("InChI="):
            inchi = value
        elif INCHIKEY_RE.match(value):
            inchikey = value

    return {
        "name": name,
        "smiles": smiles,
        "inchikey": inchikey,
        "inchi": inchi,
    }


def get_pubchem_info_by_smiles(smiles):
    if not smiles:
        return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/property/InChIKey/JSON"
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    prop = props[0]
    return {"cid": prop.get("CID"), "inchikey": prop.get("InChIKey")}


def get_pubchem_info_by_inchi(inchi):
    if not inchi:
        return None
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchi/property/InChIKey/JSON"
    response = SESSION.post(url, data={"inchi": inchi}, timeout=60)
    response.raise_for_status()
    data = response.json()
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    prop = props[0]
    return {"cid": prop.get("CID"), "inchikey": prop.get("InChIKey")}


def fetch_chembl_molecule_by_inchikey(inchikey):
    if not inchikey:
        return None
    params = urlencode({"molecule_structures__standard_inchi_key": inchikey})
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?{params}"
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    molecules = data.get("molecules", [])
    return molecules[0] if molecules else None


def normalize_structured_result(result, route, source_method):
    binary = result.get("binary")
    structured_evidence_any = 1
    structured_active_any = 1 if binary == 1 else 0
    return {
        "raw": result.get("raw", ""),
        "source": result.get("source", ""),
        "binary": binary,
        "structured_evidence_any": structured_evidence_any,
        "structured_active_any": structured_active_any,
        "pubmed_evidence_any": 0,
        "pubmed_active_any": 0,
        "pubmed_pmids": "",
        "pubmed_supportive_sentence_count": 0,
        "evidence_any": 1,
        "active_any": structured_active_any,
        "source_methods": source_method,
        "query_route": route,
    }


def build_not_found(route):
    return {
        "raw": "Not Found",
        "source": "PubChem+ChEMBL+PubMed",
        "binary": "Not Found",
        "structured_evidence_any": 0,
        "structured_active_any": 0,
        "pubmed_evidence_any": 0,
        "pubmed_active_any": 0,
        "pubmed_pmids": "",
        "pubmed_supportive_sentence_count": 0,
        "evidence_any": 0,
        "active_any": 0,
        "source_methods": "",
        "query_route": route,
    }


def build_request_error(route, error):
    result = build_not_found(route)
    result["raw"] = f"Request Error: {error}"
    result["source"] = "RequestError"
    result["query_route"] = route
    return result


def esearch_pubmed_no(name, cache_dir, requests_per_second, max_retries, retmax=10, email="", api_key=""):
    query = f'"{name}" AND (' + " OR ".join(PUBMED_QUERY_TERMS) + ")"
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": retmax,
        "term": query,
        "tool": "lgba_foodb_indicator_multisource",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_EUTILS}/esearch.fcgi?{urlencode(params)}"
    text = cached_get(cache_dir, f"pubmed_esearch_{CACHE_NAMESPACE_SUFFIX}", query, "json", url, requests_per_second, max_retries)
    if not text:
        return []
    data = json.loads(text)
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_pubmed_abstracts(pmids, cache_dir, requests_per_second, max_retries, email="", api_key=""):
    if not pmids:
        return []
    key = ",".join(pmids)
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": key,
        "tool": "lgba_foodb_indicator_multisource",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{NCBI_EUTILS}/efetch.fcgi?{urlencode(params)}"
    xml_text = cached_get(cache_dir, f"pubmed_efetch_{CACHE_NAMESPACE_SUFFIX}", key, "xml", url, requests_per_second, max_retries)
    if not xml_text.strip():
        return []
    root = ET.fromstring(xml_text)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            abstract_parts.append("".join(node.itertext()))
        abstract = " ".join(part.strip() for part in abstract_parts if part and part.strip())
        articles.append({"pmid": pmid, "title": title, "abstract": abstract})
    return articles


def pubmed_no_result(name, cache_dir, requests_per_second, max_retries, retmax=10, email="", api_key=""):
    if not name:
        return None
    pmids = esearch_pubmed_no(name, cache_dir, requests_per_second, max_retries, retmax, email, api_key)
    if not pmids:
        return None
    articles = efetch_pubmed_abstracts(pmids, cache_dir, requests_per_second, max_retries, email, api_key)
    evidence_pmids = []
    supportive_count = 0
    any_evidence = 0
    for article in articles:
        text = " ".join(part for part in [article["title"], article["abstract"]] if part)
        for sentence in split_sentences(text):
            if not has_no_signal(sentence):
                continue
            any_evidence = 1
            if article["pmid"]:
                evidence_pmids.append(article["pmid"])
            evidence_class = classify_pubmed_sentence(sentence, name)
            if evidence_class in {"supportive_antiinflammatory", "supportive_but_toxicity_context"}:
                supportive_count += 1

    if not any_evidence:
        return None

    unique_pmids = sorted(set(evidence_pmids), key=lambda value: int(value) if value.isdigit() else value)
    return {
        "raw": f"PubMed {CURRENT_LABEL}-related literature found",
        "source": "PubMed",
        "binary": "Review" if supportive_count == 0 else 1,
        "structured_evidence_any": 0,
        "structured_active_any": 0,
        "pubmed_evidence_any": 1,
        "pubmed_active_any": 1 if supportive_count > 0 else 0,
        "pubmed_pmids": ";".join(unique_pmids),
        "pubmed_supportive_sentence_count": supportive_count,
        "evidence_any": 1,
        "active_any": 1 if supportive_count > 0 else 0,
        "source_methods": "PubMed",
        "query_route": f"pubmed_name_{CACHE_NAMESPACE_SUFFIX}_query",
    }


def merge_results(structured_result, pubmed_result):
    if structured_result and pubmed_result:
        source_methods = set(filter(None, structured_result["source_methods"].split(";")))
        source_methods.update(filter(None, pubmed_result["source_methods"].split(";")))
        return {
            "raw": structured_result["raw"],
            "source": structured_result["source"],
            "binary": structured_result["binary"],
            "structured_evidence_any": structured_result["structured_evidence_any"],
            "structured_active_any": structured_result["structured_active_any"],
            "pubmed_evidence_any": pubmed_result["pubmed_evidence_any"],
            "pubmed_active_any": pubmed_result["pubmed_active_any"],
            "pubmed_pmids": pubmed_result["pubmed_pmids"],
            "pubmed_supportive_sentence_count": pubmed_result["pubmed_supportive_sentence_count"],
            "evidence_any": 1,
            "active_any": 1 if structured_result["structured_active_any"] or pubmed_result["pubmed_active_any"] else 0,
            "source_methods": ";".join(sorted(source_methods)),
            "query_route": structured_result["query_route"] + f";pubmed_name_{CACHE_NAMESPACE_SUFFIX}_query",
        }
    if structured_result:
        return structured_result
    if pubmed_result:
        return pubmed_result
    return None


def resolve_no_multisource(identifiers, cache_dir, requests_per_second, max_retries, retmax, email, api_key):
    route_parts = []
    pubchem_info = None
    structured_result = None

    if identifiers["smiles"]:
        route_parts.append("pubchem_smiles")
        try:
            pubchem_info = get_pubchem_info_by_smiles(identifiers["smiles"])
            if pubchem_info and pubchem_info.get("cid") is not None:
                result = classify_assays(pubchem_info["cid"])
                if result is not None:
                    structured_result = normalize_structured_result(result, ";".join(route_parts), "PubChem")
        except Exception:
            pass

    if structured_result is None and identifiers["inchi"]:
        route_parts.append("pubchem_inchi")
        try:
            pubchem_info = get_pubchem_info_by_inchi(identifiers["inchi"])
            if pubchem_info and pubchem_info.get("cid") is not None:
                result = classify_assays(pubchem_info["cid"])
                if result is not None:
                    structured_result = normalize_structured_result(result, ";".join(route_parts), "PubChem")
        except Exception:
            pass

    if structured_result is None and identifiers["name"]:
        route_parts.append("pubchem_name")
        try:
            pubchem_info = get_pubchem_compound_info(identifiers["name"])
            if pubchem_info and pubchem_info.get("cid") is not None:
                result = classify_assays(pubchem_info["cid"])
                if result is not None:
                    structured_result = normalize_structured_result(result, ";".join(route_parts), "PubChem")
        except Exception:
            pass

    chembl_inchikey = identifiers["inchikey"] or (pubchem_info.get("inchikey") if pubchem_info else "")
    if structured_result is None and chembl_inchikey:
        route_parts.append("chembl_inchikey")
        try:
            molecule = fetch_chembl_molecule_by_inchikey(chembl_inchikey)
            if molecule and molecule.get("molecule_chembl_id"):
                result = classify_chembl_activities(molecule["molecule_chembl_id"])
                if result is not None:
                    structured_result = normalize_structured_result(result, ";".join(route_parts), "ChEMBL")
        except Exception:
            pass

    if structured_result is None and identifiers["name"]:
        route_parts.append("chembl_name")
        try:
            molecule = fetch_chembl_molecule(identifiers["name"], inchikey=chembl_inchikey or None)
            if molecule and molecule.get("molecule_chembl_id"):
                result = classify_chembl_activities(molecule["molecule_chembl_id"])
                if result is not None:
                    structured_result = normalize_structured_result(result, ";".join(route_parts), "ChEMBL")
        except Exception:
            pass

    pubmed_result = pubmed_no_result(
        identifiers["name"],
        cache_dir,
        requests_per_second,
        max_retries,
        retmax,
        email,
        api_key,
    )
    merged = merge_results(structured_result, pubmed_result)
    if merged is not None:
        return merged
    return build_not_found(";".join(route_parts) if route_parts else "no_query")


def load_existing_output_stats(output_path):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return {
            "rows": 0,
            "evidence_rows": 0,
            "active_rows": 0,
            "unmatched": [],
        }

    stats = {
        "rows": 0,
        "evidence_rows": 0,
        "active_rows": 0,
        "unmatched": [],
    }
    with output_path.open("r", encoding="utf-8-sig", newline="") as existing_file:
        reader = csv.DictReader(existing_file)
        if not reader.fieldnames:
            raise ValueError(f"Existing output has no header row: {output_path}")
        for header in [RAW_HEADER, EVIDENCE_ANY_HEADER, ACTIVE_ANY_HEADER, SOURCE_METHODS_HEADER]:
            if header not in reader.fieldnames:
                raise ValueError(
                    f"Existing output does not match indicator '{CURRENT_INDICATOR}'. "
                    f"Missing expected column: {header}"
                )
        for row in reader:
            stats["rows"] += 1
            if str(row.get(EVIDENCE_ANY_HEADER, "")).strip() == "1":
                stats["evidence_rows"] += 1
            if str(row.get(ACTIVE_ANY_HEADER, "")).strip() == "1":
                stats["active_rows"] += 1
            if str(row.get(EVIDENCE_ANY_HEADER, "")).strip() == "0":
                stats["unmatched"].append(row.get("name", "") or row.get("Name", ""))
    return stats


def main():
    args = parse_args()
    configure_indicator(args.indicator)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(args.input, args.indicator)
    unmatched_path = output_path.with_suffix(".no_multisource_unmatched.txt")
    cache_dir = output_path.parent / f"{output_path.stem}_cache"

    cache = {}
    unmatched = []
    queried_rows = 0
    written_rows = 0
    evidence_rows = 0
    active_rows = 0
    recent_elapsed = deque(maxlen=50)
    run_start = time.time()
    resume_rows = 0
    if args.resume:
        existing = load_existing_output_stats(output_path)
        resume_rows = existing["rows"]
        queried_rows = resume_rows
        written_rows = resume_rows
        evidence_rows = existing["evidence_rows"]
        active_rows = existing["active_rows"]
        unmatched.extend(item for item in existing["unmatched"] if item)
        log(f"Resume enabled: found {resume_rows} existing output rows")

    input_file, reader, detected_encoding = open_csv_reader_with_fallback(input_path)
    log(f"Input encoding: {detected_encoding}")
    with input_file:
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        for required in [args.name_column, args.smiles_column]:
            if required not in reader.fieldnames:
                raise ValueError(f"Missing column: {required}")
        if args.resume and output_path.exists():
            for header in [RAW_HEADER, EVIDENCE_ANY_HEADER, ACTIVE_ANY_HEADER, SOURCE_METHODS_HEADER]:
                if header in reader.fieldnames:
                    log(
                        f"Warning: input already contains current indicator columns ({header}). "
                        f"Resume mode assumes the output file belongs to the same indicator."
                    )
                    break

        fieldnames = list(reader.fieldnames)
        for header in [
            RAW_HEADER,
            SOURCE_HEADER,
            BINARY_HEADER,
            STRUCTURED_EVIDENCE_ANY,
            STRUCTURED_ACTIVE_ANY,
            PUBMED_EVIDENCE_ANY,
            PUBMED_ACTIVE_ANY,
            PUBMED_PMIDS,
            PUBMED_SUPPORTIVE_SENTENCE_COUNT,
            EVIDENCE_ANY_HEADER,
            ACTIVE_ANY_HEADER,
            SOURCE_METHODS_HEADER,
            QUERY_ROUTE_HEADER,
        ]:
            if header not in fieldnames:
                fieldnames.append(header)

        open_mode = "a" if args.resume and output_path.exists() and output_path.stat().st_size > 0 else "w"
        write_header = open_mode == "w"
        skipped_for_resume = 0
        with output_path.open(open_mode, encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for row in reader:
                identifiers = extract_identifiers(row, args)
                if not identifiers["name"] and not identifiers["smiles"]:
                    continue
                if skipped_for_resume < resume_rows:
                    skipped_for_resume += 1
                    continue
                if args.limit and queried_rows >= args.limit:
                    break

                row_start = time.time()
                queried_rows += 1
                cache_key = identifiers["smiles"] or identifiers["inchi"] or identifiers["inchikey"] or identifiers["name"]
                if cache_key not in cache:
                    try:
                        cache[cache_key] = resolve_no_multisource(
                            identifiers,
                            cache_dir,
                            args.requests_per_second,
                            args.max_retries,
                            args.retmax,
                            args.email,
                            args.api_key,
                        )
                    except Exception as exc:
                        log(f"Row {queried_rows} failed after retries, continuing: {exc}")
                        cache[cache_key] = build_request_error("request_error", str(exc))
                    time.sleep(DELAY_SECONDS)

                result = cache[cache_key]
                row[RAW_HEADER] = result["raw"]
                row[SOURCE_HEADER] = result["source"]
                row[BINARY_HEADER] = result["binary"]
                row[STRUCTURED_EVIDENCE_ANY] = result["structured_evidence_any"]
                row[STRUCTURED_ACTIVE_ANY] = result["structured_active_any"]
                row[PUBMED_EVIDENCE_ANY] = result["pubmed_evidence_any"]
                row[PUBMED_ACTIVE_ANY] = result["pubmed_active_any"]
                row[PUBMED_PMIDS] = result["pubmed_pmids"]
                row[PUBMED_SUPPORTIVE_SENTENCE_COUNT] = result["pubmed_supportive_sentence_count"]
                row[EVIDENCE_ANY_HEADER] = result["evidence_any"]
                row[ACTIVE_ANY_HEADER] = result["active_any"]
                row[SOURCE_METHODS_HEADER] = result["source_methods"]
                row[QUERY_ROUTE_HEADER] = result["query_route"]
                writer.writerow(row)
                output_file.flush()
                written_rows += 1

                if result["evidence_any"] == 1:
                    evidence_rows += 1
                if result["active_any"] == 1:
                    active_rows += 1
                if result["evidence_any"] == 0:
                    unmatched.append(identifiers["name"] or cache_key)

                row_elapsed = time.time() - row_start
                recent_elapsed.append(row_elapsed)

                if queried_rows % args.progress_interval == 0:
                    elapsed_total = time.time() - run_start
                    current_run_rows = max(1, queried_rows - resume_rows)
                    avg_seconds = elapsed_total / current_run_rows
                    recent_avg = sum(recent_elapsed) / len(recent_elapsed) if recent_elapsed else avg_seconds
                    target_total = args.limit if args.limit else queried_rows
                    remaining = max(0, target_total - queried_rows)
                    eta_seconds = remaining * recent_avg
                    finish_time = datetime.now() + timedelta(seconds=eta_seconds)
                    log(
                        f"Progress {queried_rows}/{target_total}; evidence_any={evidence_rows}; "
                        f"active_any={active_rows}; cache={len(cache)}; unmatched={len(set(unmatched))}; "
                        f"last={format_duration(row_elapsed)}; avg={format_duration(avg_seconds)}; "
                        f"recent50={format_duration(recent_avg)}; ETA={format_duration(eta_seconds)}; "
                        f"finish~{finish_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

    unmatched_path.write_text("\n".join(sorted(set(unmatched))), encoding="utf-8")
    log(f"Output file: {output_path}")
    log(f"Unmatched file: {unmatched_path}")
    log(f"Queried rows: {queried_rows}")
    log(f"Written rows: {written_rows}")
    log(f"Rows with any evidence: {evidence_rows}")
    log(f"Rows with active evidence: {active_rows}")
    log(f"Unique unmatched names: {len(set(unmatched))}")


if __name__ == "__main__":
    main()
