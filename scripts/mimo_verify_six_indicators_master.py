import argparse
import csv
import hashlib
import http.client
import io
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CONFIG_JSON = "mimo_llm_config.json"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "lgba-foodb-six-indicators-mimo-review/1.0"

DEFAULT_INPUT = "Compound_FooDB_4500+_six_indicators_master.csv"
DEFAULT_OUTPUT = "Compound_FooDB_4500+_six_indicators_master_mimo.csv"
DEFAULT_SUMMARY = "Compound_FooDB_4500+_six_indicators_master_mimo_summary.txt"


INDICATOR_CONFIG = {
    "no": {
        "label": "NO inhibition",
        "prefix": "NO",
        "subject": "NO inhibition or NO-related anti-inflammatory activity",
        "evidence_examples": "NO inhibition, iNOS/NOS2 suppression, nitrite accumulation reduction, NO scavenging, or direct NO-related anti-inflammatory effect",
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
        "subject": "TNF-alpha inhibition or TNF-alpha-related anti-inflammatory activity",
        "evidence_examples": "TNF-alpha production, secretion, release, expression, level reduction, or TNF-alpha pathway suppression",
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
        "subject": "IL-6 inhibition or IL-6-related anti-inflammatory activity",
        "evidence_examples": "IL-6 production, secretion, release, expression, or level reduction",
        "patterns": [
            r"\bIL[-\s]?6\b",
            r"\bIL6\b",
            r"\binterleukin[-\s]?6\b",
        ],
    },
    "il1_beta": {
        "label": "IL-1beta inhibition",
        "prefix": "IL-1beta",
        "subject": "IL-1beta inhibition or IL-1beta-related anti-inflammatory activity",
        "evidence_examples": "IL-1beta production, secretion, release, expression, level reduction, or inflammasome-linked IL-1beta suppression",
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
        "subject": "ROS reduction or oxidative-stress-related anti-inflammatory activity",
        "evidence_examples": "ROS production, reactive oxygen species generation, oxidative stress, DCF-DA, or DCFH-DA reduction",
        "patterns": [
            r"\bROS\b",
            r"\breactive oxygen species\b",
            r"\boxidative stress\b",
            r"\bDCF[-\s]?DA\b",
            r"\bDCFH[-\s]?DA\b",
        ],
    },
    "nfkb": {
        "label": "NF-kB pathway suppression",
        "prefix": "NF-kB",
        "subject": "NF-kB pathway suppression or NF-kB-related anti-inflammatory activity",
        "evidence_examples": "NF-kB/NF-kappaB activation, p65 translocation or phosphorylation, RELA, or pathway suppression",
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


def safe_filename(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:80]
    return f"{prefix}_{digest}"


def open_csv_rows(path):
    raw_bytes = Path(path).read_bytes()
    last_error = None
    for encoding in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            text = raw_bytes.decode(encoding)
            handle = io.StringIO(text)
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header row: {path}")
            return list(reader), list(reader.fieldnames)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def write_csv(path, fieldnames, rows):
    all_fields = list(fieldnames)
    seen = set(all_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_fields.append(key)
                seen.add(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as config_file:
        return json.load(config_file)


def indicator_field(indicator, kind):
    config = INDICATOR_CONFIG[indicator]
    label = config["label"]
    prefix = config["prefix"]
    mapping = {
        "raw": f"{label} raw",
        "binary": f"{label} binary",
        "evidence": f"{label} evidence any",
        "active": f"{label} active any",
        "source_methods": f"{label} source methods",
        "pubmed_pmids": f"{prefix} pubmed pmids",
    }
    return mapping[kind]


def mimo_field(indicator, suffix):
    return f"mimo_{indicator}_{suffix}"


def mimo_fields_for_indicator(indicator):
    return [
        mimo_field(indicator, "supports_evidence"),
        mimo_field(indicator, "strength"),
        mimo_field(indicator, "direction"),
        mimo_field(indicator, "confidence"),
        mimo_field(indicator, "primary_basis"),
        mimo_field(indicator, "reason"),
        mimo_field(indicator, "pubmed_sentence_count"),
        mimo_field(indicator, "pubmed_sentences"),
        mimo_field(indicator, "raw_response"),
        mimo_field(indicator, "error"),
    ]


def request_text(url, user_agent, max_retries, delay_seconds):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with urlopen(request, timeout=120) as response:
                text = response.read().decode("utf-8")
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            return text
        except (
            HTTPError,
            URLError,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            OSError,
        ) as exc:
            last_error = exc
            sleep_seconds = min(60, max(1.0, delay_seconds) * attempt * 2)
            log(f"HTTP failed, retrying in {sleep_seconds:.1f}s: {last_error}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"HTTP request failed after retries: {last_error}")


def cached_get(cache_dir, namespace, key, suffix, url, user_agent, max_retries, delay_seconds):
    directory = Path(cache_dir) / namespace
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_filename(key)}.{suffix}"
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = request_text(url, user_agent, max_retries, delay_seconds)
    path.write_text(text, encoding="utf-8")
    return text


def split_sentences(text):
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def extract_json_array(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            return value["items"]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        return json.loads(match.group(0))
    raise ValueError("Model response does not contain a JSON array.")


class AdaptiveLimiter:
    def __init__(self, start_rpm, min_rpm, max_rpm):
        self.current_rpm = float(start_rpm)
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(max_rpm)
        self.lock = threading.Lock()
        self.next_allowed_at = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            delay = 60.0 / self.current_rpm
            if self.next_allowed_at > now:
                wait_seconds = self.next_allowed_at - now
                time.sleep(wait_seconds)
                now = time.time()
            self.next_allowed_at = max(self.next_allowed_at, now) + delay

    def set_rpm(self, new_rpm):
        with self.lock:
            self.current_rpm = max(self.min_rpm, min(self.max_rpm, float(new_rpm)))

    def get_rpm(self):
        with self.lock:
            return self.current_rpm


class AdaptiveController:
    def __init__(self, start_workers, min_workers, max_workers, start_rpm, min_rpm, max_rpm):
        self.current_workers = int(start_workers)
        self.min_workers = int(min_workers)
        self.max_workers = int(max_workers)
        self.limiter = AdaptiveLimiter(start_rpm, min_rpm, max_rpm)
        self.recent = deque(maxlen=12)
        self.lock = threading.Lock()

    def record(self, success, latency_seconds):
        with self.lock:
            self.recent.append((1 if success else 0, float(latency_seconds)))
            if len(self.recent) < 6:
                return None
            success_rate = sum(item[0] for item in self.recent) / len(self.recent)
            avg_latency = sum(item[1] for item in self.recent) / len(self.recent)
            new_workers = self.current_workers
            new_rpm = self.limiter.get_rpm()
            reason = None
            if success_rate < 0.8 or avg_latency > 90:
                new_workers = max(self.min_workers, self.current_workers - 1)
                new_rpm = max(self.limiter.min_rpm, new_rpm * 0.8)
                reason = f"slow/error recent_window success_rate={success_rate:.2f} avg_latency={avg_latency:.1f}s"
            elif success_rate == 1.0 and avg_latency < 35:
                new_workers = min(self.max_workers, self.current_workers + 1)
                new_rpm = min(self.limiter.max_rpm, new_rpm * 1.15)
                reason = f"fast/stable recent_window success_rate={success_rate:.2f} avg_latency={avg_latency:.1f}s"
            elif success_rate >= 0.9 and avg_latency < 55:
                new_rpm = min(self.limiter.max_rpm, new_rpm * 1.05)
                reason = f"stable recent_window success_rate={success_rate:.2f} avg_latency={avg_latency:.1f}s"
            changed = (new_workers != self.current_workers) or (abs(new_rpm - self.limiter.get_rpm()) > 0.01)
            self.current_workers = new_workers
            self.limiter.set_rpm(new_rpm)
            if changed:
                return {
                    "workers": self.current_workers,
                    "rpm": self.limiter.get_rpm(),
                    "reason": reason,
                }
            return None


def request_mimo(messages, config, max_retries, limiter):
    payload = {
        "model": config.get("model", "mimo-v2.5-pro"),
        "messages": messages,
        "max_completion_tokens": int(config.get("max_completion_tokens", 4096)),
        "temperature": float(config.get("temperature", 0.1)),
        "top_p": float(config.get("top_p", 0.95)),
        "stream": False,
        "stop": None,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "api-key": config["api_key"],
        "Content-Type": "application/json",
    }
    base_url = config.get("base_url", "")
    if base_url.endswith("/chat/completions"):
        request_url = base_url
    else:
        request_url = base_url.rstrip("/") + "/chat/completions"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            limiter.acquire()
            request = Request(request_url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=180) as response:
                response_text = response.read().decode("utf-8")
            data = json.loads(response_text)
            return data["choices"][0]["message"]["content"]
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            sleep_seconds = min(90, attempt * 5.0)
            log(f"MiMo request failed, retrying in {sleep_seconds:.1f}s: {last_error}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"MiMo request failed after retries: {last_error}")


def fetch_pubmed_articles(pmids, cache_dir, max_retries, delay_seconds):
    if not pmids:
        return []
    key = ",".join(pmids)
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": key,
        "tool": "lgba_foodb_six_indicator_mimo_verify",
    }
    url = f"{NCBI_EUTILS}/efetch.fcgi?{urlencode(params)}"
    xml_text = cached_get(
        cache_dir,
        "pubmed_efetch",
        key,
        "xml",
        url,
        USER_AGENT,
        max_retries,
        delay_seconds,
    )
    if not xml_text.strip():
        return []
    root = ET.fromstring(xml_text)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()) if title_node is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            abstract_parts.append("".join(node.itertext()))
        abstract = " ".join(part.strip() for part in abstract_parts if part and part.strip())
        articles.append({"pmid": pmid, "title": title, "abstract": abstract})
    return articles


def collect_indicator_sentences(row, indicator, cache_dir, max_pmids, max_sentences, max_retries):
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in INDICATOR_CONFIG[indicator]["patterns"]]
    pmid_text = str(row.get(indicator_field(indicator, "pubmed_pmids"), "") or "").strip()
    if not pmid_text:
        return []
    pmids = [part.strip() for part in pmid_text.split(";") if part.strip()][:max_pmids]
    try:
        articles = fetch_pubmed_articles(pmids, cache_dir, max_retries, 0.34)
    except Exception as exc:
        log(
            f"PubMed sentence fetch failed for indicator={indicator} "
            f"name={row.get('name', '')[:80]} pmids={';'.join(pmids[:3])}: {exc}"
        )
        return []
    sentences = []
    for article in articles:
        combined_text = " ".join(part for part in [article["title"], article["abstract"]] if part)
        for sentence in split_sentences(combined_text):
            if any(pattern.search(sentence) for pattern in patterns):
                sentences.append(f"PMID {article['pmid']}: {sentence}")
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def make_prompt(indicator, batch):
    config = INDICATOR_CONFIG[indicator]
    items = []
    for item in batch:
        row = item["row"]
        items.append(
            {
                "task_id": item["task_id"],
                "compound": row.get("name", ""),
                "source_methods": row.get(indicator_field(indicator, "source_methods"), ""),
                "structured_raw": row.get(indicator_field(indicator, "raw"), ""),
                "structured_binary": row.get(indicator_field(indicator, "binary"), ""),
                "structured_evidence_any": row.get(indicator_field(indicator, "evidence"), ""),
                "structured_active_any": row.get(indicator_field(indicator, "active"), ""),
                "pubmed_pmids": row.get(indicator_field(indicator, "pubmed_pmids"), ""),
                "pubmed_sentences": item["pubmed_sentences"],
            }
        )
    return (
        f"You are reviewing whether each compound has any evidence relevant to {config['subject']}. "
        f"Be conservative and only use the provided structured evidence text and PubMed {config['label']}-related sentences. "
        f"If there is any evidence touching {config['evidence_examples']}, supports_evidence should be true. "
        "If the evidence is only broad mention, derivative/formula context, or unclear linkage to the exact compound, mark strength as mention_only or mixed. "
        "Return only a JSON array. Each object must contain: "
        "task_id, supports_evidence, strength, direction, confidence, primary_basis, reason. "
        "Allowed values: "
        "supports_evidence=true/false; "
        "strength=structured_strong/pubmed_supportive/pubmed_mention_only/mixed/not_supported; "
        "direction=inhibit_decrease/increase_activate/mention_only/unclear; "
        "confidence=high/medium/low; "
        "primary_basis=PubChem/ChEMBL/PubMed/mixed/none; "
        "reason should be <= 12 words.\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False)}"
    )


def apply_result_to_row(row, indicator, task, result):
    row[mimo_field(indicator, "supports_evidence")] = str(result.get("supports_evidence", ""))
    row[mimo_field(indicator, "strength")] = str(result.get("strength", ""))
    row[mimo_field(indicator, "direction")] = str(result.get("direction", ""))
    row[mimo_field(indicator, "confidence")] = str(result.get("confidence", ""))
    row[mimo_field(indicator, "primary_basis")] = str(result.get("primary_basis", ""))
    row[mimo_field(indicator, "reason")] = str(result.get("reason", ""))
    row[mimo_field(indicator, "pubmed_sentence_count")] = len(task["pubmed_sentences"])
    row[mimo_field(indicator, "pubmed_sentences")] = " || ".join(task["pubmed_sentences"])
    row[mimo_field(indicator, "raw_response")] = ""
    row[mimo_field(indicator, "error")] = ""


def apply_error_to_row(row, indicator, task, error_text, raw_response=""):
    row[mimo_field(indicator, "supports_evidence")] = ""
    row[mimo_field(indicator, "strength")] = ""
    row[mimo_field(indicator, "direction")] = ""
    row[mimo_field(indicator, "confidence")] = ""
    row[mimo_field(indicator, "primary_basis")] = ""
    row[mimo_field(indicator, "reason")] = ""
    row[mimo_field(indicator, "pubmed_sentence_count")] = len(task["pubmed_sentences"])
    row[mimo_field(indicator, "pubmed_sentences")] = " || ".join(task["pubmed_sentences"])
    row[mimo_field(indicator, "raw_response")] = raw_response[:4000]
    row[mimo_field(indicator, "error")] = error_text


def process_batch(batch, config, limiter, max_retries, indicator):
    start = time.time()
    prompt = make_prompt(indicator, batch)
    messages = [
        {"role": "system", "content": "You are MiMo, an AI assistant developed by Xiaomi. Return strict JSON only."},
        {"role": "user", "content": prompt},
    ]
    results = {}
    raw_batch_response = ""
    try:
        raw_batch_response = request_mimo(messages, config, max_retries, limiter)
        parsed = extract_json_array(raw_batch_response)
        parsed_map = {str(item.get("task_id", "")): item for item in parsed}
        for task in batch:
            results[task["task_id"]] = {
                "ok": True,
                "result": parsed_map.get(task["task_id"], {}),
                "raw_response": "",
                "error": "",
            }
        return {
            "ok": True,
            "elapsed": time.time() - start,
            "results": results,
        }
    except Exception as exc:
        for task in batch:
            raw_response = ""
            try:
                single_prompt = make_prompt(indicator, [task])
                single_messages = [
                    {"role": "system", "content": "You are MiMo, an AI assistant developed by Xiaomi. Return strict JSON only."},
                    {"role": "user", "content": single_prompt},
                ]
                raw_response = request_mimo(single_messages, config, max_retries, limiter)
                parsed = extract_json_array(raw_response)
                results[task["task_id"]] = {
                    "ok": True,
                    "result": parsed[0] if parsed else {},
                    "raw_response": "",
                    "error": "",
                }
            except Exception as single_exc:
                results[task["task_id"]] = {
                    "ok": False,
                    "result": {},
                    "raw_response": raw_response or raw_batch_response,
                    "error": str(single_exc),
                }
        return {
            "ok": False,
            "elapsed": time.time() - start,
            "results": results,
            "batch_error": str(exc),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiMo review queue for all six indicators on the merged FooDB master CSV."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--config", default=CONFIG_JSON)
    parser.add_argument("--indicators", default="no,tnf_alpha,il6,il1_beta,ros,nfkb")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--max-pmids", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--start-workers", type=int, default=2)
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--start-rpm", type=float, default=24.0)
    parser.add_argument("--min-rpm", type=float, default=10.0)
    parser.add_argument("--max-rpm", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def build_tasks(rows, indicators, cache_dir, args):
    tasks = []
    for row_index, row in enumerate(rows, start=1):
        for indicator in indicators:
            evidence_field = indicator_field(indicator, "evidence")
            if str(row.get(evidence_field, "")).strip() != "1":
                continue
            support_field = mimo_field(indicator, "supports_evidence")
            error_field = mimo_field(indicator, "error")
            if str(row.get(support_field, "")).strip() and not str(row.get(error_field, "")).strip():
                continue
            tasks.append(
                {
                    "task_id": f"{indicator}:{row_index}",
                    "row_index": row_index,
                    "indicator": indicator,
                    "row": row,
                    "pubmed_sentences": collect_indicator_sentences(
                        row,
                        indicator,
                        cache_dir,
                        args.max_pmids,
                        args.max_sentences,
                        args.max_retries,
                    ),
                }
            )
            if args.limit_tasks and len(tasks) >= args.limit_tasks:
                return tasks
    return tasks


def write_summary(summary_path, rows, indicators):
    lines = [f"Rows: {len(rows)}", ""]
    for indicator in indicators:
        supports = Counter()
        strengths = Counter()
        errors = 0
        for row in rows:
            supports[str(row.get(mimo_field(indicator, "supports_evidence"), ""))] += 1
            strengths[str(row.get(mimo_field(indicator, "strength"), ""))] += 1
            if str(row.get(mimo_field(indicator, "error"), "")).strip():
                errors += 1
        lines.append(f"[{indicator}]")
        lines.append(f"errors: {errors}")
        lines.append("supports_evidence:")
        for key, count in sorted(supports.items()):
            lines.append(f"  {key or '<blank>'}: {count}")
        lines.append("strength:")
        for key, count in sorted(strengths.items()):
            lines.append(f"  {key or '<blank>'}: {count}")
        lines.append("")
    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    indicators = [item.strip() for item in args.indicators.split(",") if item.strip()]
    for indicator in indicators:
        if indicator not in INDICATOR_CONFIG:
            raise ValueError(f"Unsupported indicator: {indicator}")

    config = load_config(args.config)
    rows, fieldnames = open_csv_rows(args.input)
    for indicator in indicators:
        for field in mimo_fields_for_indicator(indicator):
            if field not in fieldnames:
                fieldnames.append(field)

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    cache_dir = output_path.parent / f"{output_path.stem}_cache"

    if output_path.exists() and output_path.stat().st_size > 0:
        existing_rows, _ = open_csv_rows(output_path)
        if len(existing_rows) == len(rows):
            rows = existing_rows
            log(f"Resume from existing output: {output_path.name}")

    batch_size = args.batch_size or int(config.get("batch_size", 8))
    tasks = build_tasks(rows, indicators, cache_dir, args)
    total_tasks = len(tasks)
    log(
        f"MiMo six-indicator queue: rows={len(rows)}; indicators={','.join(indicators)}; "
        f"pending_tasks={total_tasks}; output={output_path.name}"
    )

    if total_tasks == 0:
        write_csv(output_path, fieldnames, rows)
        write_summary(summary_path, rows, indicators)
        log("No pending tasks.")
        return

    batches = [tasks[index:index + batch_size] for index in range(0, len(tasks), batch_size)]
    controller = AdaptiveController(
        start_workers=args.start_workers,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        start_rpm=args.start_rpm,
        min_rpm=args.min_rpm,
        max_rpm=args.max_rpm,
    )

    row_map = {index + 1: row for index, row in enumerate(rows)}
    completed_batches = 0
    completed_tasks = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending_batches = deque(batches)
        in_flight = {}

        def submit_more():
            while pending_batches and len(in_flight) < controller.current_workers:
                batch = pending_batches.popleft()
                future = executor.submit(
                    process_batch,
                    batch,
                    config,
                    controller.limiter,
                    args.max_retries,
                    batch[0]["indicator"],
                )
                in_flight[future] = batch

        submit_more()
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                batch = in_flight.pop(future)
                batch_result = future.result()
                batch_error = batch_result.get("batch_error", "")
                if batch_error:
                    log(f"Batch fallback triggered for {len(batch)} tasks: {batch_error}")
                for task in batch:
                    row = row_map[task["row_index"]]
                    result_info = batch_result["results"][task["task_id"]]
                    if result_info["ok"]:
                        apply_result_to_row(row, task["indicator"], task, result_info["result"])
                    else:
                        apply_error_to_row(row, task["indicator"], task, result_info["error"], result_info["raw_response"])
                completed_batches += 1
                completed_tasks += len(batch)
                adjustment = controller.record(batch_result["ok"], batch_result["elapsed"])
                if adjustment:
                    log(
                        f"Adaptive tuning: workers={adjustment['workers']} rpm={adjustment['rpm']:.1f} "
                        f"reason={adjustment['reason']}"
                    )

                if completed_batches % args.save_every == 0 or completed_batches == len(batches):
                    write_csv(output_path, fieldnames, rows)
                    write_summary(summary_path, rows, indicators)

                if completed_batches % args.progress_interval == 0 or completed_batches == len(batches):
                    elapsed = time.time() - start_time
                    avg_per_task = elapsed / completed_tasks if completed_tasks else 0
                    remaining_tasks = total_tasks - completed_tasks
                    eta_seconds = remaining_tasks * avg_per_task / max(1, controller.current_workers)
                    finish_time = datetime.now() + timedelta(seconds=eta_seconds)
                    log(
                        f"Progress tasks {completed_tasks}/{total_tasks}; batches {completed_batches}/{len(batches)}; "
                        f"workers={controller.current_workers}; rpm={controller.limiter.get_rpm():.1f}; "
                        f"avg/task={format_duration(avg_per_task)}; ETA={format_duration(eta_seconds)}; "
                        f"finish~{finish_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            submit_more()

    write_csv(output_path, fieldnames, rows)
    write_summary(summary_path, rows, indicators)
    log(f"Review CSV: {output_path}")
    log(f"Summary TXT: {summary_path}")


if __name__ == "__main__":
    main()
