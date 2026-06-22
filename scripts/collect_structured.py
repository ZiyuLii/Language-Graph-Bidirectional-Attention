import argparse
import csv
import http.client
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


INPUT_CSV = "pubchem_polyphenol_candidates.csv"
PUBCHEM_EVIDENCE_CSV = "pubchem_layer1_bioassay/pubchem_bioassay_evidence_long.csv"
OUTPUT_DIR = "structured_activity_six_indicators"

DEFAULT_REQUESTS_PER_SECOND = 3.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_CHEMBL_MAX_ACTIVITY_PAGES = 5
DEFAULT_PROGRESS_INTERVAL = 1
DEFAULT_CHEMBL_WORKERS = 1

RATE_LIMIT_LOCK = threading.Lock()
LAST_REQUEST_AT = 0.0
CSV_WRITE_LOCK = threading.Lock()

IC50_THRESHOLD_UM = 10.0
INHIBITION_THRESHOLD_PERCENT = 50.0

INDICATORS = {
    "no": {
        "label": "NO inhibition",
        "include": [
            r"\bnitric oxide production\b",
            r"\bNO production\b",
            r"\bnitrite accumulation\b",
            r"\bnitric oxide scavenging\b",
            r"\bNO scavenging\b",
            r"\biNOS\b",
            r"\bNOS2\b",
        ],
        "exclude": [
            r"induction of nitric oxide production",
            r"\bperoxynitrite\b",
            r"\bsuperoxide\b",
            r"daf2 oxidation",
            r"dhr oxidation",
            r"phosphatidylinositol",
            r"\bpseudomonas\b",
            r"\bantibiofilm\b",
            r"\bantimicrobial\b",
            r"\bantialgal\b",
        ],
    },
    "tnf_alpha": {
        "label": "TNF-alpha inhibition",
        "include": [
            r"\bTNF[-\s]?alpha\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\bTNF[-\s]?a\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\btumou?r necrosis factor\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\binhibition of TNF[-\s]?alpha\b",
            r"\bTNF[-\s]?alpha inhibition\b",
        ],
        "exclude": [
            r"TNF[-\s]?alpha[-\s]?induced",
            r"TNF[-\s]?a[-\s]?induced",
            r"TNF[-\s]?alpha stimulated",
            r"TNF[-\s]?alpha-mediated NF",
        ],
    },
    "il6": {
        "label": "IL-6 inhibition",
        "include": [
            r"\bIL[-\s]?6\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\binterleukin[-\s]?6\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\binhibition of IL[-\s]?6\b",
            r"\bIL[-\s]?6 inhibition\b",
        ],
        "exclude": [
            r"IL[-\s]?6[-\s]?induced",
            r"IL[-\s]?6/JAK/STAT",
            r"HEK[-\s]?Blue IL[-\s]?6",
        ],
    },
    "il1_beta": {
        "label": "IL-1beta inhibition",
        "include": [
            r"\bIL[-\s]?1[-\s]?beta\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\bIL[-\s]?1B\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\binterleukin[-\s]?1[-\s]?beta\b.*\b(production|secretion|release|expression|level|levels)\b",
            r"\binhibition of IL[-\s]?1[-\s]?beta\b",
            r"\bIL[-\s]?1[-\s]?beta inhibition\b",
            r"\binflammasome.*IL[-\s]?1[-\s]?beta\b",
        ],
        "exclude": [
            r"IL[-\s]?1[-\s]?beta[-\s]?induced",
            r"radioligand binding",
        ],
    },
    "ros": {
        "label": "ROS reduction",
        "include": [
            r"\bROS\b.*\b(production|generation|level|levels|accumulation)\b",
            r"\breactive oxygen species\b.*\b(production|generation|level|levels|accumulation)\b",
            r"\boxidative stress\b",
            r"\bDCF[-\s]?DA\b",
            r"\bDCFH[-\s]?DA\b",
        ],
        "exclude": [
            r"\bsuperoxide dismutase\b",
            r"\bantioxidant capacity\b",
            r"\bDPPH\b",
            r"\bABTS\b",
            r"\bFRAP\b",
            r"\bORAC\b",
            r"\bphotodynamic\b",
        ],
    },
    "nfkb": {
        "label": "NF-kB pathway suppression",
        "include": [
            r"\bNF[-\s]?kB\b",
            r"\bNF[-\s]?kappa[-\s]?B\b",
            r"\bNFKB\b",
            r"\bp65\b.*\b(translocation|phosphorylation|activation)\b",
            r"\bRELA\b",
        ],
        "exclude": [
            r"TNF[-\s]?alpha[-\s]?induced apoptosis",
        ],
    },
}

LONG_FIELDS = [
    "Compound_CID",
    "Name",
    "indicator",
    "indicator_label",
    "source_db",
    "source_id",
    "assay_name",
    "activity_name",
    "activity_outcome",
    "activity_value",
    "activity_units",
    "rule",
    "binary",
]

CHEMBL_PROGRESS_FIELDS = [
    "Compound_CID",
    "Name",
    "InChIKey",
    "status",
    "molecule_chembl_id",
    "activity_pages_read",
    "activity_records_read",
    "evidence_records",
    "elapsed_seconds",
    "error",
]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def compile_rules():
    compiled = {}
    for key, config in INDICATORS.items():
        compiled[key] = {
            "include": [re.compile(pattern, re.IGNORECASE) for pattern in config["include"]],
            "exclude": [re.compile(pattern, re.IGNORECASE) for pattern in config["exclude"]],
        }
    return compiled


RULES = compile_rules()


def delay_from_rate(requests_per_second):
    if requests_per_second <= 0:
        raise ValueError("--requests-per-second must be greater than 0.")
    return 1.0 / requests_per_second


def wait_for_rate_slot(requests_per_second):
    global LAST_REQUEST_AT
    delay = delay_from_rate(requests_per_second)
    with RATE_LIMIT_LOCK:
        now = time.time()
        wait_seconds = LAST_REQUEST_AT + delay - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        LAST_REQUEST_AT = time.time()


def safe_get_json(url, requests_per_second, max_retries):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            wait_for_rate_slot(requests_per_second)
            request = Request(url, headers={"User-Agent": "lgba-structured-six-indicators/1.0"})
            with urlopen(request, timeout=90) as response:
                text = response.read().decode("utf-8")
            return json.loads(text)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
        ) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in {400, 404}:
                return {}
            sleep_seconds = delay_from_rate(requests_per_second) * attempt * 2
            log(f"Request failed, retrying in {sleep_seconds:.1f}s: {last_error}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Request failed after retries: {url}; last error: {last_error}")


def cached_json(cache_dir, namespace, key, url, requests_per_second, max_retries):
    directory = Path(cache_dir) / namespace
    directory.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(key))[:180]
    path = directory / f"{safe_key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = safe_get_json(url, requests_per_second, max_retries)
    tmp_path = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    try:
        tmp_path.replace(path)
    except FileExistsError:
        tmp_path.unlink(missing_ok=True)
    return data


def append_csv_row(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with CSV_WRITE_LOCK:
        needs_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


def load_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_completed_chembl_cids(progress_path, retry_failed=False):
    completed = set()
    for row in load_csv_rows(progress_path):
        cid = str(row.get("Compound_CID", "")).strip()
        status = str(row.get("status", "")).strip()
        if not cid:
            continue
        if retry_failed and status == "failed":
            continue
        completed.add(cid)
    return completed


def load_compounds(input_csv):
    compounds = {}
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            cid = str(row.get("Compound_CID", "")).strip()
            if not cid:
                continue
            compounds[cid] = {
                "Compound_CID": cid,
                "Name": row.get("Name", ""),
                "InChIKey": row.get("InChIKey", ""),
            }
    return compounds


def assay_indicators(assay_name, activity_name=""):
    text = f"{assay_name} {activity_name}"
    matches = []
    for key, rules in RULES.items():
        if not any(regex.search(text) for regex in rules["include"]):
            continue
        if any(regex.search(text) for regex in rules["exclude"]):
            continue
        matches.append(key)
    return matches


def parse_float(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def classify_activity(activity_name, outcome, value, units="", relation=""):
    activity_lower = str(activity_name or "").lower()
    outcome = str(outcome or "").strip()
    value_num = parse_float(value)

    if "ic50" == activity_lower.upper().lower() and value_num is not None:
        value_um = normalize_to_um(value_num, units)
        if value_um is None:
            return None, None
        if relation in (">", ">="):
            binary = 0 if value_um > IC50_THRESHOLD_UM else 1
        else:
            binary = 1 if value_um <= IC50_THRESHOLD_UM else 0
        return binary, f"IC50={value_um:g} uM"

    if any(token in activity_lower for token in ["inhibition", "% inhibition", "percent inhibition"]) and value_num is not None:
        binary = 1 if value_num >= INHIBITION_THRESHOLD_PERCENT else 0
        return binary, f"Inhibition={value_num:g}%"

    if outcome == "Active":
        return 1, "Outcome=Active"
    if outcome == "Inactive":
        return 0, "Outcome=Inactive"
    if str(outcome).lower() == "active":
        return 1, "Outcome=Active"
    if str(outcome).lower() == "inactive":
        return 0, "Outcome=Inactive"
    return None, None


def normalize_to_um(value, units):
    units_lower = str(units or "").strip().lower()
    if units_lower in {"um", "µm", "uM".lower(), ""}:
        return value
    if units_lower == "nm":
        return value / 1000.0
    return None


def classify_pubchem_evidence(pubchem_csv, compounds):
    evidence = []
    if not Path(pubchem_csv).exists():
        log(f"PubChem evidence CSV not found, skipping: {pubchem_csv}")
        return evidence

    with open(pubchem_csv, "r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for index, row in enumerate(reader, start=1):
            cid = str(row.get("Compound_CID", "")).strip()
            if not cid:
                continue
            if cid not in compounds:
                continue
            assay_name = str(row.get("Assay Name", "")).strip()
            activity_name = str(row.get("Activity Name", "")).strip()
            indicators = assay_indicators(assay_name, activity_name)
            if not indicators:
                continue
            binary, rule = classify_activity(
                activity_name,
                row.get("Activity Outcome", ""),
                row.get("Activity Value [uM]", ""),
                "uM",
            )
            if binary is None:
                binary = "Review"
                rule = "Relevant assay found but not classifiable"
            for indicator in indicators:
                evidence.append(
                    {
                        "Compound_CID": cid,
                        "Name": row.get("Name", compounds.get(cid, {}).get("Name", "")),
                        "indicator": indicator,
                        "indicator_label": INDICATORS[indicator]["label"],
                        "source_db": "PubChem",
                        "source_id": row.get("AID", ""),
                        "assay_name": assay_name,
                        "activity_name": activity_name,
                        "activity_outcome": row.get("Activity Outcome", ""),
                        "activity_value": row.get("Activity Value [uM]", ""),
                        "activity_units": "uM",
                        "rule": rule,
                        "binary": binary,
                    }
                )
            if index % 50000 == 0:
                log(f"Scanned PubChem evidence rows: {index}")
    return evidence


def fetch_chembl_molecule(compound, cache_dir, requests_per_second, max_retries):
    inchikey = compound.get("InChIKey", "")
    name = compound.get("Name", "")
    query_urls = []
    if inchikey:
        query_urls.append(
            (
                f"inchikey_{inchikey}",
                "https://www.ebi.ac.uk/chembl/api/data/molecule.json?"
                + urlencode({"molecule_structures__standard_inchi_key": inchikey}),
            )
        )
    if name and len(name) <= 120 and ";" not in name:
        query_urls.append(
            (
                f"name_{name}",
                "https://www.ebi.ac.uk/chembl/api/data/molecule.json?"
                + urlencode({"pref_name__iexact": name}),
            )
        )
    for key, url in query_urls:
        data = cached_json(cache_dir, "chembl_molecule", key, url, requests_per_second, max_retries)
        molecules = data.get("molecules", [])
        if molecules:
            return molecules[0]
    return None


def fetch_chembl_activities(
    molecule_chembl_id,
    cache_dir,
    requests_per_second,
    max_retries,
    max_pages=0,
    log_pages=False,
):
    activities = []
    offset = 0
    limit = 1000
    pages_read = 0
    cap_reached = False
    while True:
        params = urlencode({"molecule_chembl_id": molecule_chembl_id, "limit": limit, "offset": offset})
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?{params}"
        data = cached_json(cache_dir, "chembl_activity", f"{molecule_chembl_id}_{offset}", url, requests_per_second, max_retries)
        pages_read += 1
        batch = data.get("activities", [])
        activities.extend(batch)
        if log_pages and (pages_read == 1 or pages_read % 5 == 0):
            next_marker = "has_next" if data.get("page_meta", {}).get("next") else "last_page"
            log(
                f"ChEMBL {molecule_chembl_id} activity page {pages_read}: "
                f"batch={len(batch)}, total={len(activities)}, {next_marker}"
            )
        if not data.get("page_meta", {}).get("next"):
            break
        if max_pages and pages_read >= max_pages:
            cap_reached = True
            log(f"ChEMBL activity page cap reached for {molecule_chembl_id}: {pages_read} pages, {len(activities)} records")
            break
        offset += limit
    return activities, pages_read, cap_reached


def classify_chembl_activities_for_compound(compound, activities):
    evidence = []
    for activity in activities:
        assay_name = str(activity.get("assay_description", "")).strip()
        activity_name = str(activity.get("standard_type", "")).strip()
        indicators = assay_indicators(assay_name, activity_name)
        if not indicators:
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
        for indicator in indicators:
            evidence.append(
                {
                    "Compound_CID": compound["Compound_CID"],
                    "Name": compound.get("Name", ""),
                    "indicator": indicator,
                    "indicator_label": INDICATORS[indicator]["label"],
                    "source_db": "ChEMBL",
                    "source_id": activity.get("assay_chembl_id", ""),
                    "assay_name": assay_name,
                    "activity_name": activity_name,
                    "activity_outcome": activity.get("activity_comment", ""),
                    "activity_value": activity.get("standard_value", ""),
                    "activity_units": activity.get("standard_units", ""),
                    "rule": rule,
                    "binary": binary,
                }
            )
    return evidence


def process_chembl_compound(
    compound,
    cache_dir,
    requests_per_second,
    max_retries,
    max_activity_pages,
):
    started = time.time()
    cid = compound["Compound_CID"]
    status = "failed"
    molecule_chembl_id = ""
    pages_read = 0
    activity_count = 0
    compound_evidence = []
    error = ""
    try:
        molecule = fetch_chembl_molecule(compound, cache_dir, requests_per_second, max_retries)
        if not molecule or not molecule.get("molecule_chembl_id"):
            status = "no_match"
        else:
            molecule_chembl_id = molecule["molecule_chembl_id"]
            activities, pages_read, cap_reached = fetch_chembl_activities(
                molecule_chembl_id,
                cache_dir,
                requests_per_second,
                max_retries,
                max_pages=max_activity_pages,
                log_pages=True,
            )
            activity_count = len(activities)
            compound_evidence = classify_chembl_activities_for_compound(compound, activities)
            status = "done_page_cap" if cap_reached else "done"
    except Exception as exc:
        error = str(exc)
        log(f"ChEMBL failed for CID={cid} Name={compound.get('Name')}: {error}")

    elapsed = time.time() - started
    progress_row = {
        "Compound_CID": cid,
        "Name": compound.get("Name", ""),
        "InChIKey": compound.get("InChIKey", ""),
        "status": status,
        "molecule_chembl_id": molecule_chembl_id,
        "activity_pages_read": pages_read,
        "activity_records_read": activity_count,
        "evidence_records": len(compound_evidence),
        "elapsed_seconds": f"{elapsed:.2f}",
        "error": error,
    }
    return {
        "compound": compound,
        "progress_row": progress_row,
        "evidence": compound_evidence,
        "elapsed": elapsed,
    }


def classify_chembl_for_compounds(
    compounds,
    cache_dir,
    requests_per_second,
    max_retries,
    limit=None,
    output_dir=None,
    retry_failed=False,
    max_activity_pages=DEFAULT_CHEMBL_MAX_ACTIVITY_PAGES,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
    workers=DEFAULT_CHEMBL_WORKERS,
):
    output_dir = Path(output_dir or Path(cache_dir).parent)
    progress_path = output_dir / "chembl_progress.csv"
    incremental_evidence_path = output_dir / "chembl_structured_six_indicator_evidence_incremental.csv"

    evidence = load_csv_rows(incremental_evidence_path)
    completed_cids = load_completed_chembl_cids(progress_path, retry_failed=retry_failed)

    compound_items = list(compounds.values())
    if limit:
        compound_items = compound_items[:limit]
    pending_items = [compound for compound in compound_items if compound["Compound_CID"] not in completed_cids]
    completed_in_scope = len(compound_items) - len(pending_items)

    log(
        f"ChEMBL resume state: total={len(compound_items)}, completed={len(completed_cids)}, "
        f"pending={len(pending_items)}, existing_evidence={len(evidence)}"
    )
    if max_activity_pages:
        log(f"ChEMBL activity page limit: {max_activity_pages} pages per molecule")
    else:
        log("ChEMBL activity page limit: unlimited")
    log(f"ChEMBL workers: {workers}; global request rate: {requests_per_second}/s")

    run_started = time.time()
    recent_elapsed = []
    workers = max(1, int(workers))

    if workers == 1:
        def sequential_results():
            for compound in pending_items:
                cid = compound["Compound_CID"]
                log(f"ChEMBL compound started CID={cid} Name={compound.get('Name', '')[:100]}")
                yield process_chembl_compound(compound, cache_dir, requests_per_second, max_retries, max_activity_pages)

        result_iter = sequential_results()
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = []
        for queue_index, compound in enumerate(pending_items, start=1):
            cid = compound["Compound_CID"]
            if queue_index == 1 or queue_index % 100 == 0 or queue_index == len(pending_items):
                log(f"ChEMBL queued {queue_index}/{len(pending_items)}; latest CID={cid}")
            futures.append(
                executor.submit(
                    process_chembl_compound,
                    compound,
                    cache_dir,
                    requests_per_second,
                    max_retries,
                    max_activity_pages,
                )
            )
        result_iter = as_completed(futures)

    for index, result_or_future in enumerate(result_iter, start=1):
        if workers == 1:
            result = result_or_future
        else:
            result = result_or_future.result()
        compound = result["compound"]
        progress_row = result["progress_row"]
        compound_evidence = result["evidence"]
        elapsed = result["elapsed"]
        cid = compound["Compound_CID"]

        for record in compound_evidence:
            append_csv_row(incremental_evidence_path, LONG_FIELDS, record)
        evidence.extend(compound_evidence)
        append_csv_row(progress_path, CHEMBL_PROGRESS_FIELDS, progress_row)

        remaining = len(pending_items) - index
        overall_avg = (time.time() - run_started) / index
        recent_elapsed.append(elapsed)
        if len(recent_elapsed) > 50:
            recent_elapsed.pop(0)
        recent_avg = sum(recent_elapsed) / len(recent_elapsed)
        active_workers = max(1, min(workers, len(pending_items)))
        eta_by_wall_throughput = remaining * overall_avg
        eta_by_recent_tasks = remaining * recent_avg / active_workers
        if index < active_workers * 2:
            eta_seconds = eta_by_wall_throughput
        else:
            eta_seconds = eta_by_recent_tasks * 0.6 + eta_by_wall_throughput * 0.4
        finish_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta_seconds))
        if progress_interval <= 1 or index == 1 or index % progress_interval == 0 or index == len(pending_items):
            log(
                f"ChEMBL CID={cid} status={progress_row['status']}; molecule={progress_row['molecule_chembl_id'] or '-'}; "
                f"pages={progress_row['activity_pages_read']}; activities={progress_row['activity_records_read']}; "
                f"evidence={len(compound_evidence)}; elapsed={elapsed:.1f}s; done_this_run={index}/{len(pending_items)}; "
                f"remaining={remaining}; wall_avg={overall_avg:.1f}s/done; task_recent50={recent_avg:.1f}s; "
                f"ETA={format_duration(eta_seconds)}; finish~{finish_at}"
            )
    if workers > 1:
        executor.shutdown(wait=True)
    return evidence


def choose_best(records):
    if not records:
        return {"status": "unknown", "raw": "", "source": "", "evidence_count": 0}
    positives = [record for record in records if str(record["binary"]) == "1"]
    negatives = [record for record in records if str(record["binary"]) == "0"]
    reviews = [record for record in records if str(record["binary"]) == "Review"]
    if positives:
        chosen = positives[0]
        status = "active"
    elif negatives:
        chosen = negatives[0]
        status = "inactive_tested"
    elif reviews:
        chosen = reviews[0]
        status = "review"
    else:
        chosen = records[0]
        status = "unknown_tested"
    return {
        "status": status,
        "raw": chosen.get("rule", ""),
        "source": f"{chosen.get('source_db')} {chosen.get('source_id')}: {chosen.get('assay_name')}",
        "evidence_count": len(records),
    }


def write_outputs(compounds, evidence, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "structured_six_indicator_evidence_long.csv"
    summary_csv = output_dir / "compound_structured_six_indicator_summary.csv"
    summary_txt = output_dir / "structured_six_indicator_summary.txt"

    with open(long_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=LONG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evidence)

    by_cid_indicator = defaultdict(lambda: defaultdict(list))
    for record in evidence:
        by_cid_indicator[record["Compound_CID"]][record["indicator"]].append(record)

    fields = ["Compound_CID", "Name"]
    for indicator in INDICATORS:
        fields.extend(
            [
                f"{indicator}_structured_status",
                f"{indicator}_structured_raw",
                f"{indicator}_structured_source",
                f"{indicator}_structured_evidence_count",
            ]
        )
    fields.extend(["structured_any_active_indicator_count", "structured_any_evidence_indicator_count"])

    status_counts = {indicator: Counter() for indicator in INDICATORS}
    rows_with_any_evidence = 0
    rows_with_any_active = 0

    with open(summary_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for cid, compound in compounds.items():
            row = {"Compound_CID": cid, "Name": compound.get("Name", "")}
            active_count = 0
            evidence_count = 0
            for indicator in INDICATORS:
                best = choose_best(by_cid_indicator[cid].get(indicator, []))
                row[f"{indicator}_structured_status"] = best["status"]
                row[f"{indicator}_structured_raw"] = best["raw"]
                row[f"{indicator}_structured_source"] = best["source"]
                row[f"{indicator}_structured_evidence_count"] = best["evidence_count"]
                status_counts[indicator][best["status"]] += 1
                if best["status"] != "unknown":
                    evidence_count += 1
                if best["status"] == "active":
                    active_count += 1
            row["structured_any_active_indicator_count"] = active_count
            row["structured_any_evidence_indicator_count"] = evidence_count
            if evidence_count:
                rows_with_any_evidence += 1
            if active_count:
                rows_with_any_active += 1
            writer.writerow(row)

    lines = [
        "Structured six-indicator activity summary",
        "",
        f"Compound count: {len(compounds)}",
        f"Evidence records: {len(evidence)}",
        f"Compounds with any structured indicator evidence: {rows_with_any_evidence}",
        f"Compounds with any active structured indicator: {rows_with_any_active}",
        "",
        "Per-indicator status counts:",
    ]
    for indicator, config in INDICATORS.items():
        lines.append(f"{config['label']}: {dict(status_counts[indicator])}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return long_csv, summary_csv, summary_txt


def parse_args():
    parser = argparse.ArgumentParser(description="Collect six inflammation indicators from structured PubChem BioAssay and ChEMBL activity data.")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--pubchem-evidence", default=PUBCHEM_EVIDENCE_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--include-chembl", action="store_true", help="Supplement PubChem with ChEMBL activity queries.")
    parser.add_argument("--chembl-limit", type=int, default=0, help="Limit ChEMBL compound queries for testing. Use 0 for all.")
    parser.add_argument(
        "--chembl-max-activity-pages",
        type=int,
        default=DEFAULT_CHEMBL_MAX_ACTIVITY_PAGES,
        help=(
            "Maximum ChEMBL activity pages per molecule. Each page has up to 1000 activities. "
            "Use 0 for unlimited. Default protects long overnight runs from very large molecules."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry CIDs marked failed in chembl_progress.csv. Done/no_match CIDs are still skipped.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print detailed ChEMBL progress every N compounds. Default 1 prints every compound.",
    )
    parser.add_argument(
        "--chembl-workers",
        type=int,
        default=DEFAULT_CHEMBL_WORKERS,
        help=(
            "Number of ChEMBL compounds processed concurrently. Requests still share "
            "the global --requests-per-second limiter."
        ),
    )
    parser.add_argument("--requests-per-second", type=float, default=DEFAULT_REQUESTS_PER_SECOND)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    compounds = load_compounds(args.input)
    log(f"Loaded compounds: {len(compounds)}")

    evidence = classify_pubchem_evidence(args.pubchem_evidence, compounds)
    log(f"PubChem structured evidence records: {len(evidence)}")

    if args.include_chembl:
        chembl_evidence = classify_chembl_for_compounds(
            compounds,
            cache_dir,
            args.requests_per_second,
            args.max_retries,
            limit=args.chembl_limit or None,
            output_dir=output_dir,
            retry_failed=args.retry_failed,
            max_activity_pages=args.chembl_max_activity_pages,
            progress_interval=args.progress_interval,
            workers=args.chembl_workers,
        )
        log(f"ChEMBL structured evidence records: {len(chembl_evidence)}")
        evidence.extend(chembl_evidence)

    long_csv, summary_csv, summary_txt = write_outputs(compounds, evidence, output_dir)
    log(f"Evidence long CSV: {long_csv}")
    log(f"Compound summary CSV: {summary_csv}")
    log(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
