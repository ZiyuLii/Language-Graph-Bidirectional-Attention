import csv
from pathlib import Path


INPUT_CSV = Path("Compound_FooDB_4500+_six_indicators_master_mimo.csv")
SUMMARY_CSV = Path("foodb_six_indicator_ai_summary.csv")
SUMMARY_TXT = Path("foodb_six_indicator_ai_summary.txt")
SUPPORTED_CSV = Path("foodb_six_indicator_ai_supported_current.csv")


INDICATORS = [
    {
        "key": "no",
        "name": "NO",
        "evidence": "NO inhibition evidence any",
        "active": "NO inhibition active any",
    },
    {
        "key": "tnf_alpha",
        "name": "TNF-alpha",
        "evidence": "TNF-alpha inhibition evidence any",
        "active": "TNF-alpha inhibition active any",
    },
    {
        "key": "il6",
        "name": "IL-6",
        "evidence": "IL-6 inhibition evidence any",
        "active": "IL-6 inhibition active any",
    },
    {
        "key": "il1_beta",
        "name": "IL-1beta",
        "evidence": "IL-1beta inhibition evidence any",
        "active": "IL-1beta inhibition active any",
    },
    {
        "key": "ros",
        "name": "ROS",
        "evidence": "ROS reduction evidence any",
        "active": "ROS reduction active any",
    },
    {
        "key": "nfkb",
        "name": "NF-kB",
        "evidence": "NF-kB pathway suppression evidence any",
        "active": "NF-kB pathway suppression active any",
    },
]


def load_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader), list(reader.fieldnames or [])


def count_value(rows, field, value):
    return sum(1 for row in rows if str(row.get(field, "")).strip() == value)


def main():
    rows, fieldnames = load_rows(INPUT_CSV)
    summary_rows = []
    available_mimo_keys = []

    for item in INDICATORS:
        key = item["key"]
        support_field = f"mimo_{key}_supports_evidence"
        strength_field = f"mimo_{key}_strength"
        error_field = f"mimo_{key}_error"

        evidence_count = count_value(rows, item["evidence"], "1")
        active_count = count_value(rows, item["active"], "1")
        inactive_or_review_count = evidence_count - active_count

        has_mimo = support_field in fieldnames
        if has_mimo:
            available_mimo_keys.append(key)
            mimo_true = count_value(rows, support_field, "True")
            mimo_false = count_value(rows, support_field, "False")
            mimo_blank = sum(1 for row in rows if not str(row.get(support_field, "")).strip())
            mimo_errors = sum(1 for row in rows if str(row.get(error_field, "")).strip())
            structured_strong = count_value(rows, strength_field, "structured_strong")
            pubmed_supportive = count_value(rows, strength_field, "pubmed_supportive")
            pubmed_mention_only = count_value(rows, strength_field, "pubmed_mention_only")
            mixed = count_value(rows, strength_field, "mixed")
            not_supported = count_value(rows, strength_field, "not_supported")
        else:
            mimo_true = ""
            mimo_false = ""
            mimo_blank = ""
            mimo_errors = ""
            structured_strong = ""
            pubmed_supportive = ""
            pubmed_mention_only = ""
            mixed = ""
            not_supported = ""

        summary_rows.append(
            {
                "indicator": item["name"],
                "raw_evidence_any": evidence_count,
                "raw_active_any": active_count,
                "raw_evidence_not_active": inactive_or_review_count,
                "mimo_status": "done" if has_mimo else "missing",
                "mimo_support_true": mimo_true,
                "mimo_support_false": mimo_false,
                "mimo_blank": mimo_blank,
                "mimo_errors": mimo_errors,
                "mimo_structured_strong": structured_strong,
                "mimo_pubmed_supportive": pubmed_supportive,
                "mimo_pubmed_mention_only": pubmed_mention_only,
                "mimo_mixed": mixed,
                "mimo_not_supported": not_supported,
            }
        )

    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    supported_rows = []
    for row in rows:
        supported_indicators = []
        for key in available_mimo_keys:
            if str(row.get(f"mimo_{key}_supports_evidence", "")).strip() == "True":
                supported_indicators.append(key)
        if supported_indicators:
            out = dict(row)
            out["mimo_supported_indicators_current"] = ";".join(supported_indicators)
            out["mimo_supported_indicator_count_current"] = len(supported_indicators)
            supported_rows.append(out)

    supported_fields = list(fieldnames)
    for field in ["mimo_supported_indicators_current", "mimo_supported_indicator_count_current"]:
        if field not in supported_fields:
            supported_fields.append(field)
    with SUPPORTED_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=supported_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(supported_rows)

    lines = [
        f"Input: {INPUT_CSV}",
        f"Rows: {len(rows)}",
        "",
        "Indicator summary:",
    ]
    for row in summary_rows:
        if row["mimo_status"] == "done":
            lines.append(
                f"- {row['indicator']}: raw evidence={row['raw_evidence_any']}, raw active={row['raw_active_any']}, "
                f"MiMo true={row['mimo_support_true']}, false={row['mimo_support_false']}, errors={row['mimo_errors']}"
            )
        else:
            lines.append(
                f"- {row['indicator']}: raw evidence={row['raw_evidence_any']}, raw active={row['raw_active_any']}, "
                "MiMo result missing in current master_mimo table"
            )
    lines.extend(
        [
            "",
            f"Current MiMo-supported compound rows: {len(supported_rows)}",
            f"Summary CSV: {SUMMARY_CSV}",
            f"Supported CSV: {SUPPORTED_CSV}",
        ]
    )
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary TXT: {SUMMARY_TXT}")
    print(f"Supported CSV: {SUPPORTED_CSV}")


if __name__ == "__main__":
    main()
