import csv
import io
from pathlib import Path


BASE_CSV = Path("Compound_FooDB_4500+.csv")
OUTPUT_CSV = Path("Compound_FooDB_4500+_six_indicators_master.csv")
SUMMARY_TXT = Path("Compound_FooDB_4500+_six_indicators_master_summary.txt")


INDICATORS = [
    {
        "key": "no",
        "input": Path("Compound_FooDB_4500+_no_multisource_4526.csv"),
        "cols": [
            "NO inhibition raw",
            "NO inhibition binary",
            "NO inhibition evidence any",
            "NO inhibition active any",
            "NO inhibition source methods",
            "NO pubmed pmids",
        ],
    },
    {
        "key": "tnf_alpha",
        "input": Path("Compound_FooDB_4500+_tnf_alpha_multisource_4526.csv"),
        "cols": [
            "TNF-alpha inhibition raw",
            "TNF-alpha inhibition binary",
            "TNF-alpha inhibition evidence any",
            "TNF-alpha inhibition active any",
            "TNF-alpha inhibition source methods",
            "TNF-alpha pubmed pmids",
        ],
    },
    {
        "key": "il6",
        "input": Path("Compound_FooDB_4500+_il6_multisource_4526.csv"),
        "cols": [
            "IL-6 inhibition raw",
            "IL-6 inhibition binary",
            "IL-6 inhibition evidence any",
            "IL-6 inhibition active any",
            "IL-6 inhibition source methods",
            "IL-6 pubmed pmids",
        ],
    },
    {
        "key": "il1_beta",
        "input": Path("Compound_FooDB_4500+_il1_beta_multisource_4526.csv"),
        "cols": [
            "IL-1beta inhibition raw",
            "IL-1beta inhibition binary",
            "IL-1beta inhibition evidence any",
            "IL-1beta inhibition active any",
            "IL-1beta inhibition source methods",
            "IL-1beta pubmed pmids",
        ],
    },
    {
        "key": "ros",
        "input": Path("Compound_FooDB_4500+_ros_multisource_4526.csv"),
        "cols": [
            "ROS reduction raw",
            "ROS reduction binary",
            "ROS reduction evidence any",
            "ROS reduction active any",
            "ROS reduction source methods",
            "ROS pubmed pmids",
        ],
    },
    {
        "key": "nfkb",
        "input": Path("Compound_FooDB_4500+_nfkb_multisource_4526.csv"),
        "cols": [
            "NF-kB pathway suppression raw",
            "NF-kB pathway suppression binary",
            "NF-kB pathway suppression evidence any",
            "NF-kB pathway suppression active any",
            "NF-kB pathway suppression source methods",
            "NF-kB pubmed pmids",
        ],
    },
]


def load_csv_rows(path):
    raw_bytes = path.read_bytes()
    last_error = None
    for encoding in ["utf-8-sig", "gb18030", "utf-8"]:
        try:
            text = raw_bytes.decode(encoding)
            input_file = io.StringIO(text)
            reader = csv.DictReader(input_file)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header row: {path}")
            return list(reader), list(reader.fieldnames)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def row_key(row):
    return (
        str(row.get("id", "")).strip(),
        str(row.get("public_id", "")).strip(),
        str(row.get("name", "")).strip(),
    )


def main():
    if not BASE_CSV.exists():
        raise FileNotFoundError(f"Missing base CSV: {BASE_CSV}")

    base_rows, base_fields = load_csv_rows(BASE_CSV)
    base_keys = [row_key(row) for row in base_rows]

    merged_fields = list(base_fields)
    summaries = [f"Base rows: {len(base_rows)}", f"Output: {OUTPUT_CSV.name}", ""]

    for indicator in INDICATORS:
        input_path = indicator["input"]
        if not input_path.exists():
            raise FileNotFoundError(f"Missing indicator CSV: {input_path}")

        indicator_rows, indicator_fields = load_csv_rows(input_path)
        if len(indicator_rows) != len(base_rows):
            raise ValueError(
                f"Row count mismatch for {indicator['key']}: "
                f"base={len(base_rows)} indicator={len(indicator_rows)}"
            )

        for column in indicator["cols"]:
            if column not in indicator_fields:
                raise ValueError(f"Missing expected column in {input_path.name}: {column}")
            if column not in merged_fields:
                merged_fields.append(column)

        indicator_map = {row_key(row): row for row in indicator_rows}
        missing_keys = 0
        order_fallbacks = 0
        for index, base_row in enumerate(base_rows):
            key = base_keys[index]
            source_row = indicator_map.get(key)
            if source_row is None:
                missing_keys += 1
                if index < len(indicator_rows):
                    source_row = indicator_rows[index]
                    order_fallbacks += 1
                else:
                    for column in indicator["cols"]:
                        base_row[column] = ""
                    continue
            for column in indicator["cols"]:
                base_row[column] = source_row.get(column, "")

        summaries.append(
            f"{indicator['key']}: rows={len(indicator_rows)}, missing_key_matches={missing_keys}, "
            f"order_fallbacks={order_fallbacks}, input={input_path.name}"
        )

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(base_rows)

    SUMMARY_TXT.write_text("\n".join(summaries), encoding="utf-8")
    print(f"Merged CSV: {OUTPUT_CSV}")
    print(f"Summary TXT: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
