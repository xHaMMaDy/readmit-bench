"""Download CMS DE-SynPUF data files (all 20 samples by default).

Usage:
    python -m readmit_bench.data.download                # default: all 20 samples
    python -m readmit_bench.data.download --samples 1 2 3
    python -m readmit_bench.data.download --samples 1-5 --no-extract

Files downloaded per sample:
    - 2008 Beneficiary Summary File
    - 2009 Beneficiary Summary File
    - 2010 Beneficiary Summary File   (NOTE: Sample 1 2010 file is a known broken link on CMS;
                                       the script will warn and skip it.)
    - 2008-2010 Inpatient Claims
    - 2008-2010 Outpatient Claims

Carrier and PDE files are NOT downloaded (not used by the readmission task).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

CMS_BASE = (
    "https://www.cms.gov/research-statistics-data-and-systems/"
    "downloadable-public-use-files/synpufs/downloads"
)

# Files that are KNOWN BROKEN on the CMS server. We skip them with a warning
# instead of failing the whole download. The Phase-2 cohort code will fall
# back to the nearest available beneficiary year.
KNOWN_BROKEN = {
    # The 2010 Sample-1 beneficiary file points to Sample-20's data on the CMS site.
    # Documented at https://github.com/OHDSI/ETL-CMS/issues/62
    "de1_0_2010_beneficiary_summary_file_sample_1.zip",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class FileSpec:
    sample: int
    kind: str  # "beneficiary_2008" | "beneficiary_2009" | "beneficiary_2010" | "inpatient" | "outpatient"
    filename: str

    @property
    def url(self) -> str:
        return f"{CMS_BASE}/{self.filename}"


def list_files_for_sample(sample: int) -> list[FileSpec]:
    """Return the 5 file specs we need for a given sample number."""
    return [
        FileSpec(sample, "beneficiary_2008", f"de1_0_2008_beneficiary_summary_file_sample_{sample}.zip"),
        FileSpec(sample, "beneficiary_2009", f"de1_0_2009_beneficiary_summary_file_sample_{sample}.zip"),
        FileSpec(sample, "beneficiary_2010", f"de1_0_2010_beneficiary_summary_file_sample_{sample}.zip"),
        FileSpec(sample, "inpatient", f"de1_0_2008_to_2010_inpatient_claims_sample_{sample}.zip"),
        FileSpec(sample, "outpatient", f"de1_0_2008_to_2010_outpatient_claims_sample_{sample}.zip"),
    ]


def parse_samples_arg(arg: str | list[str]) -> list[int]:
    """Parse '1-5' or ['1', '2', '3'] into [1, 2, 3, 4, 5]."""
    if isinstance(arg, str):
        if "-" in arg:
            lo, hi = arg.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(arg)]
    out: list[int] = []
    for item in arg:
        out.extend(parse_samples_arg(item))
    return sorted(set(out))


def download_one(spec: FileSpec, dest_dir: Path, retries: int = 3) -> tuple[FileSpec, str, int]:
    """Download a single file. Returns (spec, status, bytes_downloaded)."""
    out_path = dest_dir / spec.filename
    if spec.filename in KNOWN_BROKEN:
        return spec, "skipped_known_broken", 0

    if out_path.exists() and out_path.stat().st_size > 0:
        return spec, "already_present", out_path.stat().st_size

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(spec.url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = out_path.with_suffix(".zip.part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
            return spec, "downloaded", out_path.stat().st_size
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * attempt)
    return spec, f"failed: {last_err}", 0


def _canonicalize_csv_name(zip_filename: str) -> str:
    """Convert a zip filename to its canonical CSV inside-name.

    CMS occasionally ships a zip whose internal CSV is named e.g.
    'DE1_0_2010_Beneficiary_Summary_File_Sample_17 - Copy.csv'. We always emit
    the clean name derived from the zip filename so downstream globs work.
    """
    stem = Path(zip_filename).stem  # 'de1_0_2010_..._sample_17'
    parts = stem.split("_")
    # Capitalize tokens like the canonical CMS naming used inside good zips.
    fixed: list[str] = []
    for tok in parts:
        if tok.lower() in {"de1", "0"}:
            fixed.append(tok.upper().replace("DE1", "DE1") if tok.lower() == "de1" else tok)
        elif tok.isdigit():
            fixed.append(tok)
        elif tok.lower() == "to":
            fixed.append("to")
        else:
            fixed.append(tok.capitalize())
    return "_".join(fixed) + ".csv"


def extract_one(zip_path: Path, extract_dir: Path) -> Path | None:
    """Extract a zip into extract_dir. Returns path to extracted CSV (or None)."""
    if not zip_path.exists():
        return None
    canonical = _canonicalize_csv_name(zip_path.name)
    target = extract_dir / canonical
    if target.exists() and target.stat().st_size > 0:
        return target
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            logger.warning("No CSV inside %s", zip_path.name)
            return None
        # Read first CSV entry and write under canonical name (handles CMS's
        # mis-packaged 'X - Copy.csv' inside the zip).
        with zf.open(names[0]) as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                dst.write(chunk)
    return target


def download_samples(
    samples: list[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    extract: bool = True,
    workers: int = 8,
) -> dict[str, list]:
    """Download (and optionally extract) the requested samples in parallel.

    Returns a dict with keys 'downloaded', 'skipped', 'failed', 'extracted'.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = raw_dir / "csv"
    if extract:
        extract_dir.mkdir(parents=True, exist_ok=True)

    specs: list[FileSpec] = []
    for s in samples:
        specs.extend(list_files_for_sample(s))

    logger.info("Downloading %d files across %d sample(s) using %d workers...", len(specs), len(samples), workers)

    results: dict[str, list] = {"downloaded": [], "already_present": [], "skipped": [], "failed": [], "extracted": []}
    total_bytes = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, spec, raw_dir): spec for spec in specs}
        for fut in as_completed(futures):
            spec, status, nbytes = fut.result()
            total_bytes += nbytes
            mb = nbytes / (1024 * 1024)
            tag = status.split(":")[0]
            if tag == "downloaded":
                results["downloaded"].append(spec.filename)
                logger.info("✓ DL    %-65s %6.1f MB", spec.filename, mb)
            elif tag == "already_present":
                results["already_present"].append(spec.filename)
                logger.info("· keep  %-65s %6.1f MB (already present)", spec.filename, mb)
            elif tag.startswith("skipped"):
                results["skipped"].append(spec.filename)
                logger.warning("- skip  %-65s (known broken on CMS server)", spec.filename)
            else:
                results["failed"].append(f"{spec.filename}: {status}")
                logger.error("✗ FAIL  %-65s %s", spec.filename, status)

    dl_secs = time.time() - t0
    logger.info(
        "Download complete: %d downloaded, %d already present, %d skipped, %d failed in %.1fs (%.1f MB total)",
        len(results["downloaded"]),
        len(results["already_present"]),
        len(results["skipped"]),
        len(results["failed"]),
        dl_secs,
        total_bytes / (1024 * 1024),
    )

    if extract:
        logger.info("Extracting zips → %s", extract_dir)
        t1 = time.time()
        zips = sorted(raw_dir.glob("*.zip"))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for csv_path in ex.map(lambda z: extract_one(z, extract_dir), zips):
                if csv_path:
                    results["extracted"].append(csv_path.name)
        logger.info("Extracted %d CSVs in %.1fs", len(results["extracted"]), time.time() - t1)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CMS DE-SynPUF samples.")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["1-20"],
        help="Sample numbers to download. Examples: '--samples 1 2 3', '--samples 1-5', '--samples 1-20' (default: all 20).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Destination for raw zips. Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument("--no-extract", action="store_true", help="Skip CSV extraction step.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers. Default: 8.")
    args = parser.parse_args()

    samples = parse_samples_arg(args.samples)
    bad = [s for s in samples if s < 1 or s > 20]
    if bad:
        logger.error("Sample numbers must be 1..20; got %s", bad)
        return 2

    res = download_samples(
        samples=samples,
        raw_dir=args.raw_dir,
        extract=not args.no_extract,
        workers=args.workers,
    )
    if res["failed"]:
        logger.error("Some files failed: %s", res["failed"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
