"""Record a dashboard demo GIF + a hero screenshot using Playwright headless.

Outputs:
  reports/demo.gif        — animated walkthrough (form fill → submit → result)
  reports/dashboard.png   — hero screenshot of the score page after submit

Requires the Flask app running at http://127.0.0.1:5050.
Uses Pillow to assemble frames into an optimized GIF.
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_GIF = ROOT / "reports" / "demo.gif"
OUT_HERO = ROOT / "reports" / "dashboard.png"
URL = "http://127.0.0.1:5050/"

VIEW_W, VIEW_H = 1440, 900
SCALE = 0.6  # downscale frames for smaller GIF
FPS = 8


def main() -> None:
    frames: list[Image.Image] = []

    def snap(page) -> None:
        png = page.screenshot(type="png", full_page=False)
        im = Image.open(io.BytesIO(png)).convert("RGB")
        new_size = (int(im.width * SCALE), int(im.height * SCALE))
        im = im.resize(new_size, Image.LANCZOS)
        frames.append(im)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H}, device_scale_factor=1
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(400)
        for _ in range(6):
            snap(page)
            page.wait_for_timeout(120)

        # Fill the score form (all fields are name=...)
        fields = {
            "age_at_admit": "78",
            "los_days": "9",
            "num_diagnoses": "11",
            "num_procedures": "3",
            "prior_6mo_inpatient_count": "2",
            "days_since_last_discharge_imputed": "14",
            "admit_month": "11",
            "admit_dow": "5",
            "state_code": "36",
            "admit_dx_code": "4280",
            "drg_code": "291",
        }
        for name, value in fields.items():
            page.fill(f'input[name="{name}"]', value)
            page.wait_for_timeout(40)
            snap(page)

        page.select_option('select[name="sex"]', "M")
        page.select_option('select[name="race"]', "Black")
        page.select_option('select[name="admit_dx_chapter"]', "Circulatory")
        snap(page)

        for cb in (
            "is_weekend_admit",
            "has_prior_admit",
            "chronic_chf",
            "chronic_diabetes",
            "chronic_ckd",
            "chronic_ihd",
            "chronic_copd",
        ):
            with contextlib.suppress(Exception):
                page.check(f'input[name="{cb}"]')
            page.wait_for_timeout(30)
            snap(page)

        # Submit and capture the result render
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(400)
        for _ in range(10):
            snap(page)
            page.wait_for_timeout(160)

        # Hero screenshot (full result page, full viewport)
        page.screenshot(path=str(OUT_HERO), full_page=False)

        ctx.close()
        browser.close()

    # Assemble GIF
    duration_ms = int(1000 / FPS)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size/1024:.1f} KB, {len(frames)} frames)")
    print(f"wrote {OUT_HERO} ({OUT_HERO.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.1f}s")
