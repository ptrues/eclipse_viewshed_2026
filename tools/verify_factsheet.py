"""Compare the factsheet with the processed analysis outputs."""

import json
import pathlib
import re

import pandas as pd
from bs4 import BeautifulSoup

from eclipse_viewshed import eclipse, solar
from eclipse_viewshed.project import repository_root

ROOT = repository_root()

SHEET = ROOT / "reports" / "factsheet" / "factsheet.html"
PROCESSED = ROOT / "data" / "processed"
PUBLISHED = {
    "first": 19 + 18 / 60,
    "maximum": 20 + 13 / 60,
    "last": 21 + 5 / 60,
    "obscuration": 0.892,
}

failures = []


def check(condition, label, detail=""):
    """Record and print one verification result."""
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if not condition:
        failures.append(label)


def normalized_text(element):
    """Return an element's visible text with collapsed whitespace."""
    return " ".join(element.get_text(" ", strip=True).split())


html = SHEET.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")
saved = json.loads((PROCESSED / "01_eclipse_contacts.json").read_text())
live = eclipse.contacts()


# Validate the saved eclipse contacts against a fresh calculation and an
# external published reference.
print("\nEclipse geometry")
for live_key, saved_key, published_key in (
    ("first_contact_utc", "first_contact_cest", "first"),
    ("maximum_utc", "maximum_cest", "maximum"),
    ("last_contact_utc", "last_contact_cest", "last"),
):
    calculated = live[live_key] + solar.UTC_OFFSET_HOURS
    recorded = saved[saved_key]
    published = PUBLISHED[published_key]
    check(
        abs(calculated - recorded) * 60 < 0.5,
        f"{published_key} matches notebook 01",
        f"{solar.format_cest(calculated)} vs {solar.format_cest(recorded)}",
    )
    check(
        abs(calculated - published) * 60 <= 2,
        f"{published_key} is within two minutes of the published time",
        f"{solar.format_cest(calculated)} vs {solar.format_cest(published)}",
    )

check(
    abs(live["max_obscuration"] - PUBLISHED["obscuration"]) <= 0.01,
    "maximum coverage is within one percentage point of the published value",
)


# Read the three summary cards directly from their semantic layout.
print("\nFactsheet summary")
cards = soup.select(".strip > div")
check(len(cards) == 3, "three eclipse summary cards are present")
if len(cards) == 3:
    values = [normalized_text(card.select_one(".val")) for card in cards]
    expected = [
        saved["first_contact_hhmm"],
        saved["maximum_hhmm"],
        saved["last_contact_hhmm"],
    ]
    for label, actual, recorded in zip(("first contact", "maximum", "last contact"), values, expected):
        check(actual == recorded, label, f"{actual} vs {recorded}")

    coverage = re.search(r"(\d+)%", normalized_text(cards[1].select_one(".obsc")))
    displayed = int(coverage.group(1)) if coverage else -1
    check(
        abs(displayed - 100 * saved["max_obscuration"]) < 1,
        "maximum coverage",
        f"{displayed}% vs {100 * saved['max_obscuration']:.1f}%",
    )

maximum_altitude = saved["sun_alt_at_max_deg"]
lede = normalized_text(soup.select_one("section .lede"))
quoted_altitude = re.search(r"(\d+\.\d)\s*(?:°|degrees?)", lede)
check(
    quoted_altitude is not None
    and abs(float(quoted_altitude.group(1)) - maximum_altitude) < 0.05,
    "section 01 states the altitude at maximum",
)


# Compare the three-column site table with notebook 04 and notebook 05.
print("\nViewing-location table")
visibility = pd.read_csv(PROCESSED / "04_visibility.csv").set_index("name")
figure_table = pd.read_csv(PROCESSED / "05_factsheet_table.csv").set_index("name")
rows = soup.select("table.sites tbody tr")[1:]
check(len(rows) == len(visibility), "one table row per viewing location")

for row in rows:
    cells = row.find_all("td")
    name = cells[0].contents[0].strip()
    check(name in visibility.index, f"{name} is a processed viewing location")
    if name not in visibility.index:
        continue

    duration = int(cells[1].get_text(strip=True).split()[0])
    visible = cells[2].get_text(strip=True) == "✓"
    record = visibility.loc[name]
    check(
        duration == int(record["eclipse_minutes"] + 0.5),
        f"{name}: clear viewing time",
    )
    check(visible == bool(record["visible_at_max"]), f"{name}: maximum visible")

    check(name in figure_table.index, f"{name}: present in notebook 05 output")
    if name in figure_table.index:
        second = figure_table.loc[name]
        check(
            abs(second["clearance_at_max_deg"] - record["clearance_at_max"]) <= 0.02,
            f"{name}: notebook 04 and 05 clearance agrees",
        )
        check(
            abs(second["eclipse_minutes"] - record["eclipse_minutes"]) <= 0.3,
            f"{name}: notebook 04 and 05 duration agrees",
        )


# Check the factsheet's image links and accessibility text.
print("\nFactsheet assets")
images = soup.find_all("img")
check(len(images) == 4, "four images are present")
for image in images:
    source = image.get("data-source")
    if source:
        target = ROOT / source
        expected = pathlib.Path("../figures/factsheet") / target.name
        check(target.is_file(), f"source image exists: {target.name}")
        check(image.get("src") == expected.as_posix(), f"live image link: {target.name}")
    else:
        check(image.get("src", "").startswith("data:image/png;base64,"), "statue image is embedded")
    check(bool(image.get("alt", "").strip()), "image has alternative text")

body = re.sub(r"<!--.*?-->", "", html, flags=re.S)
check(not re.search(r"\[PLACEHOLDER\]|\[note:|\{describe", body), "no drafting text remains")
check("@page" in html, "print layout is defined")

print(f"\n{len(failures)} failure(s)")
raise SystemExit(1 if failures else 0)
