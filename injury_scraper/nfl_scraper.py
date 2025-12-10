import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NFL_INJURIES_URL = "https://www.nfl.com/injuries/"

CSV_COLUMNS = [
    "id",
    "source",
    "player",
    "team",
    "position",
    "injury",
    "status",
    "text",
    "url",
    "timestamp",
]


class NFLInjuryScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        )

    def scrape(self, url: str = NFL_INJURIES_URL) -> List[dict]:
        logger.info(f"Scraping {url}")

        response = self.session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        injuries = []

        week_match = re.search(r"WEEK\s*(\d+)", response.text, re.IGNORECASE)
        week = int(week_match.group(1)) if week_match else 0

        report_wrap = soup.find("div", class_="nfl-o-injury-report__wrap")
        if not report_wrap:
            logger.warning("Could not find injury report")
            return injuries

        current_date = ""
        timestamp = datetime.now().isoformat()

        for element in report_wrap.children:
            if not hasattr(element, "name") or element.name is None:
                continue

            if element.name == "h2" and "d3-o-section-title" in element.get(
                "class", []
            ):
                current_date = element.get_text(strip=True)
                logger.info(f"Processing: {current_date}")

            elif (
                element.name == "section"
                and "nfl-o-injury-report__unit" in element.get("class", [])
            ):
                game_injuries = self._parse_game(element, week, timestamp)
                injuries.extend(game_injuries)

        logger.info(f"Scraped {len(injuries)} injuries from NFL.com")
        return injuries

    def _parse_game(self, section, week: int, timestamp: str) -> List[dict]:
        """Parse injuries from one game section."""
        injuries = []

        team_titles = section.find_all("div", class_="nfl-t-stats__title")
        teams = [t.get_text(strip=True) for t in team_titles if t.get_text(strip=True)]

        tables = section.find_all("table", class_=lambda x: x and "d3-o-table" in x)

        for i, table in enumerate(tables):
            team = teams[i] if i < len(teams) else "Unknown"

            tbody = table.find("tbody")
            if not tbody:
                continue

            for row in tbody.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 5:
                    continue

                player = cells[0].get_text(strip=True)
                position = cells[1].get_text(strip=True)
                injury = cells[2].get_text(strip=True) or "Unspecified"
                practice = cells[3].get_text(strip=True)
                status = cells[4].get_text(strip=True) or "Listed"

                if player:
                    if injury and injury != "Unspecified":
                        text = f"{player} ({injury}) - {status}"
                    else:
                        text = f"{player} - {practice}"

                    injuries.append(
                        {
                            "id": f"nfl_w{week}_{player.replace(' ', '_').lower()}",
                            "source": "NFL.com",
                            "player": player,
                            "team": team,
                            "position": position,
                            "injury": injury.lower().split(",")[0].strip(),
                            "status": status,
                            "text": text,
                            "url": NFL_INJURIES_URL,
                            "timestamp": timestamp,
                        }
                    )
                    logger.info(f"  {player} ({injury}) - {team} [{status}]")

        return injuries

    def to_csv(
        self, injuries: List[dict], output_path: Union[str, Path], append: bool = False
    ) -> Path:
        output_path = Path(output_path)
        mode = "a" if append else "w"
        file_exists = output_path.exists() and output_path.stat().st_size > 0

        with open(output_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not append or not file_exists:
                writer.writeheader()
            writer.writerows(injuries)

        logger.info(f"Saved {len(injuries)} injuries to {output_path}")
        return output_path


def scrape_nfl_injuries(
    output_csv: Optional[str] = None, append: bool = False
) -> List[dict]:
    scraper = NFLInjuryScraper()
    injuries = scraper.scrape()

    if output_csv:
        scraper.to_csv(injuries, output_csv, append=append)

    return injuries
