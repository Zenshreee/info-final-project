from .nfl_scraper import NFLInjuryScraper, scrape_nfl_injuries
from .scraper import TwitterInjuryScraper, scrape_injuries

__version__ = "1.0.0"

__all__ = [
    "NFLInjuryScraper",
    "scrape_nfl_injuries",
    "TwitterInjuryScraper",
    "scrape_injuries",
]
