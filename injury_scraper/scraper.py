import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import tweepy

logger = logging.getLogger(__name__)

INJURY_KEYWORDS = {
    "concussion",
    "ankle",
    "hamstring",
    "knee",
    "shoulder",
    "back",
    "hip",
    "groin",
    "calf",
    "quad",
    "quadricep",
    "thigh",
    "foot",
    "toe",
    "neck",
    "rib",
    "ribs",
    "elbow",
    "wrist",
    "hand",
    "finger",
    "achilles",
    "acl",
    "mcl",
    "pcl",
    "lcl",
    "meniscus",
    "labrum",
    "rotator cuff",
    "pec",
    "pectoral",
    "oblique",
    "bicep",
    "tricep",
    "illness",
    "sick",
    "flu",
    "covid",
    "collarbone",
    "arm",
    "leg",
    "head",
    "eye",
    "thumb",
}

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


def detect_injury(text: str) -> Optional[str]:
    matches = re.findall(r"\(([^)]+)\)", text, re.IGNORECASE)
    for match in matches:
        if match.lower().strip() in INJURY_KEYWORDS:
            return match.lower().strip()
    return None


def extract_player_name(text: str) -> str:
    match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)+)\s*\(", text)
    if match:
        return match.group(1)
    return ""


class TwitterInjuryScraper:
    def __init__(self, bearer_token: Optional[str] = None):
        token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        if not token:
            raise ValueError("Bearer token required. Set TWITTER_BEARER_TOKEN env var.")
        self._client = tweepy.Client(bearer_token=token)

    def scrape(
        self,
        username: str = "UnderdogNFL",
        injury_limit: int = 25,
        max_tweets: int = 100,
    ) -> List[dict]:
        logger.info(f"Scraping @{username} for {injury_limit} injuries")

        injuries = []
        timestamp = datetime.now().isoformat()

        user = self._client.get_user(username=username)
        if not user.data:
            raise ValueError(f"User @{username} not found")

        user_id = user.data.id
        logger.info(f"Found @{username} (ID: {user_id})")

        paginator = tweepy.Paginator(
            self._client.get_users_tweets,
            id=user_id,
            max_results=min(100, max_tweets),
            tweet_fields=["created_at", "text"],
            exclude=["retweets", "replies"],
        )

        for tweet in paginator.flatten(limit=max_tweets):
            if len(injuries) >= injury_limit:
                break

            text = tweet.text or ""
            injury = detect_injury(text)

            if injury:
                player = extract_player_name(text)

                injuries.append(
                    {
                        "id": str(tweet.id),
                        "source": "Twitter",
                        "player": player,
                        "team": "",
                        "position": "",
                        "injury": injury,
                        "status": "",
                        "text": text,
                        "url": f"https://x.com/{username}/status/{tweet.id}",
                        "timestamp": timestamp,
                    }
                )
                logger.info(
                    f"[{len(injuries)}/{injury_limit}] {injury}: {text[:50]}..."
                )

        logger.info(f"Found {len(injuries)} injury tweets")
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


def scrape_injuries(
    username: str = "UnderdogNFL",
    injury_limit: int = 25,
    bearer_token: Optional[str] = None,
    output_csv: Optional[str] = None,
    append: bool = False,
) -> List[dict]:
    scraper = TwitterInjuryScraper(bearer_token=bearer_token)
    injuries = scraper.scrape(username=username, injury_limit=injury_limit)

    if output_csv:
        scraper.to_csv(injuries, output_csv, append=append)

    return injuries
