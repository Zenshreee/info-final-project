import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

OUTPUT_CSV = "injuries.csv"


def main():
    from injury_scraper import scrape_nfl_injuries, scrape_injuries

    print("=" * 50)
    print("NFL Injury Scraper")
    print("=" * 50)

    all_injuries = []

    print("\n[1/2] Scraping NFL.com...")
    nfl_injuries = scrape_nfl_injuries()
    all_injuries.extend(nfl_injuries)
    print(f"✓ Found {len(nfl_injuries)} from NFL.com")

    if os.environ.get("TWITTER_BEARER_TOKEN"):
        print("\n[2/2] Scraping Twitter @UnderdogNFL...")
        try:
            twitter_injuries = scrape_injuries(injury_limit=25)
            all_injuries.extend(twitter_injuries)
            print(f"✓ Found {len(twitter_injuries)} from Twitter")
        except Exception as e:
            print(f"⚠ Twitter error: {e}")
    else:
        print("\n[2/2] Skipping Twitter (no TWITTER_BEARER_TOKEN)")

    if all_injuries:
        from injury_scraper.nfl_scraper import NFLInjuryScraper

        scraper = NFLInjuryScraper()
        scraper.to_csv(all_injuries, OUTPUT_CSV)

        print("\n" + "=" * 50)
        print(f"✓ Total: {len(all_injuries)} injuries")
        print(f"✓ Saved to: {OUTPUT_CSV}")

    return all_injuries


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    main()
