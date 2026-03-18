"""
Usage: python scripts/run_phase1.py --league EPL
       python scripts/run_phase1.py --all
"""
import asyncio
import argparse

from src.calibration.phase1_worker import run_phase1
from src.common.config import Config

LEAGUE_IDS = {
    "EPL": 1204, "LaLiga": 1399, "SerieA": 1269, "Bundesliga": 1229,
    "Ligue1": 1221, "MLS": 1440, "Brasileirao": 1141, "Argentina": 1081,
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()

    if args.all:
        for name, lid in LEAGUE_IDS.items():
            print(f"\n{'='*40}\nProcessing {name} (league_id={lid})\n{'='*40}")
            result = await run_phase1(str(lid), config)
            print(f"  -> {'GO' if result else 'NO-GO'}")
    elif args.league:
        lid = LEAGUE_IDS.get(args.league)
        if not lid:
            print(f"Unknown league: {args.league}. Available: {list(LEAGUE_IDS.keys())}")
            return
        result = await run_phase1(str(lid), config)
        print(f"  -> {'GO' if result else 'NO-GO'}")
    else:
        print("Specify --league NAME or --all")
        print(f"Available leagues: {list(LEAGUE_IDS.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
