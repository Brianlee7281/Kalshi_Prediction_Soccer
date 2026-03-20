"""
Usage: python scripts/run_phase2.py --league EPL
       Finds next upcoming match and runs Phase 2 for it.
"""
import asyncio
import argparse

from dotenv import load_dotenv
load_dotenv()

from src.prematch.phase2_pipeline import run_phase2
from src.clients.goalserve import GoalserveClient
from src.common.config import Config

LEAGUE_IDS = {
    "EPL": 1204, "LaLiga": 1399, "SerieA": 1269, "Bundesliga": 1229,
    "Ligue1": 1221, "MLS": 1440, "Brasileirao": 1141, "Argentina": 1081,
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=str, required=True)
    args = parser.parse_args()

    config = Config.from_env()
    lid = LEAGUE_IDS[args.league]

    # Find next upcoming fixture from Goalserve
    gs = GoalserveClient(api_key=config.goalserve_api_key)
    fixtures = await gs.get_upcoming_fixtures(str(lid))
    await gs.close()

    if not fixtures:
        print(f"No upcoming fixtures for {args.league}")
        return

    fixture = fixtures[0]
    print(f"Next match: {fixture['home_team']} vs {fixture['away_team']}")
    print(f"Match ID: {fixture['match_id']}")
    print(f"Kickoff: {fixture['kickoff_utc']}")

    result = await run_phase2(
        match_id=fixture["match_id"],
        league_id=lid,
        home_team=fixture["home_team"],
        away_team=fixture["away_team"],
        kickoff_utc=fixture["kickoff_utc"],
        config=config,
    )

    print(f"\nPhase2Result:")
    print(f"  verdict: {result.verdict}")
    print(f"  a_H: {result.a_H:.4f}, a_A: {result.a_A:.4f}")
    print(f"  mu_H: {result.mu_H:.4f}, mu_A: {result.mu_A:.4f}")
    print(f"  method: {result.prediction_method}")
    print(f"  param_version: {result.param_version}")
    if result.kalshi_tickers:
        print(f"  kalshi_tickers: {result.kalshi_tickers}")
    if result.market_implied:
        print(f"  market_implied: H={result.market_implied.home_win:.3f} "
              f"D={result.market_implied.draw:.3f} A={result.market_implied.away_win:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
