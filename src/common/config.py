import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_key_path(env_path: str, default: str = "keys/kalshi_private.pem") -> str:
    """Resolve Kalshi key path, falling back to local if Docker path doesn't exist."""
    if Path(env_path).exists():
        return env_path
    if Path(default).exists():
        return default
    return env_path  # return original so error message is clear


@dataclass
class Config:
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "soccer_trading"
    db_user: str = "postgres"
    db_password: str = "postgres"
    redis_host: str = "localhost"
    redis_port: int = 6379
    goalserve_api_key: str = ""
    kalshi_api_key: str = ""
    kalshi_private_key_path: str = "keys/kalshi_private.pem"
    odds_api_key: str = ""
    trading_mode: str = "paper"  # "paper" | "live"

    @classmethod
    def from_env(cls) -> "Config":
        raw_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")
        return cls(
            db_host=os.environ.get("DB_HOST", "localhost"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME", "soccer_trading"),
            db_user=os.environ.get("DB_USER", "postgres"),
            db_password=os.environ.get("DB_PASSWORD", "postgres"),
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            goalserve_api_key=os.environ.get("GOALSERVE_API_KEY", ""),
            kalshi_api_key=os.environ.get("KALSHI_API_KEY", ""),
            kalshi_private_key_path=_resolve_key_path(raw_key_path),
            odds_api_key=os.environ.get("ODDS_API_KEY", ""),
            trading_mode=os.environ.get("TRADING_MODE", "paper"),
        )
