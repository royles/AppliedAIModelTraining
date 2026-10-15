from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_database_url() -> str:
    db_path = (REPO_ROOT / "data" / "platform.db").resolve()
    return f"sqlite:///{db_path}"


def effective_cors_origins(cors_origins: str) -> list[str]:
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    port = os.environ.get("CDSW_APP_PORT")
    ip = os.environ.get("CDSW_IP_ADDRESS") or os.environ.get("CML_IP_ADDRESS")
    if port:
        for host in ("127.0.0.1", "localhost"):
            origins.append(f"http://{host}:{port}")
        if ip:
            origins.append(f"http://{ip}:{port}")
    # de-dupe preserving order
    return list(dict.fromkeys(origins))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = default_database_url()
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Ray / Cloudera
    ray_local_mode: bool = True
    cdsw_head_address: str | None = None
    default_worker_cpu: float = Field(
        default=4.0,
        description="CPU cores passed to CDSW launch_workers (match CML_RESOURCE_PROFILE_ID).",
    )
    default_worker_memory_gb: float = Field(
        default=16.0,
        ge=1.0,
        description="GiB passed to launch_workers; must match a profile with enough RAM.",
    )
    default_gpus_per_worker: int = Field(default=1, ge=0, le=8)


@lru_cache
def get_settings() -> Settings:
    return Settings()
