from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/platform.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Ray / Cloudera
    ray_local_mode: bool = True
    cdsw_head_address: str | None = None
    default_worker_cpu: float = 4.0
    default_worker_memory_gb: float = 16.0
    default_gpus_per_worker: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
