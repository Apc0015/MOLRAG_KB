"""Configuration management for MolRAG"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Main configuration class that loads from YAML files and environment variables"""

    # Paths
    config_dir: Path = Field(default=Path("config"))
    models_config_path: Path = Field(default=Path("config/models.yaml"))
    kg_config_path: Path = Field(default=Path("config/knowledge_graphs.yaml"))

    # Environment
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    drugbank_api_key: Optional[str] = Field(default=None, env="DRUGBANK_API_KEY")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="molrag", env="NEO4J_DATABASE")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")

    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # PostgreSQL
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="molrag_metadata", env="POSTGRES_DB")

    # Model paths
    kpgt_model_path: Optional[Path] = Field(
        default=None, env="KPGT_MODEL_PATH"
    )
    kano_model_path: Optional[Path] = Field(
        default=None, env="KANO_MODEL_PATH"
    )

    # Performance
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    device: str = Field(default="cuda", env="DEVICE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_models_config(self) -> Dict[str, Any]:
        """Load models configuration"""
        return self.load_yaml_config(self.models_config_path)

    def load_kg_config(self) -> Dict[str, Any]:
        """Load knowledge graphs configuration"""
        return self.load_yaml_config(self.kg_config_path)

    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration"""
        return self.load_models_config()

    @property
    def knowledge_graphs(self) -> Dict[str, Any]:
        """Get knowledge graphs configuration"""
        return self.load_kg_config()


def load_config(
    config_dir: Optional[Path] = None,
    env_file: Optional[Path] = None
) -> Config:
    """
    Load configuration from YAML files and environment variables

    Args:
        config_dir: Directory containing configuration files
        env_file: Path to .env file

    Returns:
        Config object
    """
    if env_file and env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    config = Config()

    if config_dir:
        config.config_dir = config_dir
        config.models_config_path = config_dir / "models.yaml"
        config.kg_config_path = config_dir / "knowledge_graphs.yaml"

    return config
