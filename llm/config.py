"""LLM provider configuration from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "")
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

LIFEBENCH_DATA_DIR = os.getenv(
    "LIFEBENCH_DATA_DIR",
    str(os.path.expanduser("~/source/LifeBench/life_bench_data/data_en")),
)
