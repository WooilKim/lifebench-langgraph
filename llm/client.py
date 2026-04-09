"""Unified LLM client — returns a LangChain ChatModel for the configured provider."""
from llm.config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    GLM_BASE_URL,
    GLM_API_KEY,
    GLM_MODEL,
    CLAUDE_MODEL,
    GPT_MODEL,
)


def get_llm_client(provider=None):
    """Return a LangChain ChatModel for the given (or default) provider."""
    p = (provider or LLM_PROVIDER).lower()

    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=CLAUDE_MODEL,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=2048,
        )

    if p == "gpt":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=GPT_MODEL,
            api_key=OPENAI_API_KEY,
        )

    if p == "glm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=GLM_MODEL,
            api_key=GLM_API_KEY,
            base_url=GLM_BASE_URL,
        )

    raise ValueError(f"Unknown LLM provider: {p!r}. Choose claude | gpt | glm.")
