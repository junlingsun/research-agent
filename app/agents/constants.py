# Shared constants used across agent modules
# Kept here to avoid circular imports between research_agent, plan_agent, synthesize_agent

DEPTH_CONFIG = {
    "quick": {
        "max_queries": 2,
        "max_sources": 3,
        "max_iterations": 1,  # no retries — speed matters
        "max_revisions": 1,  # internal subagent revisions
    },
    "standard": {
        "max_queries": 3,
        "max_sources": 5,
        "max_iterations": 3,
        "max_revisions": 2,
    },
    "deep": {
        "max_queries": 5,
        "max_sources": 10,
        "max_iterations": 5,
        "max_revisions": 3,
    },
}

DEFAULT_DEPTH = "standard"


def get_depth_config(depth: str) -> dict:
    """Get config for a depth level, falling back to standard."""
    return DEPTH_CONFIG.get(depth, DEPTH_CONFIG[DEFAULT_DEPTH])
