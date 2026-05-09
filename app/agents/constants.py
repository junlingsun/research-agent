# Shared constants used across agent modules
# Kept here to avoid circular imports between research_agent, plan_agent, synthesize_agent

DEPTH_CONFIG = {
    "quick":    {"max_queries": 2, "max_sources": 3},
    "standard": {"max_queries": 3, "max_sources": 5},
    "deep":     {"max_queries": 5, "max_sources": 10},
}