class AgentFailure(RuntimeError):
    """Reserved for future explicit primitive failure handling."""


class EvalParseError(ValueError):
    """Reserved for callers that want strict evaluation parsing."""
