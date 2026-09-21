import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

from config.settings import get_settings

logger = logging.getLogger("api_logger")


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class CostTrackingService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def estimate(self, total_tokens: int | None, output_text: str = "") -> CostEstimate:
        tokens = max(int(total_tokens or 0), 0)
        output_tokens = max(len(output_text.split()), 1) if output_text else 0
        if tokens and output_tokens > tokens:
            output_tokens = tokens
        input_tokens = max(tokens - output_tokens, 0)
        cost = ((input_tokens / 1000.0) * self._settings.COST_INPUT_TOKEN_USD_PER_1K) + (
            (output_tokens / 1000.0) * self._settings.COST_OUTPUT_TOKEN_USD_PER_1K
        )
        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=tokens,
            estimated_cost_usd=round(cost, 8),
        )


class MetricsService:
    _lock = threading.Lock()
    _counters: Dict[str, float] = defaultdict(float)
    _latencies: Dict[str, list[float]] = defaultdict(list)

    @classmethod
    def increment(cls, name: str, value: float = 1.0) -> None:
        with cls._lock:
            cls._counters[name] += value

    @classmethod
    def observe_latency(cls, name: str, latency_ms: float) -> None:
        with cls._lock:
            bucket = cls._latencies[name]
            bucket.append(float(latency_ms))
            if len(bucket) > 1000:
                del bucket[: len(bucket) - 1000]

    @classmethod
    def render_prometheus(cls) -> str:
        lines: list[str] = []
        with cls._lock:
            for name, value in sorted(cls._counters.items()):
                metric = cls._normalize(name)
                lines.append(f"# TYPE {metric} counter")
                lines.append(f"{metric} {value}")
            for name, values in sorted(cls._latencies.items()):
                metric = cls._normalize(name)
                count = len(values)
                total = sum(values)
                avg = total / count if count else 0.0
                lines.append(f"# TYPE {metric}_milliseconds summary")
                lines.append(f"{metric}_milliseconds_count {count}")
                lines.append(f"{metric}_milliseconds_sum {round(total, 6)}")
                lines.append(f"{metric}_milliseconds_avg {round(avg, 6)}")
        return "\n".join(lines) + "\n"

    @classmethod
    def log_event(cls, event: str, **fields: object) -> None:
        payload = {"event": event, **fields}
        logger.info(json.dumps(payload, default=str))

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        return f"rag_{normalized}"


def configure_optional_tracing() -> None:
    settings = get_settings()
    if settings.LANGSMITH_TRACING:
        import os

        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME}))
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    except Exception as exc:
        logger.warning("OpenTelemetry configuration skipped: %s", exc)
