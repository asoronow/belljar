"""JSON-RPC 2.0 server communicating over stdin/stdout.

Replaces the pattern where each pipeline step spawns a separate PythonShell subprocess.
A single long-lived server process is started by Electron and handles all RPC requests.

Protocol:
    - Requests and responses are newline-delimited JSON objects.
    - Progress notifications are sent as JSON-RPC notifications (no id).
    - Stderr is reserved for logging/debugging (not parsed by Electron).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from typing import Any, Callable

from belljar.config import BelljarConfig
from belljar.pipeline.align import AlignStep
from belljar.pipeline.collate import CollateStep
from belljar.pipeline.count import CountStep
from belljar.pipeline.detect import DetectStep
from belljar.pipeline.estimate import EstimateStep
from belljar.pipeline.max_projection import MaxProjectionStep
from belljar.pipeline.sharpen import SharpenStep
from belljar.types import StepResult

logger = logging.getLogger(__name__)


def _step_result_to_dict(result: StepResult) -> dict[str, Any]:
    """Convert a StepResult to a JSON-serializable dict."""
    return {
        "success": result.success,
        "output_path": result.output_path,
        "metrics": result.metrics,
        "errors": result.errors,
        "warnings": result.warnings,
    }


class BelljarServer:
    """JSON-RPC server for the Belljar Python backend."""

    def __init__(self, config: BelljarConfig | None = None) -> None:
        self.config = config or BelljarConfig()
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._steps: dict[str, Any] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all RPC method handlers."""
        self._handlers["config.get"] = self._get_config
        self._handlers["config.set"] = self._set_config
        self._handlers["ping"] = self._ping

        # Pipeline step handlers
        self._handlers["pipeline.max_projection"] = self._run_max_projection
        self._handlers["pipeline.sharpen"] = self._run_sharpen
        self._handlers["pipeline.align"] = self._run_align
        self._handlers["pipeline.detect"] = self._run_detect
        self._handlers["pipeline.count"] = self._run_count
        self._handlers["pipeline.collate"] = self._run_collate
        self._handlers["pipeline.estimate"] = self._run_estimate

        # Validation
        self._handlers["pipeline.validate"] = self._validate_step

    def _get_step(self, step_name: str) -> Any:
        """Lazily instantiate pipeline steps."""
        if step_name not in self._steps:
            step_classes = {
                "max_projection": MaxProjectionStep,
                "sharpen": SharpenStep,
                "align": AlignStep,
                "detect": DetectStep,
                "count": CountStep,
                "collate": CollateStep,
                "estimate": EstimateStep,
            }
            cls = step_classes.get(step_name)
            if cls is None:
                raise ValueError(f"Unknown pipeline step: {step_name}")
            self._steps[step_name] = cls(self.config)
        return self._steps[step_name]

    def _run_max_projection(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("max_projection")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _run_sharpen(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("sharpen")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _run_align(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("align")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _run_detect(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("detect")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _run_count(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("count")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _run_collate(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("collate")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _run_estimate(self, **kwargs: Any) -> dict[str, Any]:
        step = self._get_step("estimate")
        return _step_result_to_dict(step.run(self.send_progress, **kwargs))

    def _validate_step(self, step: str, **kwargs: Any) -> dict[str, Any]:
        """Validate inputs for a pipeline step without running it."""
        pipeline_step = self._get_step(step)
        errors = pipeline_step.validate_inputs(**kwargs)
        return {"valid": len(errors) == 0, "errors": errors}

    def _get_config(self) -> dict[str, Any]:
        return json.loads(self.config.model_dump_json())

    def _set_config(self, **updates: Any) -> dict[str, Any]:
        merged = json.loads(self.config.model_dump_json())
        merged.update(updates)
        self.config = BelljarConfig.model_validate(merged)
        return self._get_config()

    def _ping(self) -> str:
        return "pong"

    def send_progress(self, current: int, total: int, message: str) -> None:
        """Send a progress notification to the frontend."""
        notification = {
            "jsonrpc": "2.0",
            "method": "progress",
            "params": {"current": current, "total": total, "message": message},
        }
        sys.stdout.write(json.dumps(notification) + "\n")
        sys.stdout.flush()

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        # Notifications (no id) don't get responses
        if request_id is None:
            return None

        handler = self._handlers.get(method)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": request_id,
            }

        try:
            if isinstance(params, dict):
                result = handler(**params)
            elif isinstance(params, list):
                result = handler(*params)
            else:
                result = handler()
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        except Exception as e:
            logger.exception("Error handling %s", method)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e),
                    "data": traceback.format_exc(),
                },
                "id": request_id,
            }

    def run(self) -> None:
        """Main loop: read JSON-RPC requests from stdin, write responses to stdout."""
        logger.info("Belljar server started")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                    "id": None,
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
                continue

            response = self._handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        logger.info("Belljar server shutting down")


def main() -> None:
    """Entry point for the JSON-RPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Log to stderr so it doesn't interfere with JSON-RPC on stdout
    )
    config = BelljarConfig()
    server = BelljarServer(config)
    server.run()


if __name__ == "__main__":
    main()
