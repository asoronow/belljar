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
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from belljar.config import BelljarConfig
from belljar.pipeline.align import AlignStep
from belljar.pipeline.collate import CollateStep
from belljar.pipeline.count import CountStep
from belljar.pipeline.detect import DetectStep
from belljar.pipeline.estimate import EstimateStep
from belljar.pipeline.max_projection import MaxProjectionStep
from belljar.pipeline.sharpen import SharpenStep

if TYPE_CHECKING:
    from collections.abc import Callable

    from belljar.pipeline.base import PipelineStep
    from belljar.types import StepResult

logger = logging.getLogger(__name__)

INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601
PARSE_ERROR = -32700
SERVER_ERROR = -32000

STEP_CLASSES: dict[str, type[PipelineStep[Any]]] = {
    "max_projection": MaxProjectionStep,
    "sharpen": SharpenStep,
    "align": AlignStep,
    "detect": DetectStep,
    "count": CountStep,
    "collate": CollateStep,
    "estimate": EstimateStep,
}


class InvalidParams(ValueError):
    """Request params do not fit the method; reported as JSON-RPC -32602."""

    def __init__(self, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.data = data


def _validation_messages(error: ValidationError) -> list[str]:
    """One ``field: problem`` line per pydantic error."""
    lines: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "params"
        lines.append(f"{location}: {item['msg']}")
    return lines


def _step_result_to_dict(result: StepResult) -> dict[str, Any]:
    """Convert a StepResult to a JSON-serializable dict."""
    return {
        "success": result.success,
        "output_path": result.output_path,
        "metrics": result.metrics,
        "errors": result.errors,
        "warnings": result.warnings,
    }


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


class BelljarServer:
    """JSON-RPC server for the Belljar Python backend."""

    def __init__(self, config: BelljarConfig | None = None) -> None:
        self.config = config or BelljarConfig()
        self._handlers: dict[str, Callable[[Any], Any]] = {}
        self._steps: dict[str, PipelineStep[Any]] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all RPC method handlers; each takes the request params."""
        self._handlers["config.get"] = self._get_config
        self._handlers["config.set"] = self._set_config
        self._handlers["ping"] = self._ping
        self._handlers["pipeline.validate"] = self._validate_step

        for step_name in STEP_CLASSES:
            self._handlers[f"pipeline.{step_name}"] = self._pipeline_handler(step_name)

    def _get_step(self, step_name: str) -> PipelineStep[Any]:
        """Lazily instantiate pipeline steps."""
        if step_name not in self._steps:
            cls = STEP_CLASSES.get(step_name)
            if cls is None:
                raise InvalidParams(f"Unknown pipeline step: {step_name}")
            self._steps[step_name] = cls(self.config)
        return self._steps[step_name]

    def _pipeline_handler(self, step_name: str) -> Callable[[Any], dict[str, Any]]:
        def handler(params: Any) -> dict[str, Any]:
            step = self._get_step(step_name)
            try:
                model = step.params_class.model_validate(_require_object(params))
            except ValidationError as e:
                raise _invalid_params(f"pipeline.{step_name}", e) from e
            return _step_result_to_dict(step.run(self.send_progress, model))

        return handler

    def _validate_step(self, params: Any) -> dict[str, Any]:
        """Validate a step's params without running it.

        Params: ``{"step": <name>, "params": {...}}``. Model errors and
        filesystem errors are both returned in ``errors``.
        """
        request = _require_object(params)
        step_name = request.get("step")
        if not isinstance(step_name, str):
            raise InvalidParams("step: field required")
        step = self._get_step(step_name)
        try:
            model = step.params_class.model_validate(request.get("params", {}))
        except ValidationError as e:
            return {"valid": False, "errors": _validation_messages(e)}
        errors = step.validate_inputs(model)
        return {"valid": len(errors) == 0, "errors": errors}

    def _get_config(self, params: Any = None) -> dict[str, Any]:
        return json.loads(self.config.model_dump_json())

    def _set_config(self, params: Any) -> dict[str, Any]:
        """Apply a partial update; nested objects merge field by field."""
        merged = _deep_merge(self._get_config(), _require_object(params))
        try:
            self.config = BelljarConfig.model_validate(merged)
        except ValidationError as e:
            raise _invalid_params("config.set", e) from e
        self._steps.clear()
        return self._get_config()

    def _ping(self, params: Any = None) -> str:
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
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        # Notifications (no id) don't get responses
        if request_id is None:
            return None

        if not isinstance(method, str):
            return _error(request_id, INVALID_REQUEST, "Invalid Request: method must be a string")
        if not isinstance(params, (dict, list)):
            return _error(
                request_id, INVALID_REQUEST, "Invalid Request: params must be an object or array"
            )

        handler = self._handlers.get(method)
        if handler is None:
            return _error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            result = handler(params)
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        except InvalidParams as e:
            return _error(request_id, INVALID_PARAMS, str(e), data=e.data)
        except Exception as e:
            logger.exception("Error handling %s", method)
            return _error(request_id, SERVER_ERROR, str(e), data=traceback.format_exc())

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
                _write(_error(None, PARSE_ERROR, f"Parse error: {e}"))
                continue

            if not isinstance(request, dict):
                _write(_error(None, INVALID_REQUEST, "Invalid Request: must be a JSON object"))
                continue

            response = self._handle_request(request)
            if response is not None:
                _write(response)

        logger.info("Belljar server shutting down")


def _invalid_params(method: str, error: ValidationError) -> InvalidParams:
    messages = _validation_messages(error)
    return InvalidParams(
        f"Invalid params for {method}: " + "; ".join(messages), data="\n".join(messages)
    )


def _require_object(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise InvalidParams("params must be a JSON object")
    return params


def _error(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "error": error, "id": request_id}


def _write(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


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
