"""Tests for JSON-RPC request handling in the server."""

import json

import pytest

from belljar.server import BelljarServer


def _request(method: str, params=None, request_id: int = 1) -> dict:
    request = {"jsonrpc": "2.0", "method": method, "id": request_id}
    if params is not None:
        request["params"] = params
    return request


@pytest.fixture
def server() -> BelljarServer:
    return BelljarServer()


class TestInvalidParams:
    def test_missing_field_is_invalid_params(self, server: BelljarServer) -> None:
        response = server._handle_request(
            _request("pipeline.count", {"predictions_dir": "/p", "output_dir": "/o"})
        )
        assert response is not None
        assert response["error"]["code"] == -32602
        assert "annotations_dir" in response["error"]["message"]
        assert "annotations_dir" in response["error"]["data"]

    def test_unknown_field_is_invalid_params(self, server: BelljarServer) -> None:
        response = server._handle_request(
            _request(
                "pipeline.count",
                {
                    "predictions_dir": "/p",
                    "annotations_dir": "/a",
                    "output_dir": "/o",
                    "structure_map": {},
                },
            )
        )
        assert response is not None
        assert response["error"]["code"] == -32602
        assert "structure_map" in response["error"]["message"]

    def test_non_object_params_is_invalid_params(self, server: BelljarServer) -> None:
        response = server._handle_request(_request("pipeline.count", ["/p", "/a", "/o"]))
        assert response is not None
        assert response["error"]["code"] == -32602

    def test_error_response_is_json_serializable(self, server: BelljarServer) -> None:
        response = server._handle_request(_request("pipeline.count", {}))
        json.dumps(response)


class TestMethodDispatch:
    def test_unknown_method(self, server: BelljarServer) -> None:
        response = server._handle_request(_request("pipeline.nonsense", {}))
        assert response is not None
        assert response["error"]["code"] == -32601
        assert response["id"] == 1

    def test_ping(self, server: BelljarServer) -> None:
        response = server._handle_request(_request("ping"))
        assert response == {"jsonrpc": "2.0", "result": "pong", "id": 1}

    def test_notification_gets_no_response(self, server: BelljarServer) -> None:
        assert server._handle_request({"jsonrpc": "2.0", "method": "ping"}) is None

    def test_parse_error(self, server: BelljarServer, capsys) -> None:
        import io
        import sys

        sys.stdin = io.StringIO("{not json\n")
        try:
            server.run()
        finally:
            sys.stdin = sys.__stdin__
        out = capsys.readouterr().out.strip().splitlines()
        response = json.loads(out[-1])
        assert response["error"]["code"] == -32700
        assert response["id"] is None


class TestValidate:
    def test_validate_reports_model_errors(self, server: BelljarServer) -> None:
        response = server._handle_request(
            _request("pipeline.validate", {"step": "count", "params": {"output_dir": "/o"}})
        )
        assert response is not None
        result = response["result"]
        assert result["valid"] is False
        assert any("annotations_dir" in e for e in result["errors"])

    def test_validate_reports_filesystem_errors(self, server: BelljarServer, tmp_path) -> None:
        response = server._handle_request(
            _request(
                "pipeline.validate",
                {
                    "step": "count",
                    "params": {
                        "predictions_dir": str(tmp_path / "nope"),
                        "annotations_dir": str(tmp_path),
                        "output_dir": str(tmp_path / "out"),
                    },
                },
            )
        )
        assert response is not None
        result = response["result"]
        assert result["valid"] is False
        assert any("does not exist" in e for e in result["errors"])

    def test_validate_unknown_step_is_invalid_params(self, server: BelljarServer) -> None:
        response = server._handle_request(
            _request("pipeline.validate", {"step": "nonsense", "params": {}})
        )
        assert response is not None
        assert response["error"]["code"] == -32602


class TestConfig:
    def test_set_partial_update(self, server: BelljarServer) -> None:
        response = server._handle_request(
            _request("config.set", {"registration": {"rigid_iterations": 7}})
        )
        assert response is not None
        assert response["result"]["registration"]["rigid_iterations"] == 7
        assert server.config.registration.rigid_iterations == 7
        assert server.config.registration.affine_iterations == 100

    def test_set_unknown_key_is_invalid_params(self, server: BelljarServer) -> None:
        response = server._handle_request(
            _request("config.set", {"registration": {"rigid_learning_rate": 1.0}})
        )
        assert response is not None
        assert response["error"]["code"] == -32602
        assert "rigid_learning_rate" in response["error"]["message"]


class TestInvalidRequest:
    def test_non_string_method_is_invalid_request_and_server_survives(
        self, server: BelljarServer, capsys
    ) -> None:
        import io
        import sys

        sys.stdin = io.StringIO(
            '{"jsonrpc": "2.0", "id": 1, "method": ["x"]}\n'
            '{"jsonrpc": "2.0", "id": 2, "method": "ping"}\n'
        )
        try:
            server.run()
        finally:
            sys.stdin = sys.__stdin__
        lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
        assert lines[0]["error"]["code"] == -32600
        assert lines[0]["id"] == 1
        assert lines[1] == {"jsonrpc": "2.0", "result": "pong", "id": 2}

    def test_missing_method_is_invalid_request(self, server: BelljarServer) -> None:
        response = server._handle_request({"jsonrpc": "2.0", "id": 1})
        assert response is not None
        assert response["error"]["code"] == -32600

    def test_scalar_params_is_invalid_request(self, server: BelljarServer) -> None:
        response = server._handle_request(_request("ping", params=5))
        assert response is not None
        assert response["error"]["code"] == -32600

    def test_non_object_request_is_invalid_request(self, server: BelljarServer, capsys) -> None:
        import io
        import sys

        sys.stdin = io.StringIO("[1, 2]\n")
        try:
            server.run()
        finally:
            sys.stdin = sys.__stdin__
        response = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert response["error"]["code"] == -32600


class TestInternalValidationError:
    def test_validation_error_inside_run_is_server_error(self, server: BelljarServer) -> None:
        from pydantic import BaseModel

        class Inner(BaseModel):
            n: int

        def broken(params):
            Inner(n="not an int")

        server._handlers["pipeline.broken"] = broken
        response = server._handle_request(_request("pipeline.broken", {}))
        assert response is not None
        assert response["error"]["code"] == -32000
