from flask import Blueprint, jsonify, request

from .services import (
    PipelineConfigError,
    get_local_medical_pipeline,
    get_local_pipeline_runtime_status,
)


main_bp = Blueprint("main", __name__)


def _pipeline_health_response():
    pipeline_status = get_local_pipeline_runtime_status()
    http_status = 200 if pipeline_status.get("ready") else 503
    return (
        jsonify(
            {
                "status": "ok" if pipeline_status.get("ready") else "degraded",
                "service": "medical-local-qwen-backend",
                "pipeline": pipeline_status,
            }
        ),
        http_status,
    )


def _run_local_diagnosis(payload: dict):
    query = str(payload.get("query", "") or payload.get("symptoms", "")).strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    history = str(payload.get("history", "")).strip()
    limit = max(1, min(int(payload.get("limit", 5) or 5), 10))

    try:
        pipeline = get_local_medical_pipeline()
        result = pipeline.run(query, history=history, limit=limit)
        return jsonify(result)
    except PipelineConfigError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"local pipeline failed: {exc}"}), 500


@main_bp.route("/api/health", methods=["GET"])
def health_check():
    pipeline_status = get_local_pipeline_runtime_status()
    return jsonify(
        {
            "status": "healthy",
            "service": "medical-local-qwen-backend",
            "pipeline_ready": pipeline_status.get("ready", False),
        }
    )


@main_bp.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json(silent=True) or {}
    return _run_local_diagnosis(data)


@main_bp.route("/api/diagnose/health", methods=["GET"])
def diagnose_health():
    return _pipeline_health_response()


@main_bp.route("/api/diagnose/local-qwen-pipeline", methods=["POST"])
def diagnose_with_local_qwen_pipeline():
    data = request.get_json(silent=True) or {}
    return _run_local_diagnosis(data)


@main_bp.route("/api/diagnose/local-qwen-pipeline/health", methods=["GET"])
def local_qwen_pipeline_health():
    return _pipeline_health_response()
