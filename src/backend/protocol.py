"""WebSocket message protocol constants and validation."""

VALID_REQUEST_TYPES = {
    "create_model",
    "set_optimizer",
    "set_custom_optimizer",
    "set_dataset",
    "set_custom_dataset",
    "start_training",
    "stop_training",
    "reset_model",
    "compute_hessian",
    "compute_hessian_eigenvalues",
    "compute_ntk",
    "compute_ntk_eigenvalues",
    "compute_pca_landscape",
    "compute_random_landscape",
    "solve_newton_step",
    "solve_linear_system",
    "get_model_summary",
    "adapt_model",
    "get_config",
    "update_config",
    "connect_remote",
    "disconnect_remote",
    "get_remote_status",
    "compute_weight_histogram",
    "compute_gradient_stats",
    "compute_activation_stats",
    "compute_layer_stats",
    "compute_fisher",
    "compute_fisher_eigenvalues",
    "compute_interpolation",
}

RESPONSE_TYPES = {
    "model_created",
    "dataset_ready",
    "response",
    "training_progress",
    "training_complete",
    "hessian_computed",
    "hessian_eigenvalues",
    "ntk_computed",
    "ntk_eigenvalues",
    "landscape_computed",
    "equation_solved",
    "model_summary",
    "model_adapted",
    "error",
    "status",
    "weight_histogram",
    "gradient_stats",
    "activation_stats",
    "layer_stats",
    "fisher_computed",
    "fisher_eigenvalues",
    "interpolation_computed",
}

# Types that are push messages (not direct responses to requests)
PUSH_TYPES = {"training_progress", "status"}


def make_response(msg_id, type_, payload):
    return {"type": type_, "msg_id": msg_id, "payload": payload}


def make_push(type_, payload):
    return {"type": type_, "payload": payload}


def make_error(msg_id, code, message, details=None):
    return {
        "type": "error",
        "msg_id": msg_id,
        "payload": {"code": code, "message": message, "details": details},
    }


def make_status(level, message):
    return {"type": "status", "payload": {"level": level, "message": message}}


def validate_message(msg):
    if not isinstance(msg, dict):
        raise ValueError("Message must be a JSON object")
    if "type" not in msg:
        raise ValueError("Message missing 'type' field")
    msg_type = msg["type"]
    if msg_type in VALID_REQUEST_TYPES and "msg_id" not in msg:
        raise ValueError(f"Request message '{msg_type}' missing 'msg_id' field")
    if "payload" not in msg:
        raise ValueError("Message missing 'payload' field")
    if not isinstance(msg["payload"], dict):
        raise ValueError("Payload must be a JSON object")
