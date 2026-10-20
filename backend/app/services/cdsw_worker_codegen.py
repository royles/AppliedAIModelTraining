"""Generated Python snippets for CDSW worker engines (shared resolve_head logic)."""

from __future__ import annotations

from app.services.cai_networking import resolve_ray_gcs_port


def embedded_worker_resolve_head_source() -> str:
    """
    ``resolve_head()`` for ``launch_workers`` code strings.

    Cloudera CDSW: workers connect **to the master** using ``CDSW_MASTER_IP`` (see
    worker network communications doc). The session pins ``PLATFORM_RAY_HEAD_ADDRESS``
    at launch; if CAI's ``CDSW_MASTER_IP`` disagrees, use the launch pin.
    """
    default_port = resolve_ray_gcs_port()
    return f'''def resolve_head() -> str:
    """Ray GCS on master — connect via CDSW_MASTER_IP (Cloudera worker networking)."""
    pinned = os.environ.get("PLATFORM_RAY_HEAD_ADDRESS", "").strip() or FALLBACK_HEAD
    pin_host = pinned.split(":")[0].strip() if pinned else ""
    port_s = pinned.split(":", 1)[1].strip() if ":" in pinned else ""
    try:
        port = int(port_s or "{default_port}")
    except ValueError:
        port = {default_port}
    for key in ("CDSW_MASTER_IP", "CML_MASTER_IP"):
        ip = os.environ.get(key, "").strip()
        if not ip:
            continue
        if pin_host and ip != pin_host:
            print(
                f"[platform] WARN: {{key}}={{ip}} != launch head {{pinned}}; using launch head",
                flush=True,
            )
            break
        return f"{{ip}}:{{port}}"
    if pinned:
        return pinned if ":" in pinned else f"{{pin_host}}:{{port}}"
    return FALLBACK_HEAD
'''
