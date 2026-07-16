"""Reports routes — download and manage generated Excel reports."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, ORJSONResponse

from src.core.config import get_settings

router = APIRouter()


def _reports_dir() -> Path:
    d = Path(get_settings().reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("/reports")
async def list_reports() -> ORJSONResponse:
    """List all generated reports."""
    reports_dir = _reports_dir()
    files = []
    for f in sorted(reports_dir.iterdir(), reverse=True):
        if f.is_file() and f.suffix in (".xlsx", ".csv"):
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "created_at": f.stat().st_ctime,
                "download_url": f"/v1/reports/{f.name}",
            })
    return ORJSONResponse(content={"reports": files, "total": len(files)})


@router.get("/reports/{filename}")
async def download_report(filename: str):
    """Download a specific report file."""
    # Security: prevent directory traversal
    safe_name = Path(filename).name
    filepath = _reports_dir() / safe_name

    if not filepath.exists():
        return ORJSONResponse(
            content={"error": "report_not_found", "filename": safe_name},
            status_code=404,
        )

    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if filepath.suffix == ".xlsx"
        else "application/octet-stream"
    )
    return FileResponse(
        path=str(filepath),
        filename=safe_name,
        media_type=media_type,
    )


@router.delete("/reports/{filename}")
async def delete_report(filename: str) -> ORJSONResponse:
    """Delete a specific report file."""
    safe_name = Path(filename).name
    filepath = _reports_dir() / safe_name

    if not filepath.exists():
        return ORJSONResponse(
            content={"error": "report_not_found", "filename": safe_name},
            status_code=404,
        )

    filepath.unlink()
    return ORJSONResponse(content={"status": "deleted", "filename": safe_name})
