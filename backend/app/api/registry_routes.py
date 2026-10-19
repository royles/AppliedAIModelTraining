import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import TrainingJob
from app.schemas import (
    RegistryLifecycleInfo,
    RegistryModelRead,
    RegistryModelUpdate,
    RegistryTransitionRequest,
)
from app.services.model_registry import (
    get_registry_model,
    lifecycle_catalog,
    list_models,
    refresh_model_card,
    transition_model_state,
    update_registry_entry,
)

router = APIRouter(prefix="/registry", tags=["registry"])


def _to_read(db: Session, record) -> RegistryModelRead:
    job = db.get(TrainingJob, record.job_id)
    meta = {}
    if record.metadata_json:
        try:
            meta = json.loads(record.metadata_json)
        except json.JSONDecodeError:
            meta = {}
    bench = None
    if record.benchmark_json:
        try:
            bench = json.loads(record.benchmark_json)
        except json.JSONDecodeError:
            bench = None
    return RegistryModelRead(
        id=record.id,
        job_id=record.job_id,
        job_name=job.name if job else None,
        openai_model_id=record.openai_model_id,
        display_name=record.display_name,
        description=record.description,
        lifecycle_state=record.lifecycle_state,
        task_type=record.task_type,
        base_model=record.base_model,
        artifact_path=record.artifact_path,
        created_at_unix=record.created_at_unix,
        updated_at=record.updated_at,
        parent_model_id=record.parent_model_id,
        metadata=meta,
        benchmark=bench,
        model_card_markdown=record.model_card_markdown,
    )


@router.get("/lifecycle", response_model=list[RegistryLifecycleInfo])
def registry_lifecycle() -> list[RegistryLifecycleInfo]:
    return [RegistryLifecycleInfo(**row) for row in lifecycle_catalog()]


@router.get("/models", response_model=list[RegistryModelRead])
def registry_list_models(db: Session = Depends(get_db)) -> list[RegistryModelRead]:
    return [_to_read(db, m) for m in list_models(db)]


@router.get("/models/{registry_id}", response_model=RegistryModelRead)
def registry_get_model(registry_id: int, db: Session = Depends(get_db)) -> RegistryModelRead:
    record = get_registry_model(db, registry_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found in registry")
    return _to_read(db, record)


@router.patch("/models/{registry_id}", response_model=RegistryModelRead)
def registry_update_model(
    registry_id: int,
    payload: RegistryModelUpdate,
    db: Session = Depends(get_db),
) -> RegistryModelRead:
    try:
        record = update_registry_entry(
            db,
            registry_id,
            display_name=payload.display_name,
            description=payload.description,
            benchmark=payload.benchmark,
            model_card_markdown=payload.model_card_markdown,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_read(db, record)


@router.post("/models/{registry_id}/transition", response_model=RegistryModelRead)
def registry_transition(
    registry_id: int,
    payload: RegistryTransitionRequest,
    db: Session = Depends(get_db),
) -> RegistryModelRead:
    try:
        record = transition_model_state(db, registry_id, payload.lifecycle_state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_read(db, record)


@router.post("/models/{registry_id}/refresh-card", response_model=RegistryModelRead)
def registry_refresh_card(registry_id: int, db: Session = Depends(get_db)) -> RegistryModelRead:
    try:
        record = refresh_model_card(db, registry_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_read(db, record)
