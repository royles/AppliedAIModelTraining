import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.inference.engine import chat_complete
from app.inference.schemas_openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelListResponse,
)
from app.services.model_lifecycle import SERVING_STATES, STATE_RETIRED
from app.services.model_registry import get_model_by_openai_id, list_models

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
def list_openai_models(db: Session = Depends(get_db)) -> ModelListResponse:
    records = list_models(db)
    return ModelListResponse(
        data=[
            ModelCard(
                id=m.openai_model_id,
                created=m.created_at_unix,
                owned_by="platform",
            )
            for m in records
            if m.task_type == "causal_lm"
            and m.lifecycle_state in SERVING_STATES
            and m.lifecycle_state != STATE_RETIRED
        ]
    )


@router.get("/models/{model_id}", response_model=ModelCard)
def get_openai_model(model_id: str, db: Session = Depends(get_db)) -> ModelCard:
    record = get_model_by_openai_id(db, model_id)
    if not record or record.task_type != "causal_lm":
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelCard(id=record.openai_model_id, created=record.created_at_unix)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(
    payload: ChatCompletionRequest, db: Session = Depends(get_db)
) -> ChatCompletionResponse:
    if payload.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported yet")

    record = get_model_by_openai_id(db, payload.model)
    if not record:
        raise HTTPException(status_code=404, detail=f"Model '{payload.model}' not found")
    if record.task_type != "causal_lm":
        raise HTTPException(
            status_code=400,
            detail="Only causal_lm models support chat completions",
        )
    if record.lifecycle_state == STATE_RETIRED:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{payload.model}' is retired",
        )
    if record.lifecycle_state not in SERVING_STATES:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Model '{payload.model}' is in state '{record.lifecycle_state}'. "
                "Promote it in the Model Registry before serving."
            ),
        )

    messages = [m.model_dump() for m in payload.messages]
    max_tokens = payload.max_tokens or 256
    try:
        text, usage = chat_complete(
            record.artifact_path,
            messages,
            temperature=payload.temperature,
            max_tokens=max_tokens,
            top_p=payload.top_p,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=payload.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
