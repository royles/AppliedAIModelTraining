import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.inference.engine import chat_complete, chat_complete_stream
from app.inference.schemas_openai import (
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatMessage,
    ModelCard,
    ModelListResponse,
)
from app.models import TrainedModel
from app.services.model_lifecycle import SERVING_STATES, STATE_RETIRED
from app.services.model_registry import get_model_by_openai_id, list_models

router = APIRouter()


def _resolve_serving_model(db: Session, model_id: str) -> TrainedModel:
    record = get_model_by_openai_id(db, model_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    if record.task_type != "causal_lm":
        raise HTTPException(
            status_code=400,
            detail="Only causal_lm models support chat completions",
        )
    if record.lifecycle_state == STATE_RETIRED:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' is retired",
        )
    if record.lifecycle_state not in SERVING_STATES:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Model '{model_id}' is in state '{record.lifecycle_state}'. "
                "Promote it in the Model Registry before serving."
            ),
        )
    return record


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


def _sse_chunk(chunk: ChatCompletionStreamChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


@router.post("/chat/completions")
def create_chat_completion(payload: ChatCompletionRequest, db: Session = Depends(get_db)):
    record = _resolve_serving_model(db, payload.model)
    messages = [m.model_dump() for m in payload.messages]
    max_tokens = payload.max_tokens or 256

    if payload.stream:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def event_stream():
            first = True
            try:
                for piece in chat_complete_stream(
                    record.artifact_path,
                    messages,
                    temperature=payload.temperature,
                    max_tokens=max_tokens,
                    top_p=payload.top_p,
                ):
                    delta = ChatCompletionDelta(content=piece)
                    if first:
                        delta = ChatCompletionDelta(role="assistant", content=piece)
                        first = False
                    chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        created=created,
                        model=payload.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta=delta,
                                finish_reason=None,
                            )
                        ],
                    )
                    yield _sse_chunk(chunk)
                final = ChatCompletionStreamChunk(
                    id=completion_id,
                    created=created,
                    model=payload.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=ChatCompletionDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield _sse_chunk(final)
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                err = {"error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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
