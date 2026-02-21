import io
import logging
import os
import uuid
import wave
from datetime import datetime
from typing import Any, Dict, Optional

import chainlit as cl
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from documents import extract_documents_text
from logger import get_ai_logger
from prompts import system_prompt
from rag_client import chat as rag_chat
from rag_client import ingest_text as rag_ingest_text

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ai_logger = get_ai_logger()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


@cl.on_shared_thread_view
async def on_shared_thread_view(thread: Dict[str, Any], current_user: cl.User) -> bool:
    return True


@cl.set_starters
async def set_starters():
    return []


@cl.step(type="tool", show_input=False)
async def audio_to_text(audio_buffer):
    request_contents = [
        types.Part.from_bytes(data=audio_buffer, mime_type="audio/wav"),
        "Transcribe this audio exactly as spoken. Return ONLY the text.",
    ]
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=request_contents,
    )
    await _log_ai_call(
        purpose="audio_to_text",
        model="gemini-2.5-flash",
        request=request_contents,
        response=response.text,
    )
    return response.text


async def process_audio():
    audio_chunks = cl.user_session.get("audio_chunks")
    if not audio_chunks:
        return

    concatenated = np.concatenate(audio_chunks)
    sample_rate = 24000
    duration = concatenated.shape[0] / float(sample_rate)

    if duration <= 0.5:
        cl.user_session.set("audio_chunks", [])
        return

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(concatenated.tobytes())

    wav_buffer.seek(0)
    cl.user_session.set("audio_chunks", [])
    audio_buffer = wav_buffer.getvalue()

    transcription = await audio_to_text(audio_buffer)

    selected_command = cl.user_session.get("selected_command")
    user_message = cl.Message(
        content=transcription,
        author="User",
        type="user_message",
        command=selected_command,
    )
    await user_message.send()
    await on_message(user_message)


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)


@cl.on_audio_end
async def on_audio_end():
    await process_audio()
    return True


@cl.on_chat_start
async def start():
    separator = "======================="
    logger.info(separator)
    ai_logger.info(separator)
    cl.user_session.set("rag_scope_id", str(uuid.uuid4()))

    await cl.context.emitter.set_commands(
        [
            {
                "id": "search",
                "name": "Search",
                "description": "Search the web for relevant information",
                "icon": "scan-search",
                "button": True,
                "persistent": True,
            }
        ]
    )


@cl.on_message
async def on_message(msg: cl.Message):
    if not cl.user_session.get("is_thread_renamed", False):
        try:
            thread_name_request = (
                f"Summarize this query in MAX 8 words for a chat thread name: `{msg.content}`"
            )
            thread_name_response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=thread_name_request,
            )
            thread_name = thread_name_response.text.strip()
            await _log_ai_call(
                purpose="thread_rename",
                model="gemini-2.5-flash",
                request=thread_name_request,
                response=thread_name,
            )
            await cl.context.emitter.init_thread(thread_name)
            cl.user_session.set("is_thread_renamed", True)
        except Exception as exc:
            logger.error("Failed to rename thread: %s", exc)

    scope_id = cl.user_session.get("rag_scope_id")
    if not scope_id:
        scope_id = str(uuid.uuid4())
        cl.user_session.set("rag_scope_id", scope_id)

    uploads = msg.elements or []
    if getattr(msg, "files", None):
        uploads.extend(msg.files or [])

    docs = [
        f
        for f in uploads
        if str(f.name).lower().endswith((".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"))
    ]
    imgs = [
        f
        for f in uploads
        if str(f.name).lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    uploaded_docs_text: Optional[str] = None
    if docs:
        docs_text = await extract_documents_text(docs)
        if docs_text.strip():
            uploaded_docs_text = docs_text
            try:
                await rag_ingest_text(
                    docs_text,
                    source="upload",
                    title="user_uploads",
                    metadata={"filenames": [str(f.name) for f in docs]},
                    scope_id=scope_id,
                )
            except Exception as exc:
                logger.warning("RAG ingest failed for uploads: %s", exc)

    if imgs:
        logger.info(
            "Image uploads detected (%s files). Backend /chat currently handles text/doc context only.",
            len(imgs),
        )

    use_web_search = msg.command == "search"
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    temperature = float(os.getenv("RAG_TEMPERATURE", "0.7"))

    answer_text = ""
    try:
        chat_result = await rag_chat(
            msg.content,
            top_k=top_k,
            scope_id=scope_id,
            system_prompt=system_prompt + f"\nCurrent Time: {datetime.now()}",
            uploaded_text=uploaded_docs_text,
            use_web_search=use_web_search,
            temperature=temperature,
        )
        answer_text = (chat_result.get("answer") or "").strip()
    except Exception as exc:
        logger.warning("RAG chat failed: %s", exc)
        answer_text = "I could not reach the backend chat service. Please try again."

    if not answer_text:
        answer_text = "I could not generate a response."

    final_answer = cl.Message(content=answer_text)
    await final_answer.send()

    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": msg.content})
    chat_history.append({"role": "assistant", "content": final_answer.content})
    cl.user_session.set("chat_history", chat_history)


async def _log_ai_call(*, purpose: str, model: str, request: Any, response: Any) -> None:
    try:
        prompt_tokens = None
        try:
            count = await client.aio.models.count_tokens(model=model, contents=request)
            prompt_tokens = getattr(count, "total_tokens", None)
        except Exception:
            prompt_tokens = None

        response_tokens = None
        try:
            count = await client.aio.models.count_tokens(
                model=model, contents=str(response)
            )
            response_tokens = getattr(count, "total_tokens", None)
        except Exception:
            response_tokens = None

        total_tokens = None
        if prompt_tokens is not None or response_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (response_tokens or 0)

        logger.info(
            "AI_CALL purpose=%s model=%s prompt_tokens=%s response_tokens=%s total_tokens=%s request=%s response=%s",
            purpose,
            model,
            prompt_tokens,
            response_tokens,
            total_tokens,
            _preview(request),
            _preview(response),
        )
        ai_logger.info(
            "AI_CALL purpose=%s model=%s prompt_tokens=%s response_tokens=%s total_tokens=%s request=%s response=%s",
            purpose,
            model,
            prompt_tokens,
            response_tokens,
            total_tokens,
            _preview(request),
            _preview(response),
        )
    except Exception as exc:
        logger.warning("AI_CALL logging failed: %s", exc)


def _preview(value: Any, limit: int = 1200) -> str:
    if value is None:
        return "None"
    if isinstance(value, list):
        parts = []
        for item in value[:5]:
            parts.append(_preview(item, limit=300))
        extra = "" if len(value) <= 5 else f"...(+{len(value)-5} more)"
        return "[" + ", ".join(parts) + f"]{extra}"
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text
