import io
import logging
import os
import wave
from datetime import datetime
from typing import Any, Dict, Optional

import chainlit as cl
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import system_prompt
from documents import extract_documents_text
from rag_client import ingest_text as rag_ingest_text
from rag_client import retrieve as rag_retrieve
from logger import get_ai_logger

load_dotenv()
# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ai_logger = get_ai_logger()

# Initialize Gemini Client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# @cl.oauth_callback
# def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User,
# ) -> Optional[cl.User]:
#     return default_user

@cl.on_shared_thread_view
async def on_shared_thread_view(thread: Dict[str, Any], current_user: cl.User) -> bool:
    return True

@cl.set_starters
async def set_starters():
    return []

@cl.step(type="tool", show_input=False)
async def audio_to_text(audio_buffer):
    """Sends audio bytes directly to Gemini 2.5 for transcription/understanding"""
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
    
    # Filter out very short audio glitches
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

    # Call the new Gemini-based audio tool
    transcription = await audio_to_text(audio_buffer)
    
    # Send the transcription as a user message
    selected_command = cl.user_session.get("selected_command")
    user_message = cl.Message(
        content=transcription,
        author="User",
        type="user_message",
        command=selected_command,
    )
    await user_message.send()
    
    # Trigger the chat response logic
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
    await cl.context.emitter.set_commands([
        {
            "id": "search",
            "name": "Search",
            "description": "Search the web for relevant information",
            "icon": "scan-search",
            "button": True,
            "persistent": True,
        }
    ])

@cl.on_message
async def on_message(msg: cl.Message):
    # --- 1. Thread Renaming (using Gemini) ---
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
        except Exception as e:
            logger.error(f"Failed to rename thread: {e}")

    # --- 2. Prepare Tools (Google Search) ---
    tools = []
    # If the user explicitly clicks "Search", OR if we detect it's a general question
    if msg.command == "search":
        logger.info("Enabling Google Search Tool")
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    # --- 3. Process Uploads (Docs & Images) ---
    uploads = msg.elements or []
    # Chainlit attaches files to msg.files in newer versions, check both
    if getattr(msg, "files", None):
        uploads.extend(msg.files or [])

    docs = [f for f in uploads if str(f.name).lower().endswith((".pdf", ".docx", ".doc"))]
    imgs = [f for f in uploads if str(f.name).lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    
    content_parts = []
    
    # Add System Prompt
    content_parts.append(system_prompt + f"\nCurrent Time: {datetime.now()}")

    # Handle Documents (Extract text)
    if docs:
        docs_text = await extract_documents_text(docs)
        content_parts.append(f"User uploaded documents:\n{docs_text}")
        try:
            await rag_ingest_text(
                docs_text,
                source="upload",
                title="user_uploads",
                metadata={"filenames": [str(f.name) for f in docs]},
            )
        except Exception as e:
            logger.warning(f"RAG ingest failed for uploads: {e}")
    
    # Handle Images (Pass raw bytes directly to Gemini)
    for img in imgs:
        # Read image bytes
        path = getattr(img, "path", None)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                img_bytes = f.read()
                # Determine mime type roughly or default to jpeg
                mime_type = "image/png" if path.endswith(".png") else "image/jpeg"
                content_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

    # Add the user's actual text message
    try:
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        results = await rag_retrieve(msg.content, top_k=top_k)
        if results:
            context_blocks = []
            for i, r in enumerate(results, start=1):
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                context_blocks.append(f"[{i}] {text}")
            if context_blocks:
                content_parts.append("Retrieved context:\n" + "\n\n".join(context_blocks))
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")

    content_parts.append(msg.content)

    # --- 4. Generate Response ---
    final_answer = cl.Message(content="")
    
    # Use config for tools and streaming
    config = types.GenerateContentConfig(
        tools=tools,
        temperature=0.7,
        response_modalities=["TEXT"]
    )

    # Stream the response from Gemini
    response_text = ""
    async for chunk in await client.aio.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=content_parts,
        config=config
    ):
        if chunk.text:
            response_text += chunk.text
            await final_answer.stream_token(chunk.text)
            
    await final_answer.send()
    await _log_ai_call(
        purpose="chat_response",
        model="gemini-2.5-flash",
        request=content_parts,
        response=response_text,
    )

    # Update Chat History
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

