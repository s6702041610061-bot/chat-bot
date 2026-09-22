"""Run explicitly when FAQ changes; this command consumes Gemini embedding quota."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from rag import (EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, embedding_metadata,
                 canonical_faq_bytes, parse_faq, validate_vectors)

APP_DIR = Path(__file__).resolve().parent
EMBEDDING_ITEMS_PER_WINDOW = 80
EMBEDDING_WINDOW_DELAY_SECONDS = 80
EMBEDDING_MAX_RETRIES = 3


def rate_limit_retry_delay(error):
    """Return a safe delay for a per-minute embedding quota error."""
    is_rate_limit = (
        type(error).__name__ == "ResourceExhausted"
        or getattr(error, "code", None) == 429
        or getattr(error, "status", None) == "RESOURCE_EXHAUSTED"
    )
    if not is_rate_limit:
        return None

    message = str(error)
    if "PerMinute" not in message and "retry" not in message.lower():
        return None

    retry_match = re.search(r"Please retry in\s+([0-9.]+)s", message, re.IGNORECASE)
    requested_delay = math.ceil(float(retry_match.group(1))) + 1 if retry_match else 0
    return max(EMBEDDING_WINDOW_DELAY_SECONDS, requested_delay)


def build_embeddings(directory, embed_content, sleep_fn=time.sleep, progress=print,
                     pause_at_half=False):
    directory = Path(directory)
    faq_bytes = canonical_faq_bytes((directory / "FAQ_Chatbot_100.md").read_bytes())
    chunks = parse_faq(faq_bytes.decode("utf-8"))
    batches = []
    start = 0
    items_in_window = 0
    halfway = math.ceil(len(chunks) / 2) if pause_at_half else None

    while start < len(chunks):
        if halfway is not None and start == halfway:
            progress(
                f"Embedded first half ({start}/{len(chunks)} FAQs); waiting "
                f"{EMBEDDING_WINDOW_DELAY_SECONDS} seconds before second half..."
            )
            sleep_fn(EMBEDDING_WINDOW_DELAY_SECONDS)
            items_in_window = 0
            halfway = None
        if items_in_window >= EMBEDDING_ITEMS_PER_WINDOW:
            progress(
                f"Embedded {start}/{len(chunks)} FAQs; waiting "
                f"{EMBEDDING_WINDOW_DELAY_SECONDS} seconds for the quota window..."
            )
            sleep_fn(EMBEDDING_WINDOW_DELAY_SECONDS)
            items_in_window = 0

        remaining_in_window = EMBEDDING_ITEMS_PER_WINDOW - items_in_window
        batch_size = min(EMBEDDING_BATCH_SIZE, remaining_in_window,
                         len(chunks) - start,
                         halfway - start if halfway is not None else len(chunks) - start)
        batch = chunks[start:start + batch_size]

        retries = 0
        while True:
            try:
                result = embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document",
                    request_options={"timeout": 60, "retry": None},
                )
                break
            except Exception as error:
                delay = rate_limit_retry_delay(error)
                if delay is None or retries >= EMBEDDING_MAX_RETRIES:
                    raise
                retries += 1
                progress(
                    f"Embedding quota reached; waiting {delay} seconds before "
                    f"retry {retries}/{EMBEDDING_MAX_RETRIES}..."
                )
                sleep_fn(delay)
                # Retrying happens in a new per-minute quota window.
                items_in_window = 0

        vectors = np.asarray(result["embedding"], dtype=np.float32)
        if vectors.ndim == 1 and len(batch) == 1:
            vectors = vectors.reshape(1, -1)
        batches.append(validate_vectors(vectors, len(batch)))
        start += len(batch)
        items_in_window += len(batch)
        progress(f"Embedded {start}/{len(chunks)} FAQs")
    vectors = validate_vectors(np.vstack(batches), len(chunks))
    if canonical_faq_bytes((directory / "FAQ_Chatbot_100.md").read_bytes()) != faq_bytes:
        raise ValueError("FAQ changed during generation; run again")
    metadata = embedding_metadata(faq_bytes, len(chunks))
    # Stage both files; metadata last. Interrupted writes fail checksum validation.
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        binary = Path(temporary) / "faq_embeddings.npz"
        meta = Path(temporary) / "faq_embeddings.meta.json"
        np.savez_compressed(binary, embeddings=vectors)
        metadata["embeddings_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(binary, directory / binary.name)
        os.replace(meta, directory / meta.name)
    return len(chunks)


def main():
    from dotenv import load_dotenv
    from gemini_service import create_client, embedding_function

    load_dotenv(APP_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY_INSURVERSE")
    if not api_key:
        print("Missing GEMINI_API_KEY_INSURVERSE in .env/environment")
        return 1
    client = create_client(api_key)
    try:
        count = build_embeddings(APP_DIR, embedding_function(client), pause_at_half=True)
    except Exception as error:
        print(f"Embedding build failed: {error}")
        return 1
    finally:
        if client is not None:
            client.close()
    print(f"Saved faq_embeddings.npz and faq_embeddings.meta.json: {count} FAQs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
