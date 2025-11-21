import io
import json
import time

from flask import Flask, request
from faster_whisper import WhisperModel
import soundfile as sf
import ollama

# ----------------------------------------
# CONFIG
# ----------------------------------------
OLLAMA_MODEL = "phi4:latest"
WHISPER_MODEL_NAME = "tiny.en"   # fast, english-only

app = Flask(__name__)

print("\n==============================")
print("🚀 Optimized Local AI Server")
print("==============================\n")


# ----------------------------------------
# Helper: clean LLM output into pure JSON
# ----------------------------------------
def extract_json_block(raw: str) -> str:
    """
    Try to strip ```json ... ``` or any text before/after the JSON object,
    so that json.loads() can parse it.
    """
    if not raw:
        return ""

    raw = raw.strip()

    # Strip leading ``` / ```json lines
    if raw.startswith("```"):
        # drop first line (``` or ```json)
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]

        # strip trailing ``` if present
        if raw.endswith("```"):
            raw = raw[:-3]

        raw = raw.strip()

    # Keep only from first '{' to last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    return raw.strip()


# ----------------------------------------
# Load Whisper (STT)
# ----------------------------------------
print(f"Loading Whisper ({WHISPER_MODEL_NAME})...")
t0 = time.time()
whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
print(f"Whisper loaded in {time.time() - t0:.2f} seconds.\n")

# ----------------------------------------
# Preload Ollama model (phi4:latest)
# ----------------------------------------
print(f"Preloading Ollama model '{OLLAMA_MODEL}'...")
t1 = time.time()
try:
    _ = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )
    print(f"Ollama model ready in {time.time() - t1:.2f} seconds.\n")
except Exception as e:
    print("❌ Error preloading Ollama model:", e)
    print("Make sure the model exists and is downloaded with:")
    print(f"  ollama pull {OLLAMA_MODEL}")
    # /chat will still try, but will likely fail until fixed.


# ----------------------------------------
# /stt  → Speech to Text
# ----------------------------------------
@app.route("/stt", methods=["POST"])
def stt():
    try:
        audio_bytes = request.data
        if not audio_bytes:
            print("❌ /stt: No audio data received.")
            return {"text": ""}

        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # stereo → mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        segments, _ = whisper_model.transcribe(audio, beam_size=1)
        text = " ".join([s.text.strip() for s in segments])

        print("🎤 STT:", text)
        return {"text": text}

    except Exception as e:
        print("❌ /stt Error:", e)
        return {"text": ""}


# ----------------------------------------
# /chat → Text to JSON reply from phi4
# ----------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        messages = data.get("messages", [])

        if messages:
            print("\n=== Chat Request ===")
            for m in messages[-3:]:
                print(f"{m['role']}: {m['content'][:120]}")

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=False,
        )

        raw = response["message"]["content"].strip()
        print("\n=== Raw LLM Output ===")
        print(raw)

        # Try to parse cleaned JSON
        try:
            cleaned = extract_json_block(raw)
            parsed = json.loads(cleaned)
        except Exception as e:
            print("❌ ERROR: LLM did not return valid JSON:", e)
            # Fallback: wrap raw text into expected structure
            parsed = {
                "reply": raw,
                "order": [],
                "check_sandwich": False,
            }

        # Ensure keys exist
        if "reply" not in parsed:
            parsed["reply"] = ""
        if "order" not in parsed or parsed["order"] is None:
            parsed["order"] = []
        if "check_sandwich" not in parsed:
            parsed["check_sandwich"] = False

        print("\n=== Parsed AI Output ===")
        print("Reply:", parsed["reply"])
        print("Order:", parsed["order"])
        print("Check Sandwich:", parsed["check_sandwich"])
        print()

        return parsed

    except Exception as e:
        print("❌ /chat Error:", e)
        return {
            "reply": "Something went wrong on the server.",
            "order": [],
            "check_sandwich": False,
        }


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == "__main__":
    print("🎉 AI Server ready on http://localhost:5000")
    print("Endpoints:")
    print("  POST /stt  (audio bytes) → { text: \"...\" }")
    print("  POST /chat (JSON messages) → { reply, order, check_sandwich }\n")

    app.run(host="0.0.0.0", port=5000)
