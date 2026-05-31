import os
try:
    import google.genai as genai
    from google.genai import types
except Exception:  # pragma: no cover - dependency layout can vary by environment
    genai = None
    types = None
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RESPONSE_MODES = {"preserve_cs", "normalized"}


class ApiQuotaExceededError(RuntimeError):
    pass

def _split_api_keys(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []

    normalized = raw_value.replace("\n", ",").replace(";", ",")
    keys = [item.strip() for item in normalized.split(",")]
    return [item for item in keys if item]


API_KEYS = _split_api_keys(os.getenv("GEMINI_API_KEYS"))
if not API_KEYS:
    single_key = os.getenv("GEMINI_API_KEY")
    if single_key:
        API_KEYS = [single_key]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Prompt sistem yang digunakan untuk membimbing gaya respons LLM
system_instruction = """
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.

Your answers must:
- Be written in polite and easily understandable Indonesian.
- Be short and to the point (maximum 2–3 sentences).
- Avoid repeating the user's question; respond directly with the answer.
- Obey the response mode explicitly provided in the prompt.
- If the mode is preserve_cs, preserve natural code-switching when it appears in the user's input.
- If the mode is normalized, answer in normalized Indonesian and minimize mixed-language output.
    - If the mode is normalized, answer in normalized Indonesian and minimize mixed-language output.

Example tone:
User: Cuaca hari ini gimana?
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat.

User: Kamu tahu siapa presiden Indonesia?
Assistant: Presiden Indonesia saat ini adalah Joko Widodo.

If you're unsure about an answer, be honest and say that you don't know.
"""

history_adapter = TypeAdapter(list[types.Content]) if types is not None else None


def _create_gemini_client(api_key: str):
    if genai is None or types is None:
        return None
    return genai.Client(api_key=api_key)


def _load_chat_history_data():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return None

    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return None

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()

    if not json_str or history_adapter is None:
        return None

    try:
        return history_adapter.validate_json(json_str)
    except Exception as e:
        print(f"[ERROR] Gagal load history chat: {e}")
        return None


def _create_chat(client_obj):
    if client_obj is None or chat_config is None:
        return None

    history = _load_chat_history_data()
    if history is None:
        return client_obj.chats.create(model=MODEL, config=chat_config)
    return client_obj.chats.create(model=MODEL, config=chat_config, history=history)


def _is_api_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    keywords = (
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "too many requests",
        "429",
    )
    return any(keyword in error_text for keyword in keywords)


class LocalChat:
    def __init__(self):
        self._history: list[dict[str, str]] = []

    def get_history(self):
        return self._history

    def send_message(self, prompt: str):
        # Lightweight fallback that parses our generated prompt format and
        # returns a concise assistant reply without echoing system instructions.
        raw = prompt.strip()
        if not raw:
            text = "Silakan ulangi pertanyaannya."

        else:
            # detect mode if present in the prompt (case-insensitive)
            mode = "preserve_cs"
            low = raw.lower()
            if "mode=normalized" in low:
                mode = "normalized"
            elif "mode=preserve_cs" in low:
                mode = "preserve_cs"

            # extract user transcript after our marker if present
            user_text = raw
            if "transkrip pengguna:" in low:
                try:
                    user_text = raw.split("Transkrip pengguna:", 1)[1].strip()
                except Exception:
                    user_text = raw

            # strip surrounding tags like [ID]...[/ID]
            user_text = user_text.replace("[ID]", "").replace("[/ID]", "").replace("[EN]", "").replace("[/EN]", "").strip()

            # Construct concise fallback reply according to mode
            if mode == "normalized":
                text = f"Jawaban (dinormalisasi): Saya memahami: {user_text}. Berikut saran singkat: ..."
            else:
                # preserve_cs
                text = f"Jawaban (preserve CS): Saya menerima: {user_text}. Jawaban singkat: ..."

        # append to history
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": text})

        class _Response:
            def __init__(self, text: str):
                self.text = text

        return _Response(text)


clients = []
if API_KEYS and genai is not None and types is not None:
    for api_key in API_KEYS:
        client_obj = _create_gemini_client(api_key)
        if client_obj is not None:
            clients.append(client_obj)

if genai is not None and types is not None:
    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.4,
        max_output_tokens=256,
    )
else:
    chat_config = None

# Fungsi untuk menyimpan/memuat riwayat chat
def export_chat_history(chat) -> str:
    if history_adapter is None:
        return "[]"
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")

def save_chat_history(chat):
    json_history = export_chat_history(chat)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(json_history)

def load_chat_history():
    if not clients or chat_config is None:
        return LocalChat()

    return _create_chat(clients[0]) or LocalChat()

# Inisialisasi sesi chat saat aplikasi dimulai
chat = load_chat_history()

# Kirim prompt ke LLM dan kembalikan respons teks
def build_mode_prompt(prompt: str, response_mode: str = "preserve_cs") -> str:
    mode = response_mode if response_mode in RESPONSE_MODES else "preserve_cs"
    if mode == "normalized":
        instruction = (
            "Mode respons: normalized. "
            "Jawab dalam Bahasa Indonesia yang sudah dinormalisasi, minim campuran bahasa, "
            "dan pertahankan makna inti."
        )
    else:
        instruction = (
            "Mode respons: preserve_cs. "
            "Pertahankan code-switching yang natural, jangan memaksa semua kata menjadi Bahasa Indonesia."
        )

    return f"{instruction}\n\nTranskrip pengguna:\n{prompt}"


def generate_response(prompt: str, response_mode: str = "preserve_cs") -> str:
    compiled_prompt = build_mode_prompt(prompt, response_mode=response_mode)

    if not clients or chat_config is None:
        fallback = LocalChat().send_message(compiled_prompt).text.strip()
        return (
            "Layanan Gemini belum aktif, jadi sistem memakai fallback lokal sementara. "
            + fallback
        )

    last_error = None
    quota_error = None
    for index, client_obj in enumerate(clients, start=1):
        chat_session = _create_chat(client_obj)
        if chat_session is None:
            continue

        try:
            response = chat_session.send_message(compiled_prompt)
            save_chat_history(chat_session)
            return response.text.strip()
        except Exception as e:
            if _is_api_limit_error(e):
                quota_error = e
                last_error = e
                print(f"[WARN] Gemini key #{index} kena quota/rate limit, mencoba key berikutnya: {e}")
                continue
            last_error = e
            print(f"[WARN] Gemini key #{index} gagal, mencoba key berikutnya: {e}")

    try:
        if quota_error is not None:
            raise ApiQuotaExceededError(str(quota_error)) from quota_error
        fallback = LocalChat().send_message(compiled_prompt).text.strip()
        suffix = f" Detail error terakhir: {last_error}" if last_error else ""
        return (
            "Layanan Gemini sedang sibuk, kuota habis, atau semua key gagal. "
            "Sistem memakai fallback lokal sementara. "
            + fallback
            + suffix
        )
    except Exception:
        return "Layanan Gemini sedang sibuk atau kuota habis. Sistem memakai fallback lokal sementara."
