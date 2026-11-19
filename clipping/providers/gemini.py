"""Gemini transport with the established Developer API routing."""


class GeminiProvider:
    name = "gemini"

    def __init__(self, cfg, *, generate_json, timeout_ms):
        self.cfg = cfg
        self.generate_json = generate_json
        self.timeout_ms = timeout_ms

    def generate_clips(self, prompt: str, schema: dict) -> list[dict]:
        import google.genai as genai
        from google.genai import types

        cfg = self.cfg
        print(f"[3/3] Menganalisis Top {cfg.jumlah_clip} momen terbaik menggunakan Gemini...")
        client = genai.Client(
            api_key=cfg.api_key_gemini,
            vertexai=False,
            http_options=types.HttpOptions(
                api_version="v1beta",
                timeout=self.timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )
        return self.generate_json(
            client=client,
            model=cfg.gemini_model,
            fallback_model=getattr(cfg, "gemini_fallback_model", None),
            contents=prompt,
            config=config,
        )
