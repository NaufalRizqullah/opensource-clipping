"""NVIDIA NIM transport; its request, SDK retry defaults and parsing stay unchanged."""

import json
import re


class NvidiaNimProvider:
    name = "nvidia"

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg)

    def generate_clips(self, prompt: str, schema: dict) -> list[dict]:
        cfg = self.cfg
        from openai import OpenAI

        print(f"[3/3] Menganalisis Top {cfg.jumlah_clip} momen menggunakan NVIDIA ({cfg.nvidia_model})...")

        if not cfg.api_key_nvidia:
            raise ValueError("NVIDIA_API_KEY tidak ditemukan di environment.")

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=cfg.api_key_nvidia
        )


        # Define the strict schema for Guided JSON (NVIDIA NIM specific)
        clips_schema = schema

        completion = client.chat.completions.create(
            model=cfg.nvidia_model,
            messages=[
                {"role": "system", "content": "You are a professional video editor and strategist. Return JSON only. Follow the provided JSON schema exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=16384,
            extra_body={
                "chat_template_kwargs": {"thinking": False},
                "nvext": {
                    "guided_json": clips_schema
                }
            }
        )

        content = completion.choices[0].message.content

        if "```" in content:
            content = re.sub(r"```(json)?", "", content).strip()
            content = content.split("```")[0].strip()

        hasil = json.loads(content)

        # Guided JSON should return an array directly if schema says type: array
        # but we keep the unwrapper just in case of non-conforming fallbacks
        if isinstance(hasil, dict):
            for key in ["clips", "data", "highlights"]:
                if key in hasil and isinstance(hasil[key], list):
                    hasil = hasil[key]
                    break

        if not isinstance(hasil, list):
            if isinstance(hasil, dict):
                return [hasil]
            raise ValueError(f"Provider NVIDIA mengembalikan format non-list/dict: {type(hasil)}")

        return hasil
