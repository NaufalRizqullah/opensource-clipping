"""Strict local-output parsing; legacy cloud parsing remains unchanged."""

import json
import re

from .base import InvalidOutputError


def clean_text(content):
    if not isinstance(content, str) or not content.strip():
        raise InvalidOutputError("vLLM mengembalikan content kosong.")
    text = content.strip()
    if text.startswith("<think>"):
        match = re.match(r"^<think>.*?</think>\s*", text, re.DOTALL)
        if not match:
            raise InvalidOutputError("Output thinking vLLM terpotong.")
        text = text[match.end():].strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if not match:
            raise InvalidOutputError("Fence JSON vLLM tidak lengkap.")
        text = match.group(1).strip()
    if not text:
        raise InvalidOutputError("vLLM tidak menghasilkan jawaban final.")
    return text


def parse_clips(content, schema):
    from jsonschema import Draft202012Validator

    def invalid_constant(_):
        raise ValueError

    try:
        value = json.loads(clean_text(content), parse_constant=invalid_constant)
    except (ValueError, TypeError):
        raise InvalidOutputError("vLLM harus menghasilkan satu objek JSON lengkap dengan field clips.") from None
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error is not None:
        # Do not quote validation messages: they can contain full user content.
        raise InvalidOutputError("Output vLLM tidak memenuhi schema klip yang diminta.")
    return value["clips"]
