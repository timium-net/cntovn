#!/usr/bin/env python3
"""Translate Chinese SRT subtitles to Vietnamese using neural machine translation.

This script keeps the original subtitle timing and structure so the result can be
used directly with Vietnamese text-to-speech tools.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import srt
from transformers import pipeline

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed yet
    torch = None


@dataclass
class LineRef:
    """Reference to a subtitle line within the parsed SRT structure."""

    subtitle_index: int
    line_index: int
    text: str


def iter_batches(items: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    """Yield the input sequence in fixed-size batches."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalise_text(text: str) -> str:
    """Light clean-up for translation output to improve readability."""

    cleaned = text.strip()
    replacements = {
        " ,": ",",
        " .": ".",
        " ?": "?",
        " !": "!",
        " :": ":",
        " ;": ";",
        "' ": "'",
        "  ": " ",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def resolve_device() -> int:
    if torch is None:
        return -1
    try:
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def build_translators(device: int, zh_en_model: str, en_vi_model: str):
    """Instantiate translation pipelines for Chinese→English and English→Vietnamese."""

    zh_en = pipeline("translation", model=zh_en_model, device=device)
    en_vi = pipeline("translation", model=en_vi_model, device=device)
    return zh_en, en_vi


def translate_texts(
    texts: Sequence[str],
    zh_en_translator,
    en_vi_translator,
    batch_size: int,
) -> List[str]:
    if not texts:
        return []

    english: List[str] = []
    for batch in iter_batches(texts, batch_size):
        outputs = zh_en_translator(list(batch))
        english.extend(result["translation_text"] for result in outputs)

    vietnamese: List[str] = []
    for batch in iter_batches(english, batch_size):
        outputs = en_vi_translator(list(batch))
        vietnamese.extend(result["translation_text"] for result in outputs)

    return [normalise_text(text) for text in vietnamese]


def gather_lines(subtitles: Sequence[srt.Subtitle]) -> Tuple[List[LineRef], List[List[str]]]:
    """Collect subtitle lines while keeping their positions."""

    references: List[LineRef] = []
    storage: List[List[str]] = []

    for sub_index, subtitle in enumerate(subtitles):
        lines = subtitle.content.splitlines()
        storage.append(lines[:])
        for line_index, line in enumerate(lines):
            text = line.strip()
            if text:
                references.append(LineRef(sub_index, line_index, text))

    return references, storage


def apply_translations(
    subtitles: List[srt.Subtitle],
    storage: List[List[str]],
    translations: Dict[Tuple[int, int], str],
) -> None:
    for sub_index, subtitle in enumerate(subtitles):
        lines = storage[sub_index]
        for line_index in range(len(lines)):
            key = (sub_index, line_index)
            if key in translations:
                lines[line_index] = translations[key]
        subtitle.content = "\n".join(lines)


def translate_srt(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    zh_en_model: str,
    en_vi_model: str,
    batch_size: int,
) -> None:
    raw = input_path.read_text(encoding="utf-8")
    subtitles = list(srt.parse(raw))

    references, storage = gather_lines(subtitles)

    texts = [ref.text for ref in references]

    device = resolve_device()
    zh_en_translator, en_vi_translator = build_translators(device, zh_en_model, en_vi_model)

    translated_texts = translate_texts(texts, zh_en_translator, en_vi_translator, batch_size)

    translations = {
        (ref.subtitle_index, ref.line_index): translated
        for ref, translated in zip(references, translated_texts)
    }

    apply_translations(subtitles, storage, translations)

    output_path.write_text(srt.compose(subtitles), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="Chinese subtitle file in SRT format")
    parser.add_argument(
        "output", type=pathlib.Path, help="Where to write the translated Vietnamese SRT"
    )
    parser.add_argument(
        "--zh-en-model",
        default="Helsinki-NLP/opus-mt-zh-en",
        help="Hugging Face model to translate from Chinese to English",
    )
    parser.add_argument(
        "--en-vi-model",
        default="Helsinki-NLP/opus-mt-en-vi",
        help="Hugging Face model to translate from English to Vietnamese",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of lines to translate per batch (adjust for memory and speed)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    translate_srt(args.input, args.output, args.zh_en_model, args.en_vi_model, args.batch_size)


if __name__ == "__main__":
    main()
