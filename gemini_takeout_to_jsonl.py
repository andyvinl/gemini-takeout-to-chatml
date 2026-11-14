#!/usr/bin/env python3
"""
Convert a Google Takeout Gemini chat history export into JSONL for LLM fine-tuning.

Each output line looks like:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Input can be a Takeout .zip, an unzipped Takeout folder, or a single JSON/HTML file.
Inside a zip/folder every .json and .txt file is read (Gemini in Workspace stores
its JSON conversations as .txt), plus the Gemini Apps MyActivity.html; other
products' activity pages and attachments are ignored.

Supported input layouts
-----------------------
1. Google Takeout "My Activity > Gemini Apps" (MyActivity.json) -- the usual export.
   A top-level array of activity records such as:
     {"header": "Gemini Apps",
      "title": "Prompted How do I ...",
      "time": "2025-01-01T12:00:00.000Z",
      "safeHtmlItem": [{"html": "<p>Gemini's answer</p>"}], ...}
   The prompt is the title after "Prompted "; the answer is HTML, converted to
   Markdown-style plain text. Records that are not prompts ("Used ...",
   "Gave feedback", "Created Gem", etc.) are skipped.

   The HTML version of this export (Takeout's default format) is also supported.
   Each entry links to its chat (gemini.google.com/app/<id>), so --multi-turn
   can rebuild whole conversations from it.

2. Takeout "Gemini in Workspace/Conversation History/conversation_*.txt":
     {"conversation_turns": [{"user_turn": {"prompt": "..."}},
                             {"system_turn": {"text": [{"data": "..."}], "model_thoughts": [...]}}]}
   Only the prompt and the visible reply text are kept; model_thoughts
   (hidden reasoning) and citations are dropped.

3. Generic conversation exports: objects that hold a list of turns with
   role/author/sender fields ("user"/"human" vs "model"/"assistant"/"gemini")
   and text in content/text/parts/message. Consecutive user->model pairs
   become examples.

Large files
-----------
If the optional `ijson` package is installed, the file is streamed so memory
stays flat even for multi-GB exports. Otherwise the whole file is loaded with
the standard json module. `tqdm` is used for a progress bar if installed;
otherwise progress is printed periodically.

    pip install ijson tqdm      # optional, recommended for huge files

Usage
-----
    python3 gemini_takeout_to_jsonl.py takeout-XXXX.zip -o gemini_finetune.jsonl
    python3 gemini_takeout_to_jsonl.py MyActivity.json -o gemini_finetune.jsonl
    python3 gemini_takeout_to_jsonl.py takeout-XXXX.zip --multi-turn   # one line per conversation
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from typing import IO, Any, Iterable, Iterator

try:
    import ijson  # type: ignore
except ImportError:  # pragma: no cover
    ijson = None

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


# --------------------------------------------------------------------------- #
# HTML -> Markdown-ish text (stdlib only)
# --------------------------------------------------------------------------- #
class _HTMLToText(HTMLParser):
    """Small HTML-to-Markdown converter that keeps paragraphs, lists, code and headings."""

    BLOCK = {"p", "div", "section", "article", "table", "tr", "blockquote", "br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.list_stack: list[list[Any]] = []  # [kind, counter]
        self.in_pre = 0
        self.href: str | None = None
        self.skip = 0  # inside <script>/<style>

    def _nl(self, n: int = 1) -> None:
        text = "".join(self.out)
        trailing = len(text) - len(text.rstrip("\n"))
        if text and trailing < n:
            self.out.append("\n" * (n - trailing))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._nl(2)
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "pre":
            self._nl(2)
            self.out.append("```\n")
            self.in_pre += 1
        elif tag == "code" and not self.in_pre:
            self.out.append("`")
        elif tag in ("strong", "b"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag in ("ul", "ol"):
            self._nl(1 if self.list_stack else 2)
            self.list_stack.append([tag, 0])
        elif tag == "li":
            self._nl(1)
            depth = max(len(self.list_stack) - 1, 0)
            if self.list_stack and self.list_stack[-1][0] == "ol":
                self.list_stack[-1][1] += 1
                bullet = f"{self.list_stack[-1][1]}. "
            else:
                bullet = "- "
            self.out.append("  " * depth + bullet)
        elif tag == "a":
            self.href = dict(attrs).get("href")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "br":
            self.out.append("\n")
        elif tag in self.BLOCK:
            self._nl(2)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip = max(self.skip - 1, 0)
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._nl(2)
        elif tag == "pre":
            self._nl(1)
            self.out.append("```")
            self._nl(2)
            self.in_pre = max(self.in_pre - 1, 0)
        elif tag == "code" and not self.in_pre:
            self.out.append("`")
        elif tag in ("strong", "b"):
            self.out.append("**")
        elif tag in ("em", "i"):
            self.out.append("*")
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self._nl(1 if self.list_stack else 2)
        elif tag == "a":
            if self.href and self.href.startswith("http"):
                self.out.append(f" ({self.href})")
            self.href = None
        elif tag in self.BLOCK:
            self._nl(2)

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.in_pre:
            self.out.append(data)
        else:
            self.out.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        raw = "".join(self.out)
        lines = [ln.rstrip() for ln in raw.split("\n")]
        raw = "\n".join(lines)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(fragment: str) -> str:
    if "<" not in fragment:
        return html.unescape(fragment).strip()
    parser = _HTMLToText()
    try:
        parser.feed(fragment)
        parser.close()
        return parser.text()
    except Exception:
        # Fallback: crude tag strip.
        return html.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


# --------------------------------------------------------------------------- #
# Cleaning / validation
# --------------------------------------------------------------------------- #
# Text that is pure system metadata rather than real conversation content.
_METADATA_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*$",
        r"^(null|none|undefined|n/?a)$",
        r"^\[?(image|file|attachment|audio|video)( attached| omitted| uploaded)?\]?$",
        r"^gemini apps?$",
        r"^[\W_]*$",  # only punctuation, e.g. a reply whose text was lost and left as ". . ."
        r"^(response|answer|output) (unavailable|not available|was blocked)\.?",
        r"^something went wrong\.?( please try again\.?)?",
    )
]


def clean_text(value: Any) -> str:
    """Normalise to a stripped string; None/non-text -> ''."""
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(clean_text(v) for v in value)
    elif isinstance(value, dict):
        value = extract_text(value)
    if not isinstance(value, str):
        return ""
    value = value.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def is_metadata_only(text: str) -> bool:
    return any(p.fullmatch(text) for p in _METADATA_PATTERNS)


def valid_pair(user: str, assistant: str) -> bool:
    return bool(user) and bool(assistant) and not is_metadata_only(user) and not is_metadata_only(assistant)


# --------------------------------------------------------------------------- #
# Format 1: Google Takeout My Activity (Gemini Apps)
# --------------------------------------------------------------------------- #
_PROMPT_PREFIXES = ("Prompted ", "Asked ", "Said ")


def is_activity_record(obj: Any) -> bool:
    return isinstance(obj, dict) and "title" in obj and ("header" in obj or "time" in obj or "products" in obj)


def pair_from_activity(rec: dict) -> tuple[str, str] | None:
    title = rec.get("title")
    if not isinstance(title, str):
        return None
    title = title.replace(" ", " ")
    prompt = None
    for prefix in _PROMPT_PREFIXES:
        if title.startswith(prefix):
            prompt = title[len(prefix):]
            break
    if prompt is None:
        return None  # "Used an Assistant feature", "Gave feedback", etc.
    user = clean_text(html.unescape(prompt))

    parts: list[str] = []
    for item in rec.get("safeHtmlItem") or []:
        if isinstance(item, dict) and isinstance(item.get("html"), str):
            parts.append(html_to_text(item["html"]))
    if not parts:
        # Some exports put the response in other fields.
        for key in ("response", "responseText", "text"):
            if rec.get(key):
                parts.append(html_to_text(clean_text(rec[key])))
                break
    assistant = clean_text("\n\n".join(p for p in parts if p))
    return user, assistant


# --------------------------------------------------------------------------- #
# Format 1b: the same My Activity export, but saved as HTML (Takeout's default)
# --------------------------------------------------------------------------- #
_HTML_CELL_SPLIT = re.compile(r'<div class="outer-cell\b')
_HTML_HEADER = re.compile(r'class="header-cell[^"]*"><p[^>]*>(.*?)<', re.S)
_HTML_BODY = re.compile(
    r'mdl-typography--body-1">(.*?)</div><div class="content-cell[^"]*mdl-typography--text-right"', re.S)
_HTML_DATE = re.compile(
    r"<br>\s*([A-Z][a-z]{2,8}\.? \d{1,2}, \d{4},? \d{1,2}:\d{2}(?::\d{2})?[\s ]?[AaPp]\.?[Mm]\.?"
    r"(?: [A-Za-z][A-Za-z0-9+:\-]{1,10})?)<br>")
_HTML_CONV = re.compile(r"gemini\.google\.com/app/([0-9a-zA-Z_-]+)")


def _strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment))


def iter_html_activity(text: str, stats: dict) -> Iterator[dict]:
    """Turn a Takeout MyActivity.html into activity records shaped like the JSON export.
    Each cell is: '<title><br><date><br><response html>' plus a caption with the chat link.
    Anything after the first <br> in the title (attachment lists, '1 generated image.') is dropped."""
    for cell in _HTML_CELL_SPLIT.split(text)[1:]:
        header = _HTML_HEADER.search(cell)
        if header and "gemini" not in _strip_tags(header.group(1)).lower():
            continue  # another product's activity
        body = _HTML_BODY.search(cell)
        date = _HTML_DATE.search(body.group(1)) if body else None
        if not body or not date:
            stats["skipped_unknown"] += 1
            continue
        b = body.group(1)
        title = _strip_tags(b[:date.start()].split("<br>", 1)[0])
        conv = _HTML_CONV.search(cell)
        yield {
            "header": "Gemini Apps",
            "title": title,
            "time": date.group(1),
            "safeHtmlItem": [{"html": b[date.end():]}],
            "conversation_id": conv.group(1) if conv else None,
        }


# --------------------------------------------------------------------------- #
# Format 2: generic conversation / turn lists
# --------------------------------------------------------------------------- #
_USER_ROLES = {"user", "human", "you", "me", "prompt", "request"}
_MODEL_ROLES = {"model", "assistant", "gemini", "bard", "ai", "bot", "response"}
_ROLE_KEYS = ("role", "author", "sender", "speaker", "from", "type")
_TEXT_KEYS = ("content", "text", "parts", "message", "body", "value", "html")
_TURN_LIST_KEYS = ("conversation_turns", "messages","turns", "chunks", "conversation", "history", "entries", "items", "contents")


def extract_text(obj: Any) -> str:
    """Pull human-readable text out of a turn (str / list of parts / nested dicts)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return html_to_text(obj) if "<" in obj and ">" in obj else obj.strip()
    if isinstance(obj, list):
        return "\n".join(t for t in (extract_text(x) for x in obj) if t).strip()
    if isinstance(obj, dict):
        for key in _TEXT_KEYS:
            if key in obj:
                t = extract_text(obj[key])
                if t:
                    return t
    return ""


_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~\"'])")


def render_questionnaire(raw: str) -> str:
    """Gemini in Workspace 'questionnaire' replies are Markdown-escaped JSON; render as text."""
    unescaped = _MD_ESCAPE.sub(r"\1", raw)
    try:
        q = json.loads(unescaped)
    except (json.JSONDecodeError, TypeError):
        return unescaped.strip()
    if not isinstance(q, dict):
        return unescaped.strip()
    header = [str(q[k]) if k != "title" else f"**{q[k]}**" for k in ("title", "description") if q.get(k)]
    questions: list[str] = []
    items = q.get("question_with_answers") or q.get("questions") or []
    for i, item in enumerate((it for it in items if isinstance(it, dict)), 1):
        questions.append(f"{i}. {item.get('question', '')}".rstrip())
        for ans in item.get("answers") or []:
            label = ans if isinstance(ans, str) else (ans.get("text") or ans.get("answer") if isinstance(ans, dict) else None)
            if label:
                questions.append(f"   - {label}")
    blocks = header + (["\n".join(questions)] if questions else [])
    return "\n\n".join(blocks).strip()


def workspace_turn(turn: dict) -> tuple[str, str] | None:
    """Gemini in Workspace format: {"user_turn": {"prompt": ...}} / {"system_turn": {"text": [...]}}.
    model_thoughts (hidden reasoning) and citations are deliberately dropped."""
    if isinstance(turn.get("user_turn"), dict):
        return "user", clean_text(turn["user_turn"].get("prompt"))
    for key in ("system_turn", "model_turn"):
        st = turn.get(key)
        if isinstance(st, dict):
            parts: list[str] = []
            for item in st.get("text") or []:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("data"), str):
                        parts.append(item["data"])
                    elif isinstance(item.get("questionnaire"), str):
                        parts.append(render_questionnaire(item["questionnaire"]))
            return "assistant", clean_text("\n\n".join(p.strip() for p in parts if p and p.strip()))
    return None


def role_and_text(turn: Any) -> tuple[str, str] | None:
    """(role, text) for any supported turn shape, or None for system/tool/unknown turns."""
    if not isinstance(turn, dict):
        return None
    ws = workspace_turn(turn)
    if ws is not None:
        return ws
    r = role_of(turn)
    if r is None:
        return None
    return r, clean_text(extract_text(turn))


def role_of(turn: dict) -> str | None:
    for key in _ROLE_KEYS:
        val = turn.get(key)
        if isinstance(val, dict):
            val = val.get("role") or val.get("name")
        if isinstance(val, str):
            v = val.strip().lower()
            if v in _USER_ROLES:
                return "user"
            if v in _MODEL_ROLES:
                return "assistant"
    return None


def find_turn_list(obj: dict) -> list | None:
    for key in _TURN_LIST_KEYS:
        val = obj.get(key)
        if isinstance(val, list) and any(role_and_text(t) for t in val):
            return val
    return None


def examples_from_conversation(turns: list, multi_turn: bool) -> Iterator[list[dict]]:
    """Yield message lists from an ordered list of role-tagged turns."""
    normalized: list[tuple[str, str]] = []
    for t in turns:
        rt = role_and_text(t)
        if rt is None:
            continue  # system / tool / metadata turn
        r, txt = rt
        # Merge consecutive turns from the same speaker.
        if normalized and normalized[-1][0] == r:
            normalized[-1] = (r, (normalized[-1][1] + "\n\n" + txt).strip())
        else:
            normalized.append((r, txt))

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(normalized) - 1:
        if normalized[i][0] == "user" and normalized[i + 1][0] == "assistant":
            pairs.append((normalized[i][1], normalized[i + 1][1]))
            i += 2
        else:
            i += 1

    if multi_turn:
        msgs: list[dict] = []
        for u, a in pairs:
            if valid_pair(u, a):
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        if msgs:
            yield msgs
    else:
        for u, a in pairs:
            yield [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


# --------------------------------------------------------------------------- #
# Reading the input
# --------------------------------------------------------------------------- #
_DATA_EXTS = (".json", ".txt")


def _records_from_stream(fh: IO[bytes]) -> Iterator[Any]:
    """Yield top-level records from one JSON document (binary stream with .peek()).
    Large top-level arrays are streamed with ijson when it is installed."""
    first = fh.peek(4096).lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if first not in (b"[", b"{"):
        raise ValueError("not a JSON document")
    if first == b"[" and ijson is not None:
        yield from ijson.items(fh, "item", use_float=True)
        return
    data = json.load(io.TextIOWrapper(fh, encoding="utf-8-sig"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        if find_turn_list(data) is not None or is_activity_record(data):
            yield data  # a single conversation file
        else:
            for value in data.values():
                if isinstance(value, list):
                    yield from value
                else:
                    yield value


def iter_top_level(path: str, stats: dict) -> Iterator[Any]:
    """Yield records from a JSON file, a Takeout .zip, or a folder (e.g. an unzipped Takeout)."""

    def from_stream(fh: IO[bytes], label: str) -> Iterator[Any]:
        stats["files_read"] += 1
        try:
            yield from _records_from_stream(fh)
        except (ValueError, json.JSONDecodeError) as e:
            stats["files_skipped"] += 1
            print(f"  Skipping {label}: {e}", file=sys.stderr)

    def from_html(fh: IO[bytes]) -> Iterator[Any]:
        stats["files_read"] += 1
        yield from iter_html_activity(fh.read().decode("utf-8", errors="replace"), stats)

    def is_gemini_html(name: str) -> bool:
        # Only Gemini's My Activity page; other products' MyActivity.html files are ignored.
        return name.lower().endswith(".html") and "gemini" in name.lower() and "myactivity" in name.lower()

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(_DATA_EXTS):
                    with zf.open(name) as raw:
                        yield from from_stream(io.BufferedReader(raw), name)  # type: ignore[arg-type]
                elif is_gemini_html(name):
                    with zf.open(name) as raw:
                        yield from from_html(raw)  # type: ignore[arg-type]
    elif os.path.isdir(path):
        for root, _dirs, files in sorted(os.walk(path)):
            for name in sorted(files):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, path)
                if name.lower().endswith(_DATA_EXTS):
                    with open(full, "rb") as fh:
                        yield from from_stream(fh, rel)
                elif is_gemini_html(rel):
                    with open(full, "rb") as fh:
                        yield from from_html(fh)
    elif path.lower().endswith((".html", ".htm")):
        with open(path, "rb") as fh:
            yield from from_html(fh)
    else:
        with open(path, "rb") as fh:
            stats["files_read"] += 1
            yield from _records_from_stream(fh)


def walk_records(rec: Any, multi_turn: bool, stats: dict) -> Iterator[list[dict]]:
    """Turn one top-level record into zero or more message lists."""
    if is_activity_record(rec):
        stats["records_activity"] += 1
        pair = pair_from_activity(rec)
        if pair is None:
            stats["skipped_non_prompt"] += 1
            return
        yield [{"role": "user", "content": pair[0]}, {"role": "assistant", "content": pair[1]}]
        return

    if isinstance(rec, dict):
        turns = find_turn_list(rec)
        if turns is not None:
            stats["records_conversation"] += 1
            yield from examples_from_conversation(turns, multi_turn)
            return
        # Single turn object with prompt/response fields.
        for uk, ak in (("prompt", "response"), ("input", "output"), ("question", "answer"),
                       ("user", "model"), ("query", "reply")):
            if uk in rec and ak in rec:
                stats["records_pair"] += 1
                yield [{"role": "user", "content": clean_text(extract_text(rec[uk]))},
                       {"role": "assistant", "content": clean_text(extract_text(rec[ak]))}]
                return
        # Nested containers (e.g. {"conversations": [...]}).
        for val in rec.values():
            if isinstance(val, list):
                for sub in val:
                    yield from walk_records(sub, multi_turn, stats)
        return

    if isinstance(rec, list):
        if any(role_and_text(t) for t in rec):
            stats["records_conversation"] += 1
            yield from examples_from_conversation(rec, multi_turn)
        else:
            for sub in rec:
                yield from walk_records(sub, multi_turn, stats)
        return

    stats["skipped_unknown"] += 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
class _Progress:
    """tqdm if available, otherwise periodic prints. Tracks bytes read when possible."""

    def __init__(self, total_bytes: int) -> None:
        self.count = 0
        self.bar = tqdm(desc="Records", unit=" rec", dynamic_ncols=True) if tqdm else None
        self.total_bytes = total_bytes

    def update(self, written: int) -> None:
        self.count += 1
        if self.bar is not None:
            self.bar.update(1)
            if self.count % 500 == 0:
                self.bar.set_postfix(written=written)
        elif self.count % 1000 == 0:
            print(f"  ... processed {self.count:,} records, wrote {written:,} examples", flush=True)

    def close(self, written: int) -> None:
        if self.bar is not None:
            self.bar.set_postfix(written=written)
            self.bar.close()


def _parse_time(value: Any) -> Any:
    """Best-effort timestamp for ordering turns ('2025-01-01T12:00:00Z' or 'Nov 9, 2026, 9:04:53 PM CST')."""
    if not isinstance(value, str):
        return None
    v = value.replace("\u202f", " ").replace("\u00a0", " ").strip()
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    v = re.sub(r"\s+[A-Za-z][A-Za-z0-9+:\-]*$", "", v)  # drop trailing time zone
    for fmt in ("%b %d, %Y, %I:%M:%S %p", "%b %d, %Y, %I:%M %p", "%b %d, %Y %I:%M:%S %p"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def convert(in_path: str, out_path: str, multi_turn: bool = False, dedupe: bool = True) -> dict:
    stats = {k: 0 for k in (
        "files_read", "files_skipped", "records_total", "records_activity", "records_conversation", "records_pair",
        "skipped_non_prompt", "skipped_unknown", "skipped_empty_or_metadata",
        "skipped_duplicate", "written",
    )}
    seen: set[int] = set()
    if os.path.isdir(in_path):
        size = sum(os.path.getsize(os.path.join(r, n)) for r, _d, fs in os.walk(in_path) for n in fs)
    else:
        size = os.path.getsize(in_path)

    print(f"Input : {in_path} ({size / 1_048_576:,.1f} MB)")
    print(f"Output: {out_path}")
    print(f"Parser: {'ijson (streaming)' if ijson else 'json (loads whole file into memory)'}")
    if ijson is None and size > 500 * 1_048_576:
        print("  Tip: `pip install ijson` to stream this file with much lower memory use.")

    progress = _Progress(size)
    # With --multi-turn, activity records that carry a chat id are collected per chat and
    # written at the end, oldest turn first (Takeout lists activity newest-first).
    chats: dict[str, list[tuple[int, Any, tuple[str, str]]]] = {}
    tmp_path = out_path + ".partial"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as out:

        def emit(messages: list[dict]) -> None:
            if dedupe:
                key = hash(json.dumps(messages, ensure_ascii=False, sort_keys=True))
                if key in seen:
                    stats["skipped_duplicate"] += 1
                    return
                seen.add(key)
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            stats["written"] += 1

        for rec in iter_top_level(in_path, stats):
            stats["records_total"] += 1
            if multi_turn and is_activity_record(rec) and rec.get("conversation_id"):
                stats["records_activity"] += 1
                pair = pair_from_activity(rec)
                if pair is None:
                    stats["skipped_non_prompt"] += 1
                elif not valid_pair(*pair):
                    stats["skipped_empty_or_metadata"] += 1
                else:
                    turns = chats.setdefault(rec["conversation_id"], [])
                    turns.append((-len(turns), _parse_time(rec.get("time")), pair))
                progress.update(stats["written"])
                continue
            for messages in walk_records(rec, multi_turn, stats):
                if not multi_turn or len(messages) == 2:
                    u, a = messages[0]["content"], messages[1]["content"]
                    if not valid_pair(u, a):
                        stats["skipped_empty_or_metadata"] += 1
                        continue
                emit(messages)
            progress.update(stats["written"])

        for turns in chats.values():
            if all(t[1] is not None for t in turns):
                turns.sort(key=lambda t: (t[1], t[0]))
            else:
                turns.sort(key=lambda t: t[0])  # reverse of file order
            msgs: list[dict] = []
            for _i, _t, (u, a) in turns:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            emit(msgs)
    progress.close(stats["written"])
    os.replace(tmp_path, out_path)
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert Google Takeout Gemini history to fine-tuning JSONL.")
    ap.add_argument("input", help="A Takeout .zip, an unzipped Takeout folder, or a single JSON/HTML file "
                                   "(e.g. 'My Activity/Gemini Apps/MyActivity.json')")
    ap.add_argument("-o", "--output", default="gemini_finetune.jsonl", help="Output .jsonl path")
    ap.add_argument("--multi-turn", action="store_true",
                    help="Emit one line per chat (all its turns, oldest first) instead of one "
                         "line per user/assistant pair.")
    ap.add_argument("--keep-duplicates", action="store_true", help="Do not drop identical examples")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    try:
        stats = convert(args.input, args.output, args.multi_turn, not args.keep_duplicates)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: could not parse JSON: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; partial output left at", args.output + ".partial", file=sys.stderr)
        return 130

    print("\nDone.")
    print(f"  Files read               : {stats['files_read']:,}"
          + (f"  ({stats['files_skipped']:,} not JSON, skipped)" if stats["files_skipped"] else ""))
    print(f"  Records read             : {stats['records_total']:,}")
    print(f"    activity records       : {stats['records_activity']:,}")
    print(f"    conversation records   : {stats['records_conversation']:,}")
    print(f"  Skipped (not a prompt)   : {stats['skipped_non_prompt']:,}")
    print(f"  Skipped (empty/metadata) : {stats['skipped_empty_or_metadata']:,}")
    print(f"  Skipped (duplicate)      : {stats['skipped_duplicate']:,}")
    print(f"  Skipped (unrecognised)   : {stats['skipped_unknown']:,}")
    print(f"  Examples written         : {stats['written']:,}  ->  {args.output}")
    if stats["written"] == 0:
        print("\nWarning: nothing was written. Check that the input is a Gemini export in JSON form "
              "(for My Activity, choose JSON, not HTML, when exporting from Takeout).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
