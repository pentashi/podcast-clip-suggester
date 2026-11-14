#!/usr/bin/env python3
"""Podcast Clip Suggester.

Turns one podcast episode (audio file or transcript JSON) into:
  - a timestamped transcript,
  - five suggested 30-45 s clips with a "why" line and a draft caption,
  - a draft episode title and a short summary,
  - review.html (self-contained) and clips.json.

Everything is rule-based and extractive: captions, title and summary are
sentences copied word for word from the transcript. Nothing is published.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "review_template.html"

MIN_CLIP = 30.0
MAX_CLIP = 45.0
MIN_GAP = 10.0
NUM_CLIPS = 5

# --- Signal lexicons (hand-written guesses, not learned) ---------------------
STORY = [
    "i remember", "one day", "one night", "one winter", "years ago", "back then",
    "the first time", "that's when", "that was the moment", "the moment",
    "suddenly", "turned out", "ended up", "turning point", "i realised", "i realized", "i decided",
    "i started", "i was", "we were", "next morning", "after that",
]
EMOTION = [
    "terrified", "scared", "afraid", "nervous", "cried", "tears", "angry", "upset",
    # everyday praise words ("love", "best", "amazing") left out: in interviews they are
    # mostly politeness and pushed weak clips up
    "proud", "excited", "grateful", "heartbroken", "lost everything",
    "worst", "shocked", "painful", "struggle", "struggled", "broke",
    "hard way", "never expected", "panic", "alone", "lose everything",
]
TAKEAWAY = [
    "the lesson", "i learned", "what i learned", "my advice", "advice",
    "the key", "the secret", "you need", "you have to", "you don't need", "if you",
    "the truth is", "the point is", "here's the thing", "most important",
    "biggest mistake", "nobody tells you", "start small", "start before",
    "the reason", "actually", "better to", "rather than", "makes a difference",
    "massive difference", "the problem", "make sure", "comes down to", "the thing is",
    "mistake", "the stats", "what matters",
]
# concrete numbers make advice quotable ("90% of people", "15 minutes")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|percent\b)|\b\d+\b|\b(?:ten|fifteen|twenty|thirty|"
                       r"forty|fifty|hundred|thousand|half)\b", re.I)
HOOK_OPENERS = [
    "here's", "the truth is", "nobody", "imagine", "what if", "the biggest",
    "i'll never forget", "let me tell you", "that's when", "my advice", "the lesson",
    "i remember", "three years ago", "honestly",
]
FILLERS = ["um", "uh", "erm", "ah", "like,", "you know", "i mean", "sort of",
           "kind of", "basically", "i guess", "yeah,"]
MID_THOUGHT_STARTERS = {"and", "but", "so", "he", "she", "it", "they", "because",
                        "or", "then", "which", "that", "also", "plus", "this",
                        "whereas", "like", "yeah", "okay", "um", "uh", "them"}
PRONOUN_STARTS = MID_THOUGHT_STARTERS | {"these", "those", "there", "we", "you", "her", "his"}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z']+")


class ClipSuggesterError(Exception):
    """Expected, user-facing problem (bad input, missing dependency...)."""


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Clip:
    start: float
    end: float
    first: int  # first segment index
    last: int   # last segment index (inclusive)
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    why: str = ""
    caption: str = ""
    excerpt: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# --- Helpers ------------------------------------------------------------------
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def full_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments)


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]


def sentences_with_times(segments: list[Segment]) -> tuple[list[str], list[float]]:
    """Sentences of the joined transcript, each with the start time of its segment."""
    offsets, pos = [], 0
    for seg in segments:
        offsets.append(pos)
        pos += len(seg.text) + 1
    text = full_text(segments)
    sents, starts, k = [], [], 0
    for m in re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", text):
        while k + 1 < len(offsets) and offsets[k + 1] <= m.start():
            k += 1
        sents.append(m.group(0).strip())
        starts.append(segments[k].start)
    return sents, starts


def find_phrases(text: str, phrases: list[str]) -> list[str]:
    """Phrases present in text; a phrase inside a longer found one is dropped."""
    low = " " + text.lower() + " "
    found = [p for p in phrases
             if re.search(r"(?<![a-z'])" + re.escape(p) + r"(?![a-z'])", low)]
    return [p for p in found if not any(p != q and p in q for q in found)]


def count_fillers(text: str) -> int:
    low = " " + text.lower() + " "
    n = 0
    for f in FILLERS:
        n += len(re.findall(r"(?<![a-z'])" + re.escape(f) + r"(?![a-z'])", low))
    return n


def first_word(text: str) -> str:
    m = WORD_RE.search(text.lower())
    return m.group(0) if m else ""


# --- Input ----------------------------------------------------------------------
def load_transcript(path: str | os.PathLike) -> list[Segment]:
    p = Path(path)
    if not p.is_file():
        raise ClipSuggesterError(f"Transcript file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ClipSuggesterError(f"Transcript is not valid JSON ({p}): {e}") from e
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    if not isinstance(data, list):
        raise ClipSuggesterError("Transcript JSON must be a list of {start, end, text} objects.")
    return normalise_segments(data)


def normalise_segments(items) -> list[Segment]:
    segs: list[Segment] = []
    for i, it in enumerate(items):
        try:
            start, end, text = float(it["start"]), float(it["end"]), clean(it["text"])
        except (KeyError, TypeError, ValueError) as e:
            raise ClipSuggesterError(
                f"Transcript segment {i} must have numeric 'start', 'end' and a 'text' field.") from e
        if not text:
            continue
        if end < start:
            raise ClipSuggesterError(f"Transcript segment {i} ends before it starts ({start} > {end}).")
        segs.append(Segment(start, end, text))
    segs.sort(key=lambda s: s.start)
    if not segs:
        raise ClipSuggesterError("Transcript is empty: no speech/text segments found.")
    return segs


def is_prompt_echo(text: str, names: list[str]) -> bool:
    """True when a segment is only the spelling prompt repeated back (a Whisper hallucination)."""
    said = WORD_RE.findall(text.lower())
    prompt = set(WORD_RE.findall(" ".join(names).lower()))
    return len(said) >= 2 and all(w in prompt for w in said)


def transcribe(audio_path: str, model_size: str = "small", language: str | None = None,
               names: list[str] | None = None) -> list[Segment]:
    """Transcribe with faster-whisper on CPU (int8, VAD on).

    Whisper segments can be long, so they are split at sentence ends using
    word timestamps to give the clip finder finer boundaries.
    `names` (people, brands, places) go into Whisper's opening prompt so the
    intro spells them correctly. (Not as hotwords: on this setup Whisper then
    sometimes wrote the names in place of real speech.)
    """
    p = Path(audio_path)
    if not p.is_file():
        raise ClipSuggesterError(f"Audio file not found: {p}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ClipSuggesterError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt`, "
            "or pass a ready-made transcript with --transcript.") from e

    print(f"Loading Whisper model '{model_size}' (CPU, int8). First run downloads it once...", file=sys.stderr)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    try:
        seg_iter, info = model.transcribe(str(p), vad_filter=True, word_timestamps=True,
                                          language=language, beam_size=5,
                                          # stops punctuation fading out on long episodes
                                          condition_on_previous_text=False,
                                          initial_prompt=", ".join(names) + "." if names else None)
    except Exception as e:  # decoding errors from PyAV / ctranslate2
        raise ClipSuggesterError(f"Could not read or transcribe the audio file {p}: {e}") from e
    print(f"Detected language: {info.language} (p={info.language_probability:.2f}); "
          f"audio length {fmt_time(info.duration)}", file=sys.stderr)

    out = []
    for seg in seg_iter:
        if names and is_prompt_echo(seg.text, names):
            print(f"  [{fmt_time(seg.start)}] (dropped: Whisper repeated the names prompt)", file=sys.stderr)
            continue
        print(f"  [{fmt_time(seg.start)}] {seg.text.strip()}", file=sys.stderr)
        words = list(seg.words or [])
        if not words:
            out.append({"start": seg.start, "end": seg.end, "text": seg.text})
            continue
        chunk = []
        for w in words:
            chunk.append(w)
            if w.word.strip().endswith((".", "?", "!")) and len(chunk) >= 3:
                out.append({"start": chunk[0].start, "end": chunk[-1].end,
                            "text": "".join(x.word for x in chunk)})
                chunk = []
        if chunk:
            out.append({"start": chunk[0].start, "end": chunk[-1].end,
                        "text": "".join(x.word for x in chunk)})
    return normalise_segments(out)


# --- Topic changes ----------------------------------------------------------------
def make_blocks(segments: list[Segment], block_seconds: float) -> list[list[int]]:
    blocks, cur, cur_start = [], [], None
    for i, s in enumerate(segments):
        if cur_start is None:
            cur_start = s.start
        cur.append(i)
        if s.end - cur_start >= block_seconds:
            blocks.append(cur)
            cur, cur_start = [], None
    if cur:
        if blocks and segments[cur[-1]].end - segments[cur[0]].start < block_seconds / 2:
            blocks[-1].extend(cur)
        else:
            blocks.append(cur)
    return blocks


def detect_topic_changes(segments: list[Segment], block_seconds: float = 30.0,
                         context_blocks: int = 2, min_depth: float = 0.1,
                         min_spacing: float = 60.0) -> list[float]:
    """TextTiling-style boundaries using TF-IDF cosine similarity.

    The transcript is cut into ~block_seconds blocks. At each gap between
    blocks we compare the text of the `context_blocks` before with the ones
    after. A gap whose similarity is a deep valley (depth >= cutoff) is a
    topic change. Returns boundary times (start of the block after the gap).
    Changes in the first/last `context_blocks` blocks are not detected.
    """
    blocks = make_blocks(segments, block_seconds)
    if len(blocks) < 2 * context_blocks:
        return []
    texts = [" ".join(segments[i].text for i in b) for b in blocks]
    try:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts)
    except ValueError:  # only stop words
        return []
    # Only gaps with full context on both sides: edge comparisons use less
    # text and look artificially dissimilar.
    gaps = list(range(context_blocks, len(blocks) - context_blocks + 1))
    if not gaps:
        return []
    sims = []
    for g in gaps:
        left = " ".join(texts[max(0, g - context_blocks):g])
        right = " ".join(texts[g:g + context_blocks])
        m = vec.transform([left, right])
        sims.append(float(cosine_similarity(m[0], m[1])[0, 0]))
    sims = np.array(sims)
    depths = np.zeros_like(sims)
    for i, s in enumerate(sims):
        lp = s
        for j in range(i - 1, -1, -1):
            if sims[j] < lp:
                break
            lp = sims[j]
        rp = s
        for j in range(i + 1, len(sims)):
            if sims[j] < rp:
                break
            rp = sims[j]
        depths[i] = (lp - s) + (rp - s)
    positive = depths[depths > 0]
    if positive.size == 0:
        return []
    cutoff = max(min_depth, float(positive.mean() - positive.std() / 2))
    cands = sorted(((depths[i], i) for i in range(len(sims)) if depths[i] >= cutoff), reverse=True)
    chosen: list[float] = []
    for _, i in cands:
        t = segments[blocks[gaps[i]][0]].start
        if all(abs(t - c) >= min_spacing for c in chosen):
            chosen.append(t)
    return sorted(chosen)


# --- Cold-open / teaser detection --------------------------------------------------
def _ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    w = WORD_RE.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def repeated_segments(segments: list[Segment], min_ratio: float = 0.5,
                      min_distance: float = 60.0) -> set[int]:
    """Indexes of segments whose words are said again later (e.g. a cold-open teaser
    that previews lines from the interview). The later, original copy is kept."""
    grams = [_ngrams(s.text) for s in segments]
    last_said: dict[tuple[str, ...], float] = {}
    for seg, g in zip(segments, grams):
        for x in g:
            last_said[x] = seg.start
    out = set()
    for i, g in enumerate(grams):
        if len(g) < 3:
            continue
        again = sum(1 for x in g if last_said[x] >= segments[i].end + min_distance)
        if again / len(g) >= min_ratio:
            out.add(i)
    return out


# --- Candidate windows and scoring -----------------------------------------------
def candidate_windows(segments: list[Segment], min_len: float = MIN_CLIP,
                      max_len: float = MAX_CLIP) -> list[Clip]:
    """Every window that starts and ends on segment boundaries and lasts min..max seconds."""
    out = []
    n = len(segments)
    for i in range(n):
        for j in range(i, n):
            d = segments[j].end - segments[i].start
            if d > max_len:
                break
            if d >= min_len:
                out.append(Clip(segments[i].start, segments[j].end, i, j))
    return out


def score_window(clip: Clip, segments: list[Segment], boundaries: list[float]) -> Clip:
    segs = segments[clip.first:clip.last + 1]
    text = full_text(segs)
    words = max(1, len(text.split()))
    sentences = split_sentences(text)
    opener = sentences[0] if sentences else text

    story = find_phrases(text, STORY)
    emotion = find_phrases(text, EMOTION)
    takeaway = find_phrases(text, TAKEAWAY)
    numbers = list(dict.fromkeys(m.group(0) for m in NUMBER_RE.finditer(text)))
    hook = find_phrases(opener[:120], HOOK_OPENERS)
    opener_question = opener.endswith("?")

    fillers = count_fillers(text)
    gaps = [max(0.0, b.start - a.end) for a, b in zip(segs, segs[1:])]
    long_silence = sum(g for g in gaps if g > 1.5)
    crosses = [b for b in boundaries if clip.start + 2 < b < clip.end - 2]
    fw = first_word(text)
    prev_unfinished = clip.first > 0 and not segments[clip.first - 1].text.endswith((".", "?", "!"))
    mid_thought = fw in MID_THOUGHT_STARTERS or prev_unfinished
    ends_clean = text.endswith((".", "?", "!"))

    parts = {
        "story": min(len(story), 3) / 3,
        "emotion": min(len(emotion), 2) / 2,
        "point": min(len(takeaway), 3) / 3,
        "specific": min(len(numbers), 2) / 2,
        "hook": 1.0 if hook else (0.5 if opener_question else 0.0),
        "filler_penalty": min(1.0, fillers / words * 20),
        "silence_penalty": min(1.0, long_silence / 6),
        "topic_cross_penalty": 1.0 if crosses else 0.0,
        "mid_thought_penalty": 1.0 if mid_thought else 0.0,
        "unfinished_end_penalty": 0.0 if ends_clean else 1.0,
    }
    score = (0.25 * parts["story"] + 0.15 * parts["emotion"] + 0.35 * parts["point"]
             + 0.15 * parts["specific"] + 0.20 * parts["hook"]
             - 0.25 * parts["filler_penalty"] - 0.20 * parts["silence_penalty"]
             - 0.30 * parts["topic_cross_penalty"] - 0.40 * parts["mid_thought_penalty"]
             - 0.10 * parts["unfinished_end_penalty"])
    clip.score = round(score, 3)
    clip.parts = {k: round(v, 3) for k, v in parts.items()}
    clip.signals = {"story": story, "emotion": emotion, "takeaway": takeaway, "numbers": numbers,
                    "hook": hook, "opens_with_question": opener_question,
                    "filler_count": fillers, "long_silence_s": round(long_silence, 1),
                    "crosses_topic_change_at": crosses, "starts_mid_thought": mid_thought}
    clip.excerpt = text
    return clip


def non_overlapping(a: Clip, b: Clip, gap: float = MIN_GAP) -> bool:
    return a.end + gap <= b.start or b.end + gap <= a.start


def select_clips(cands: list[Clip], k: int = NUM_CLIPS, gap: float = MIN_GAP,
                 boundaries: list[float] | None = None) -> list[Clip]:
    """Greedy best-first pick, at most one clip per topic section while possible.
    If greedy finds fewer than k but k would fit, fall back to a small DP that
    maximises count, then total score."""
    ranked = sorted(cands, key=lambda c: (-c.score, c.start))
    section = lambda c: sum(1 for b in (boundaries or []) if b <= c.start)  # noqa: E731
    greedy: list[Clip] = []
    for one_per_section in ([True, False] if boundaries else [False]):
        for c in ranked:
            if len(greedy) == k:
                break
            if c in greedy or not all(non_overlapping(c, g, gap) for g in greedy):
                continue
            if one_per_section and any(section(c) == section(g) for g in greedy):
                continue
            greedy.append(c)
    if len(greedy) == k or not cands:
        return sorted(greedy, key=lambda c: c.start)

    by_end = sorted(cands, key=lambda c: c.end)
    ends = [c.end for c in by_end]
    import bisect
    n = len(by_end)
    # best[i][m] = (score, picks) using first i candidates, m picks
    NEG = (-1e9, ())
    best = [[NEG] * (k + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        best[i][0] = (0.0, ())
    for i in range(1, n + 1):
        c = by_end[i - 1]
        p = bisect.bisect_right(ends, c.start - gap, 0, i - 1)
        for m in range(1, k + 1):
            skip = best[i - 1][m]
            prev = best[p][m - 1]
            take = (prev[0] + c.score, prev[1] + (i - 1,)) if prev[0] > -1e8 else NEG
            best[i][m] = max(skip, take, key=lambda x: x[0])
    for m in range(k, 0, -1):
        if best[n][m][0] > -1e8:
            if m > len(greedy):
                return sorted((by_end[i] for i in best[n][m][1]), key=lambda c: c.start)
            break
    return sorted(greedy, key=lambda c: c.start)


# --- Extractive text ------------------------------------------------------------------
def explain(clip: Clip) -> str:
    s = clip.signals
    q = lambda xs: ", ".join(f'"{x}"' for x in sorted(xs, key=len, reverse=True)[:2])
    reasons = []
    if s["story"]:
        reasons.append(f"story markers ({q(s['story'])})")
    if s["emotion"]:
        reasons.append(f"emotional words ({q(s['emotion'])})")
    if s["takeaway"]:
        reasons.append(f"a clear point ({q(s['takeaway'])})")
    if s.get("numbers"):
        reasons.append(f"specific numbers ({q(s['numbers'])})")
    if s["hook"]:
        reasons.append(f"a strong opening ({q(s['hook'])})")
    elif s["opens_with_question"]:
        reasons.append("opens with a question")
    if not reasons:
        reasons.append("no strong signals; picked as the best remaining self-contained stretch")
    extra = []
    if not s["starts_mid_thought"]:
        extra.append("starts on a full sentence")
    if s["filler_count"] == 0:
        extra.append("no filler words")
    line = "Has " + "; ".join(reasons) if not reasons[0].startswith("no strong") else reasons[0].capitalize()
    if extra:
        line += "; " + ", ".join(extra)
    return line + "."


def truncate_words(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "..."


def sentence_score(sent: str) -> float:
    n = len(sent.split())
    sc = (len(find_phrases(sent, STORY)) + 1.5 * len(find_phrases(sent, EMOTION))
          + 1.5 * len(find_phrases(sent, TAKEAWAY)) + 1.0 * len(find_phrases(sent, HOOK_OPENERS))
          + 1.0 * min(2, len(NUMBER_RE.findall(sent))))
    if 8 <= n <= 35:
        sc += 1.0
    elif n < 6:
        sc -= 1.0
    if first_word(sent) in MID_THOUGHT_STARTERS:
        sc -= 1.5
    if count_fillers(sent):
        sc -= 2.0
    if sent.endswith("?"):
        sc -= 2.0  # usually the host's question, not the quotable answer
    return sc


def draft_caption(clip: Clip, max_chars: int = 220,
                  segments: list[Segment] | None = None) -> str:
    """Best whole sentence (or two short consecutive ones) in the clip that fits in
    max_chars, verbatim. When Whisper left long unpunctuated run-ons, runs of whole
    transcript lines inside the clip are also considered; those end in "..." if
    they stop mid-sentence."""
    sents = split_sentences(clip.excerpt)
    options = [(t, sentence_score(t)) for t in sents]
    options += [(a + " " + b, sentence_score(a + " " + b)) for a, b in zip(sents, sents[1:])
                if len(a.split()) < 12]
    options = [(t, sc) for t, sc in options if len(t) <= max_chars and len(t.split()) >= 5]
    if segments is not None:
        segs = segments[clip.first:clip.last + 1]
        for i in range(len(segs)):
            starts_sentence = (clip.first + i == 0
                               or segments[clip.first + i - 1].text.endswith((".", "?", "!")))
            text = ""
            for seg in segs[i:]:
                nxt = (text + " " + seg.text).strip()
                if len(nxt) > max_chars:
                    break
                text = nxt
                if len(text.split()) >= 8 and text not in (t for t, _ in options):
                    sc = sentence_score(text) - (0 if starts_sentence else 5.0)
                    if not text.endswith((".", "?", "!")):
                        sc -= 1.5
                    options.append((text, sc))
    if not options:
        best = max(sents or [clip.excerpt], key=sentence_score)
        return truncate_words(best, max_chars)
    best, _ = max(options, key=lambda x: (x[1], -clip.excerpt.find(x[0])))
    return best if best.endswith((".", "?", "!")) else best.rstrip(",;:") + "..."


INTRO_MARKERS = ["my guest", "joining me", "joining us", "today", "this episode", "welcome"]


def _centrality(sents: list[str]) -> np.ndarray:
    """Cosine similarity of each sentence to the TF-IDF centroid of all of them."""
    try:
        m = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sents)
    except ValueError:
        return np.zeros(len(sents))
    return cosine_similarity(m, np.asarray(m.mean(axis=0))).ravel()


def draft_titles(segments: list[Segment], guest: str | None = None, n: int = 3) -> list[str]:
    """Up to n short transcript sentences that work as a title: 6-14 words, no
    question, no filler, not starting with a pronoun or mid-thought word."""
    sents = split_sentences(full_text(segments))
    cands = [s for s in sents if 6 <= len(s.split()) <= 14 and not s.endswith("?")
             and first_word(s) not in PRONOUN_STARTS and not count_fillers(s)
             and len(set(WORD_RE.findall(s.lower())) - ENGLISH_STOP_WORDS) >= 3]
    if not cands:
        cands = sents
    central = _centrality(cands)

    def title_score(i: int) -> float:
        x = cands[i]
        return (2 * len(find_phrases(x, TAKEAWAY)) + len(find_phrases(x, HOOK_OPENERS))
                + len(find_phrases(x, STORY)) + 0.5 * len(find_phrases(x, EMOTION))
                + min(2, len(NUMBER_RE.findall(x))) + 4 * central[i]
                + (1.0 if len(x.split()) <= 10 else 0.0))
    picked: list[str] = []
    for i in sorted(range(len(cands)), key=lambda i: (-title_score(i), i)):
        t = truncate_words(cands[i], 90)
        if t.lower() not in (p.lower() for p in picked):
            picked.append(t)
        if len(picked) == n:
            break
    return [f"{guest}: {t}" if guest else t for t in picked]


def draft_title(segments: list[Segment], guest: str | None = None) -> str:
    return draft_titles(segments, guest, 1)[0]


def draft_summary(segments: list[Segment], boundaries: list[float], n: int = 3) -> str:
    """One complete sentence from each of the early, middle and late part of the
    episode (the most central usable one), in time order. An intro line ("thanks
    for joining us...") near the start is preferred for the first part."""
    # Split the joined text (not each segment): Whisper often ends a segment mid-sentence.
    sents, starts = sentences_with_times(segments)
    seen: set[str] = set()
    dup = set()
    for i, x in enumerate(sents):
        key = " ".join(WORD_RE.findall(x.lower()))
        if key in seen:
            dup.add(i)
        seen.add(key)
    ok = [i for i, x in enumerate(sents) if 8 <= len(x.split()) <= 35 and i not in dup
          and x.endswith(".") and not count_fillers(x)
          and first_word(x) not in PRONOUN_STARTS]
    if not ok:
        ok = [i for i in range(len(sents)) if i not in dup] or list(range(len(sents)))
    if len(ok) <= n:
        return " ".join(sents[i] for i in ok)
    central = _centrality(sents)
    rank = lambda i: central[i] + 0.05 * sentence_score(sents[i])  # noqa: E731
    t0, t1 = segments[0].start, segments[-1].end
    part = lambda i: min(n - 1, int(n * (starts[i] - t0) / max(t1 - t0, 1e-9)))  # noqa: E731
    picked = []
    for k in range(n):
        pool = [i for i in ok if part(i) == k]
        if k == 0:
            intro = [i for i in pool if find_phrases(sents[i], INTRO_MARKERS)]
            pool = intro or pool
        if pool:
            picked.append(max(pool, key=rank))
    for i in sorted(ok, key=rank, reverse=True):
        if len(picked) >= n:
            break
        if i not in picked:
            picked.append(i)
    return " ".join(sents[i] for i in sorted(picked))


# --- Pipeline ----------------------------------------------------------------------
def suggest(segments: list[Segment], guest: str | None = None, k: int = NUM_CLIPS) -> dict:
    if not segments:
        raise ClipSuggesterError("Transcript is empty: no speech/text segments found.")
    duration = segments[-1].end - segments[0].start
    if duration < MIN_CLIP:
        raise ClipSuggesterError(
            f"Episode is too short ({duration:.0f}s of speech). Need at least {MIN_CLIP:.0f}s for one clip.")
    boundaries = detect_topic_changes(segments)
    repeats = repeated_segments(segments)
    cands = [score_window(c, segments, boundaries) for c in candidate_windows(segments)
             if sum(1 for i in range(c.first, c.last + 1) if i in repeats) <= (c.last - c.first + 1) * 0.3]
    original = [s for i, s in enumerate(segments) if i not in repeats] or segments
    warnings = []
    if repeats:
        first, last = min(repeats), max(repeats)
        warnings.append(f"Skipped {len(repeats)} line(s) that are repeated later in the episode "
                        f"(likely a teaser/cold open, around {fmt_time(segments[first].start)}-"
                        f"{fmt_time(segments[last].end)}). Clips use the original moments.")
    if not cands:
        raise ClipSuggesterError(
            "No 30-45 second window lines up with the transcript's segment boundaries "
            "(segments may be too long). Try a transcript with shorter segments.")
    clips = select_clips(cands, k, boundaries=boundaries)
    if len(clips) < k:
        warnings.append(f"Only {len(clips)} non-overlapping 30-45s clip(s) fit in this "
                        f"{fmt_time(duration)} episode (asked for {k}).")
    for c in clips:
        c.why = explain(c)
        c.caption = draft_caption(c, segments=segments)
    return {
        "title_options": (titles := draft_titles(original, guest)),
        "title": titles[0],
        "summary": draft_summary(original, boundaries),
        "topic_changes": boundaries,
        "clips": clips,
        "warnings": warnings,
        "duration": segments[-1].end,
    }


def clip_line(i: int, c: dict | Clip) -> str:
    g = (lambda k: c[k]) if isinstance(c, dict) else (lambda k: getattr(c, k))
    return (f"Clip {i}. Start {fmt_time(g('start'))}. End {fmt_time(g('end'))}. "
            f"Why: {g('why')} Draft caption: {g('caption')}")


def build_payload(segments, result, *, source, guest, sample, audio_src, model) -> dict:
    return {
        "generator": "podcast-clip-suggester 0.1",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "audio_src": audio_src,
        "guest": guest,
        "sample": sample,
        "model": model,
        "duration": round(result["duration"], 2),
        "title": result["title"],
        "title_options": result["title_options"],
        "summary": result["summary"],
        "topic_changes": [round(b, 2) for b in result["topic_changes"]],
        "warnings": result["warnings"],
        "clips": [
            {"id": i + 1, "start": round(c.start, 2), "end": round(c.end, 2),
             "duration": round(c.duration, 2), "score": c.score, "why": c.why,
             "caption": c.caption, "excerpt": c.excerpt, "scores": c.parts,
             "signals": c.signals}
            for i, c in enumerate(result["clips"])
        ],
        "transcript": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
                       for s in segments],
        "note": "Suggestions only. Nothing is edited, posted or published. "
                "A human approves every clip.",
    }


def render_html(payload: dict, template_path: Path = TEMPLATE_PATH) -> str:
    if not template_path.is_file():
        raise ClipSuggesterError(f"HTML template not found: {template_path}")
    tpl = template_path.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    if "/*__DATA__*/null" not in tpl:
        raise ClipSuggesterError("HTML template is missing the /*__DATA__*/null placeholder.")
    return tpl.replace("/*__DATA__*/null", data)


def write_outputs(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(
        "\n".join(f"[{fmt_time(s['start'])} - {fmt_time(s['end'])}] {s['text']}"
                  for s in payload["transcript"]) + "\n", encoding="utf-8")
    (out_dir / "review.html").write_text(render_html(payload), encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Suggest five 30-45s clips from one podcast episode. Suggestions only; nothing is published.")
    ap.add_argument("audio", nargs="?", help="Episode audio file (mp3/wav/m4a). Used for transcription "
                                           "and for playback on the review page.")
    ap.add_argument("--transcript", help="Use a ready-made transcript JSON [{start,end,text}, ...] "
                                         "instead of transcribing.")
    ap.add_argument("--guest", help="Guest name (used only in the draft title).")
    ap.add_argument("--out", default="output", help="Output folder (default: output)")
    ap.add_argument("--model", default="small", help="faster-whisper model size (default: small)")
    ap.add_argument("--language", help="Force a language code, e.g. en (default: auto-detect)")
    ap.add_argument("--names", help='Comma-separated spellings Whisper should use, e.g. '
                                    '"Anish Ahuja, Imprint Studio, JLT". The --guest name is added automatically.')
    ap.add_argument("--sample", action="store_true", help='Mark output with a "SAMPLE: fictional text" banner.')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if not args.audio and not args.transcript:
            raise ClipSuggesterError("Give an audio file, or --transcript path/to/transcript.json.")
        if args.audio and not Path(args.audio).is_file():
            raise ClipSuggesterError(f"Audio file not found: {args.audio}")
        if args.transcript:
            segments = load_transcript(args.transcript)
            source, model = Path(args.transcript).name, None
        else:
            names = [n.strip() for n in (args.names or "").split(",") if n.strip()]
            if args.guest and args.guest not in names:
                names.append(args.guest)
            segments = transcribe(args.audio, args.model, args.language, names)
            source, model = Path(args.audio).name, f"faster-whisper {args.model} (cpu, int8, vad)"
        out_dir = Path(args.out)
        audio_src = None
        if args.audio:
            audio_src = os.path.relpath(Path(args.audio).resolve(), out_dir.resolve()).replace(os.sep, "/")
        result = suggest(segments, args.guest)
        payload = build_payload(segments, result, source=source, guest=args.guest,
                                sample=args.sample, audio_src=audio_src, model=model)
        write_outputs(payload, out_dir)
    except ClipSuggesterError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.sample:
        print("SAMPLE: fictional text")
    print(f"Title (draft): {payload['title']}")
    print(f"Summary (draft): {payload['summary']}")
    print(f"Topic changes at: {', '.join(fmt_time(b) for b in payload['topic_changes']) or 'none found'}")
    for w in payload["warnings"]:
        print(f"Warning: {w}")
    for c in payload["clips"]:
        print(clip_line(c["id"], c))
    print(f"\nWrote {out_dir/'review.html'}, {out_dir/'clips.json'}, {out_dir/'transcript.txt'}")
    print("Suggestions only. Nothing is edited, posted or published. A human approves every clip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
