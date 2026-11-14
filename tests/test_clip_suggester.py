import builtins
import json
from pathlib import Path

import pytest

import clip_suggester as cs

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_transcript.json"

COOKING = [
    "I bake bread every morning with flour, water, salt and a slow sourdough starter.",
    "The dough needs to rest overnight so the flour can absorb the water.",
    "A hot oven and a heavy pot give the bread a crisp crust.",
    "Good bread flour has more protein, which makes the dough stretchy.",
    "When the loaf sounds hollow, the bread is ready to cool.",
    "Never slice bread straight from the oven, the crumb is still cooking.",
]
SPACE = [
    "The rocket engine burns liquid oxygen and fuel to reach orbit.",
    "Satellites in low orbit circle the planet every ninety minutes.",
    "The launch window depends on where the space station is in its orbit.",
    "Astronauts train for years before a rocket launch.",
    "Telescopes in orbit can see galaxies billions of light years away.",
    "Re-entry heats the capsule as it falls back through the atmosphere.",
]


def make_segments(sentences, seg_len=5.0, gap=0.3, t0=0.0):
    segs, t = [], t0
    for s in sentences:
        segs.append(cs.Segment(round(t, 2), round(t + seg_len, 2), s))
        t += seg_len + gap
    return segs


def sample_segments():
    return cs.load_transcript(SAMPLE)


# --- topic boundary detection ------------------------------------------------
def test_topic_boundary_found_between_two_topics():
    segs = make_segments(COOKING * 3 + SPACE * 3)
    switch = segs[len(COOKING) * 3].start
    bounds = cs.detect_topic_changes(segs, block_seconds=20)
    assert bounds, "expected at least one topic change"
    assert min(abs(b - switch) for b in bounds) <= 21


def test_no_topic_boundary_in_single_topic():
    segs = make_segments(COOKING * 6)
    assert cs.detect_topic_changes(segs, block_seconds=20) == []


def test_topic_boundaries_are_segment_starts():
    segs = sample_segments()
    starts = {s.start for s in segs}
    for b in cs.detect_topic_changes(segs):
        assert b in starts


# --- window limits / non-overlap ----------------------------------------------
def test_candidate_windows_respect_length_limits():
    segs = sample_segments()
    cands = cs.candidate_windows(segs)
    assert cands
    for c in cands:
        assert cs.MIN_CLIP <= c.duration <= cs.MAX_CLIP


def test_exactly_five_clips_within_limits():
    result = cs.suggest(sample_segments())
    clips = result["clips"]
    assert len(clips) == 5
    for c in clips:
        assert 30 <= c.duration <= 45


def test_clips_do_not_overlap_and_are_10s_apart():
    clips = cs.suggest(sample_segments())["clips"]
    for i, a in enumerate(clips):
        for b in clips[i + 1:]:
            assert a.end + 10 <= b.start or b.end + 10 <= a.start


def test_select_clips_fallback_finds_more_than_greedy():
    # The best window sits in the middle and blocks two others; DP should pick the two.
    a = cs.Clip(0, 30, 0, 0, score=0.5)
    mid = cs.Clip(25, 60, 0, 0, score=0.9)
    b = cs.Clip(40, 70, 0, 0, score=0.5)
    picked = cs.select_clips([a, mid, b], k=2, gap=10)
    assert [(c.start, c.end) for c in picked] == [(0, 30), (40, 70)]


# --- extractive only -----------------------------------------------------------
def _strip_ellipsis(s):
    return s[:-3] if s.endswith("...") else s


def test_captions_title_summary_are_verbatim():
    segs = sample_segments()
    text = cs.full_text(segs)
    result = cs.suggest(segs, guest="Guest Name")
    for c in result["clips"]:
        assert _strip_ellipsis(c.caption) in text, c.caption
        assert c.excerpt in text
    title = result["title"]
    assert title.startswith("Guest Name: ")
    assert _strip_ellipsis(title[len("Guest Name: "):]) in text
    for sent in cs.split_sentences(result["summary"]):
        assert sent in text, sent


def test_why_line_only_cites_found_phrases():
    segs = sample_segments()
    text = cs.full_text(segs).lower()
    for c in cs.suggest(segs)["clips"]:
        for quoted in c.why.split('"')[1::2]:
            assert quoted in c.excerpt.lower() and quoted in text


def test_truncated_caption_is_still_a_substring():
    long = " ".join(["word"] * 80) + "."
    clip = cs.Clip(0, 40, 0, 0, excerpt=long)
    cap = cs.draft_caption(clip, max_chars=60)
    assert cap.endswith("...") and _strip_ellipsis(cap) in long


# --- short / empty / errors ----------------------------------------------------
def test_short_transcript_returns_fewer_clips_with_warning():
    segs = make_segments(COOKING * 2)  # ~64 s: room for one clip only
    result = cs.suggest(segs)
    assert 1 <= len(result["clips"]) < 5
    assert result["warnings"]


def test_very_short_episode_raises_clear_error():
    segs = make_segments(COOKING[:2])  # ~10 s
    with pytest.raises(cs.ClipSuggesterError, match="too short"):
        cs.suggest(segs)


def test_empty_transcript_raises(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("[]")
    with pytest.raises(cs.ClipSuggesterError, match="empty"):
        cs.load_transcript(p)
    p.write_text(json.dumps([{"start": 0, "end": 1, "text": "   "}]))
    with pytest.raises(cs.ClipSuggesterError, match="empty"):
        cs.load_transcript(p)


def test_bad_transcript_segment_raises(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"start": "x", "text": "hi"}]))
    with pytest.raises(cs.ClipSuggesterError, match="numeric"):
        cs.load_transcript(p)


def test_missing_audio_file_is_clear_error(capsys):
    assert cs.main(["does_not_exist.mp3"]) == 2
    assert "Audio file not found" in capsys.readouterr().err


def test_missing_faster_whisper_is_clear_error(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "faster_whisper":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(cs.ClipSuggesterError, match="faster-whisper is not installed"):
        cs.transcribe(str(audio))


# --- end to end ------------------------------------------------------------------
def test_end_to_end_writes_self_contained_outputs(tmp_path, capsys):
    out = tmp_path / "out"
    assert cs.main(["--transcript", str(SAMPLE), "--sample", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "SAMPLE: fictional text" in printed
    assert printed.count("Clip ") >= 5 and "Draft caption:" in printed
    data = json.loads((out / "clips.json").read_text())
    assert len(data["clips"]) == 5 and data["sample"] is True
    html = (out / "review.html").read_text()
    assert "/*__DATA__*/" not in html
    assert "SAMPLE: fictional text" in html
    for bad in ("<script src", "<link", "http://", "https://", "@import"):
        assert bad not in html.lower()
    assert (out / "transcript.txt").read_text().startswith("[0:01 - ")


def test_summary_uses_whole_sentences_across_segment_breaks():
    # Whisper often splits one sentence over two segments.
    halves = []
    for sent in COOKING * 4:
        words = sent.split()
        halves += [" ".join(words[:5]), " ".join(words[5:])]
    segs = make_segments(halves, seg_len=3.0)
    summary = cs.draft_summary(segs, [])
    sents = cs.split_sentences(summary)
    assert sents
    for x in sents:
        assert x in COOKING


def test_cold_open_teaser_is_detected_and_not_clipped():
    teaser = ["Losing my first shop taught me that pricing is everything in this business.",
              "The city feels like home now and my whole team is here.",
              "Customers remember how they felt when they left the shop.",
              "My advice is to step out of the business and work on it.",
              "We hit number one in five of the charts with our first launch.",
              "Working all hours for little profit is the most common problem I see."]
    body = COOKING * 2 + teaser + SPACE * 2
    segs = make_segments(teaser + body)
    assert cs.repeated_segments(segs) == set(range(6))
    result = cs.suggest(segs)
    # a window may overlap at most 30% teaser lines, so none can start in its first half
    assert all(c.start >= segs[3].start for c in result["clips"])
    assert any("teaser" in w for w in result["warnings"])


def test_prompt_echo_filter():
    names = ["Anish Ahuja", "Imprint Studio", "JLT"]
    assert cs.is_prompt_echo("Anish Ahuja, Imprint Studio, JLT.", names)
    assert not cs.is_prompt_echo("Welcome to Imprint Studio in JLT, I'm Anish Ahuja.", names)


def test_title_is_not_a_tiny_filler_sentence():
    segs = make_segments(["I don't even know."] + COOKING * 4)
    assert cs.draft_title(segs) != "I don't even know."


def test_clips_spread_across_topic_sections():
    segs = cs.load_transcript(SAMPLE)
    bounds = cs.detect_topic_changes(segs)
    clips = cs.suggest(segs)["clips"]
    sections = [sum(1 for b in bounds if b <= c.start) for c in clips]
    assert len(set(sections)) >= min(len(clips), len(bounds) + 1)


def test_title_options_skip_pronoun_starts():
    segs = cs.load_transcript(SAMPLE)
    titles = cs.draft_titles(segs)
    assert 1 <= len(titles) <= 3 and len(set(titles)) == len(titles)
    full = cs.full_text(segs)
    for t in titles:
        assert cs.first_word(t) not in cs.PRONOUN_STARTS and t in full


def test_numbers_count_as_specific_detail():
    segs = make_segments(["About 90% of people don't book on price, and 15 minutes adds up."] * 8)
    clip = cs.score_window(cs.candidate_windows(segs)[0], segs, [])
    assert clip.parts["specific"] > 0 and "90%" in clip.signals["numbers"]
