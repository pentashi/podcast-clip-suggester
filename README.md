# Podcast Clip Suggester

A small prototype that turns **one public podcast episode** into a short list of clip
suggestions that a human reviews and approves. Built as a demo for the AI Intern role at
Imprint Studio.

**It only suggests.** Nothing is edited, posted or published. A human approves every clip.

## What it gives you

From one audio file (mp3/wav/m4a) or a ready-made transcript JSON:

1. A full transcript with timestamps (`transcript.txt`, and inside `clips.json`).
2. Five suggested clips, each 30–45 s, non-overlapping and at least 10 s apart, with
   start, end, a one-line "why" and a draft caption.
3. One draft episode title.
4. A short (3-sentence) episode summary.
5. `review.html` (one self-contained page, data embedded, no external scripts, fonts or CDNs)
   and `clips.json` (machine-readable).

Console example (from the fictional sample):

```
Clip 2. Start 2:13. End 2:47. Why: Has story markers ("that was the moment", "next morning"); emotional words ("worst", "alone"); a clear point ("the lesson"); starts on a full sentence, no filler words. Draft caption: The lesson I took from the flood is simple: you are never as alone as you feel on the worst night.
```

## Install

Python 3.10+.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -r requirements.txt          # faster-whisper, scikit-learn, numpy (+ a PyAV pin)
pip install pytest                       # only to run the tests
```

`requirements.txt` pins `av<16`: with PyAV 16+ faster-whisper 1.2.1 fails to open audio
(`unexpected keyword argument 'metadata_errors'`). The first transcription downloads the Whisper model once (`small` is about 460 MB).
After that it makes no network calls. No paid APIs, no API keys.

## Run

```bash
# Real episode (transcribes on CPU, then suggests clips)
python clip_suggester.py episode.mp3 --guest "Guest Name" --out output

# Ready-made transcript JSON: [{"start": 0.0, "end": 4.2, "text": "..."}, ...]
python clip_suggester.py --transcript sample_transcript.json --sample --out sample_output

# Transcript JSON plus the audio file, so the review page can play clips
python clip_suggester.py episode.mp3 --transcript episode.json --out output
```

Then open `output/review.html` in a browser. If the page can't find the audio next to it
(say you moved the folder), use **Load episode audio from this computer**.

Options: `--model` (faster-whisper size, default `small`; `base` is faster, `medium` is more
accurate), `--language en` (skip language detection), `--names "Anish Ahuja, Imprint Studio, JLT"`
(spellings for people, brands and places, given to Whisper as its opening prompt so the
intro gets them right; the `--guest` name is added automatically. Names said later in the
episode still need a human check), `--sample` (adds the
"SAMPLE: fictional text" banner).

Tests: `python -m pytest -q`

## How it works

All of it is in `clip_suggester.py` (about 600 lines). The page is `review_template.html`.

1. **Transcribe.** faster-whisper on CPU, `compute_type="int8"`, `vad_filter=True`,
   word timestamps on, `condition_on_previous_text=False` (otherwise punctuation faded out
   after ~13 minutes on a real episode, which broke sentence splitting). Long Whisper segments are split at sentence ends using the word
   timings, so clips can start and end on sentence boundaries.
2. **Topic changes.** The transcript is cut into ~30 s blocks. At each gap the two blocks
   before are compared with the two blocks after using TF-IDF cosine similarity
   (scikit-learn). Deep dips in similarity (a TextTiling-style "depth score") are marked as
   topic changes: thin ticks on the timeline and dashed lines in the transcript.
3. **Candidate windows.** Every stretch that starts and ends on a segment boundary and lasts
   30–45 s.
   Windows that are mostly lines said again later in the episode (a cold-open teaser that
   previews the interview) are skipped, so clips come from the original moment. Detected by
   4-word overlaps with text at least 60 s later.
4. **Scoring.** Each window gets points for story markers ("I remember", "that's when",
   "one day"...), emotional words, a clear point or takeaway ("the lesson", "my advice",
   "the reason", "rather than", "makes a difference"...), concrete numbers ("90%",
   "15 minutes") and a hook in the opening sentence. Everyday praise words ("love", "best",
   "amazing") don't count: in interviews they are mostly politeness. It loses points for filler
   words (um, uh, you know, like...), long silences, crossing a topic change, starting
   mid-thought (first word "and", "but", "so", "he", "she", "it"... or the previous line
   not finished) and ending mid-sentence.
5. **Pick five.** Best-scoring window first, then the next best that is at least 10 s away
   from the ones already chosen, taking at most one clip per topic section while that is
   possible (so five clips don't all come from one story). If that greedy pass finds fewer than five but five would
   fit, a small dynamic-programming pass picks the set with the most clips and highest
   total score.
6. **Extractive text only.** The caption is the strongest whole sentence inside the clip
   (up to 220 characters), copied word for word. Where Whisper left a long unpunctuated
   run-on, a run of whole transcript lines is used instead and ends in "..." if it stops
   mid-sentence. Three title ideas are short transcript sentences that don't start with a
   pronoun ("they", "it"...); the page shows them as buttons. The summary is one complete
   sentence from each of the early, middle and late part of the episode (an intro line like
   "thank you for joining us" is preferred at the start). The "why" line only quotes phrases that were actually found in
   that clip's text. Nothing is generated or paraphrased.
7. **Review page.** `review_template.html` with the JSON embedded. Timeline, a card per
   clip (play, edit caption, approve), editable title and summary, clickable transcript,
   **Copy approved clips** and **Download approved.json** (approved clips only, with your
   edited captions). Light/dark (follows the system, plus a toggle), keyboard usable,
   works on mobile.

## What was tested

- **Fictional sample** (`sample_transcript.json`, a made-up pottery-studio interview,
  7:15 long, invented timings). Full pipeline run; output is in `sample_output/` and is
  marked "SAMPLE: fictional text".
- **Real transcription, end to end** on a 7:38 audio file made locally with the
  `espeak-ng` text-to-speech engine reading the same fictional script. Checked: transcript
  has timestamps (94 lines), exactly five clips (31–44 s), no overlaps and at least 10 s
  apart, every caption a verbatim substring of the transcript. It took about 60 s on an
  8-core cloud VM with the `small` model (plus the one-time model download).
  Whisper misheard one name ("Rana Vel" for "Rana Vell"), which shows why a human check
  is needed.
- **Real public audio:** the public *Imprint of Success* Season 1 launch trailer (1:18 mp3,
  provided for this test; not included in this repo). Transcribed end to end in about 12 s
  (model already downloaded): 23 timestamped lines, language detected as English (p=0.98).
  Only **2 clips** came out (32 s and 31 s, 12 s apart, captions verbatim), with the
  warning "Only 2 non-overlapping 30-45s clip(s) fit in this 1:15 episode". That is
  expected: five 30 s clips with 10 s gaps need at least about 3.5 minutes, so the
  "exactly five" check needs a full episode.
- **Real full episode:** one public 30-minute *Imprint of Success* episode (mp3 provided
  for this test; no audio, transcript or guest details are in this repo). About 5 minutes
  on an 8-core cloud VM with the `small` model: 547 timestamped lines, **exactly five
  clips** (32-44 s, each from a different topic section), every caption a verbatim
  substring. What this run found and fixed:
  - Punctuation faded out after ~13 minutes, so "sentences" became 60-word run-ons. Fixed
    with `condition_on_previous_text=False`.
  - Passing names as Whisper `hotwords` made Whisper write the names *instead of* real
    speech (about 28 s lost). Names now go in the opening prompt only, and any line that is
    just the prompt repeated is dropped.
  - The episode opens with a 1-minute teaser of lines from later on. Teaser lines are now
    detected and skipped, and the summary drops duplicate sentences.
  - The first draft title was "I don't even know." Titles now need 6+ words, 3+ content
    words and no pronoun start, and the page offers 3 title ideas.
  - Two clips came from the same story; clips are now spread across topic sections.
  - Second round: of 4 moments I shortlisted by hand from the transcript (my picks, not an
    Imprint editor's), the first version found 0 and the current one finds 2. The other 2
    still lose, partly because they cross a (noisy) topic change. Captions are still weak
    where Whisper `small` left no punctuation, and it misheard a book title. `--model
    medium` would help but ran about 10x slower on this VM, so it wasn't used.
  All of this was tuned while looking at this one episode, so it says nothing about how
  the tool does on other episodes. It is a demo run, not an evaluation.
- What the trailer showed (real speech is messier than TTS):
  - Whisper often ends a segment mid-sentence, and sometimes misses a full stop between
    speakers. The first summary version cut sentences at segment ends; it now splits the
    joined transcript (fixed, with a test). One caption can still join two speakers when
    Whisper left out the full stop. The wording is still verbatim, but a human has to check it.
  - "turning point" and "lose everything" were missing from the word lists; I added them
    after seeing this trailer. That's a small bit of tuning on the test file itself, so
    treat the trailer result as a smoke test, not an evaluation.
  - Trailers are montages, so topic detection found nothing (too short for its 30 s blocks).
  - Names may be misspelled by Whisper; check guest and host names by hand.
- **Unit tests** (`tests/`, pytest, 24 tests): topic boundary detection, window length
  limits, non-overlap and 10 s spacing, extractive-only captions/title/summary, the "why"
  line only quoting real text, summaries that keep whole sentences across segment breaks,
  teaser detection, dropping prompt echoes, title quality and title options, number signals, one clip per topic section, transcripts too short for five clips, and clear errors for
  a missing audio file, an empty transcript, a very short episode and missing
  faster-whisper.
- **Review page** checked in headless Chrome at 1280 px and 360 px wide, light and dark:
  no horizontal overflow; timeline bar click scrolls to and focuses the card; "Play this
  clip" starts at the clip start and pauses at the end; transcript timestamps seek the
  audio; copy and download include only approved clips with edited captions.
  Not tested: Safari, Firefox, or a real screen reader.

## Honest limits

- **Rule-based, not a trained model.** The word lists and weights in `clip_suggester.py`
  are my guesses. They were not tuned on real Imprint episodes, and no accuracy has been
  measured. A good clip for an editor is a taste call; this tool only narrows the search.
- **Verbatim captions need human polish.** Spoken sentences rarely read like captions.
- **English only in practice.** Whisper can transcribe other languages, but the word lists
  are English.
- **Topic detection is rough.** It compares words, not meaning, so it misses changes where
  the vocabulary stays the same, and it can't mark changes in the first or last minute.
- **Clip edges follow transcript lines**, so a clip can start a beat early or late. The
  editor sets the final cut.
- **Who is speaking is not detected** (no diarisation), so a clip may switch speakers.

**Suggested next step:** an optional LLM pass that rewrites the captions, title and summary
into natural copy, shown next to the verbatim original, with the same human approval step.
Other ideas: speaker labels, tuning the weights on clips Imprint's editors actually chose,
and loudness/laughter cues from the audio.

## 2-minute screen-recording script

1. **Run it on one episode** (0:00–0:25). Terminal:
   `python clip_suggester.py episode.mp3 --guest "Guest Name"`. Point at the five
   "Clip N. Start ... End ..." lines as they print.
2. **Show the transcript with timestamps** (0:25–0:45). Open `output/review.html`, scroll to
   "Full transcript", click a timestamp to show the audio jumping there.
3. **Show five clips on the timeline** (0:45–1:05). Scroll up to the timeline: five bars with
   start/end times, thin ticks for topic changes.
4. **Click one clip, play it, read the caption** (1:05–1:40). Click a bar, it jumps to the
   card. Press "Play this clip" (it stops at the end), read the "why" line and the draft
   caption, tweak the caption, tick "Approve for editor", press "Download approved.json".
5. **Close** (1:40–2:00). Say: *"This is a prototype. A human approves every clip before
   anything publishes."*

## Files

- `clip_suggester.py` — the whole pipeline and CLI
- `review_template.html` — review page template (data is injected at `/*__DATA__*/null`)
- `requirements.txt` — runtime dependencies
- `sample_transcript.json` — fictional sample transcript (not Imprint content, not a real person)
- `sample_output/` — `review.html`, `clips.json`, `transcript.txt` from the sample
- `tests/` — pytest unit tests

Public content only. Don't run it on private or client recordings without permission.
