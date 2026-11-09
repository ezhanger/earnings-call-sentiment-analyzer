"""Split transcripts into sentences tagged with section, speaker role and forward-looking flag."""

import re

import pandas as pd
import spacy

_nlp = spacy.blank("en")  # tokenizer exceptions keep "Mr." and "Q3." from splitting sentences
_nlp.add_pipe("sentencizer")
_nlp.max_length = 2_000_000

HEADER_SPEAKERS = {"executives", "analysts"}  # participant lists in older transcripts, not speech
QA_START = re.compile(
    r"(first|next) question|question[- ]and[- ]answer|\bQ&A\b|investor questions"
    r"|questions from (our )?(shareholders|investors)"
    r"|open (it |the (call|lines?|floor) )?(up )?(for|to) questions",
    re.IGNORECASE,
)
# Operator/host lines that introduce a questioner ("next question is from Brian Nowak with ...").
INTRO = re.compile(r"questions? (is |comes )?from|line of|turn on your camera", re.IGNORECASE)
QUESTIONER = re.compile(r"^\s*Q\s*[-–—]")
TITLE = re.compile(r"chief executive|\bco-?ceo\b|\bceo\b|chief financial|\bcfo\b", re.IGNORECASE)
FORWARD = re.compile(r"\b(expect\w*|guid(e|ance|ing)|outlook|anticipat\w*|forecast\w*)\b", re.IGNORECASE)


def clean_speaker(raw: str) -> str:
    """'A - Marvin D. Burkett' / 'Jen-Hsun Huang, President and CEO' / 'Name - Company' -> name."""
    return re.split(r",| [-–—] ", re.sub(r"^\s*[QA]\s*[-–—]\s*", "", raw))[0].strip()


def _is_continuation(label: str) -> bool:
    """Speaker labels that are really the start of a paragraph ("Moving on to YouTube. We're")."""
    head = clean_speaker(label)
    return len(head.split()) > 4 or bool(re.search(r"\b\w{3,}[.?!]( |$)", head))


def _role(title: str) -> str:
    return "cfo" if re.search(r"financ|cfo", title, re.IGNORECASE) else "ceo"


def _role_from_title(name: str, text: str) -> str | None:
    """CEO/CFO from an apposition right after the name ("Name, President and CEO"), else from
    "our CFO, Name". The apposition ends at the next clause, so "Mozes, and with me is ...,
    CEO" and "Sandberg, COO, and David Wehner, CFO" don't borrow someone else's title."""
    first, last = re.escape(name.split()[0]), re.escape(name.split()[-1])
    for m in re.finditer(rf"\b{last}\b\s*[,(–—-]([^;.\n]{{0,60}})", text):
        apposition = re.split(r",\s*and\b|\bwith\b|\bjoined\b|\bme\b", m.group(1))[0]
        # A title ends at its first "Officer" or C-level acronym: "COO and Dave Wehner, CFO".
        if end := re.search(r"\bofficer\b|\bc[a-z]o\b", apposition, re.IGNORECASE):
            apposition = apposition[: end.end()]
        if t := TITLE.search(apposition):
            return _role(t.group(0))
    if m := re.search(rf"({TITLE.pattern})[ ,:]{{1,3}}({first}|{last})\b", text, re.IGNORECASE):
        return _role(m.group(1))
    return None


def tag_turns(turns: list[dict]) -> pd.DataFrame:
    """One row per spoken turn: speaker_name, speaker_role, section, text."""
    headers = {t["speaker"].strip().lower(): t["text"] for t in turns
               if t["speaker"].strip().lower() in HEADER_SPEAKERS}
    rows, in_qa, qa_next, seen_speaker = [], False, False, False
    for t in turns:
        if t["speaker"].strip().lower() in HEADER_SPEAKERS:
            continue
        if rows and _is_continuation(t["speaker"]):
            rows[-1]["text"] += " " + t["speaker"] + " " + t["text"]
            continue
        is_op = t["speaker"].strip().lower().startswith("operator")
        # Q&A starts at an operator cue, a "Q - " speaker, or the turn after a host's cue.
        if not in_qa and (qa_next or (seen_speaker and (
                (is_op and QA_START.search(t["text"])) or QUESTIONER.match(t["speaker"])))):
            in_qa = True
        qa_next = seen_speaker and not in_qa and bool(QA_START.search(t["text"]))
        seen_speaker |= not is_op
        rows.append({"raw": t["speaker"], "is_operator": is_op, "text": t["text"],
                     "speaker_name": "Operator" if is_op else clean_speaker(t["speaker"]),
                     "section": "qa" if in_qa else "prepared"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Titles appear in intros ("Mark Zuckerberg, CEO"), header lists and old-style speaker labels.
    # Newline-joined: title patterns stop at "\n", so one label never reads into the next turn.
    title_text = "\n".join([*headers.values(), *df["raw"], *df.loc[~df.is_operator, "text"]])
    # Analysts speak only in Q&A and are introduced (by last name in operator Q&A lines; by first
    # or last name in any "next question is from Mariana with ..." line), listed under
    # "Analysts", or labelled "Q - ".
    qa = df[df.section.eq("qa")]
    intros = " ".join(s for s in re.split(r"(?<=[.?!])\s+", " ".join(df.text)) if INTRO.search(s))
    operator_text = " ".join([*qa.loc[qa.is_operator, "text"], headers.get("analysts", "")])
    roles = {}
    for name, g in df[~df.is_operator].groupby("speaker_name"):
        words = name.split()
        first, last = (re.escape(words[0]), re.escape(words[-1])) if words else ("", "")
        introduced = bool(words) and g.section.eq("qa").all() and bool(
            re.search(rf"\b{last}\b", operator_text) or re.search(rf"\b({first}|{last})\b", intros))
        if g["raw"].str.match(QUESTIONER).any() or introduced or "analyst" in name.lower():
            roles[name] = "analyst"
        else:
            roles[name] = (words and _role_from_title(name, title_text)) or "other_exec"
    df["speaker_role"] = ["operator" if op else roles[n]
                          for n, op in zip(df.speaker_name, df.is_operator)]
    return df[["speaker_name", "speaker_role", "section", "text"]]


def to_sentences(call_id: str, turns: list[dict]) -> pd.DataFrame:
    """Sentence rows for one call, operator lines dropped (design doc, Stage 1)."""
    tagged = tag_turns(turns)
    if tagged.empty:
        return pd.DataFrame()
    rows = []
    for t in tagged[tagged.speaker_role.ne("operator")].itertuples():
        for s in _nlp(t.text).sents:
            text = s.text.strip()
            if len(text) > 1:
                rows.append({"call_id": call_id, "text": text, "section": t.section,
                             "speaker_name": t.speaker_name, "speaker_role": t.speaker_role,
                             "is_forward_looking": bool(FORWARD.search(text))})
    if not rows:  # operator-only transcript
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.insert(1, "sentence_idx", range(len(out)))
    return out


def propagate_roles(sentences: pd.DataFrame, call_dates: pd.Series, years: int = 2) -> pd.Series:
    """Fill untitled execs with their CEO/CFO role from the nearest titled call of the same
    company within `years`. call_dates is indexed by call_id; call_id starts with the ticker."""
    s = sentences[["call_id", "speaker_name", "speaker_role"]].drop_duplicates().copy()
    s["ticker"] = s.call_id.str.split("_").str[0]
    s["date"] = s.call_id.map(call_dates)
    titled = s[s.speaker_role.isin(["ceo", "cfo"])]
    fill = {}
    for r in s[s.speaker_role.eq("other_exec")].itertuples():
        c = titled[(titled.ticker == r.ticker) & (titled.speaker_name == r.speaker_name)]
        gap = (c.date - r.date).abs()
        c = c[gap <= pd.Timedelta(days=365 * years)]
        if len(c):
            fill[(r.call_id, r.speaker_name)] = c.loc[gap[c.index].idxmin(), "speaker_role"]
    keys = list(zip(sentences.call_id, sentences.speaker_name))
    return pd.Series([fill.get(k, role) for k, role in zip(keys, sentences.speaker_role)],
                     index=sentences.index)
