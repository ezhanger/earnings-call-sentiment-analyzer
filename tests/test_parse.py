from ecsa.parse.transcript import clean_speaker, tag_turns, to_sentences


def roles(turns):
    t = tag_turns(turns)
    return dict(zip(t.speaker_name, t.speaker_role)), t


# Modern operator-led call, modelled on META 2019 Q4.
MODERN = [
    {"speaker": "Operator", "text": "Welcome. After the remarks there will be a question-and-answer session."},
    {"speaker": "Deborah Crawford", "text": "Joining me today are Mark Zuckerberg, CEO; Sheryl Sandberg, COO; and Dave Wehner, CFO."},
    {"speaker": "Mark Zuckerberg", "text": "We had a strong quarter. We expect growth to continue."},
    {"speaker": "Dave Wehner", "text": "Revenue grew 25%. Our outlook is unchanged."},
    {"speaker": "Operator", "text": "Your first question comes from the line of Brian Nowak from Morgan Stanley."},
    {"speaker": "Brian Nowak", "text": "Thanks. How are ads trending?"},
    {"speaker": "Sheryl Sandberg", "text": "Ads are trending well."},
]


def test_modern_call():
    r, t = roles(MODERN)
    assert r == {"Operator": "operator", "Deborah Crawford": "other_exec", "Mark Zuckerberg": "ceo",
                 "Dave Wehner": "cfo", "Brian Nowak": "analyst", "Sheryl Sandberg": "other_exec"}
    assert t.section.tolist() == ["prepared"] * 4 + ["qa"] * 3


# Older format with participant headers, titles in labels and Q/A prefixes, modelled on NVDA 2006.
OLD = [
    {"speaker": "Executives", "text": "Jen-Hsun Huang - CEO; Marvin Burkett - CFO"},
    {"speaker": "Analysts", "text": "Mark Edelstone - Morgan Stanley"},
    {"speaker": "Operator", "text": "Good afternoon."},
    {"speaker": "Jen-Hsun Huang, President and Chief Executive Officer", "text": "Record revenue."},
    {"speaker": "Marvin Burkett, Chief Financial Officer", "text": "Margins rose."},
    {"speaker": "Q - Mark Edelstone", "text": "What about inventory?"},
    {"speaker": "A - Marvin D. Burkett", "text": "Inventory is fine."},
]


def test_old_format():
    r, t = roles(OLD)
    assert r["Jen-Hsun Huang"] == "ceo"
    assert r["Marvin Burkett"] == "cfo"
    assert r["Mark Edelstone"] == "analyst"
    assert "Executives" not in r
    assert t.section.tolist() == ["prepared"] * 3 + ["qa"] * 2


# Host-led Q&A with no operator, title before the name and a broken paragraph label, modelled on
# TSLA 2023 Q4 and GOOGL 2023 Q4.
HOSTED = [
    {"speaker": "Martin Viecha", "text": "Welcome to the Q&A webcast. I'm joined by Elon and our CFO, Vaibhav."},
    {"speaker": "Elon Musk", "text": "Record deliveries."},
    {"speaker": "Vaibhav Taneja - Tesla, Inc.", "text": "Margins held."},
    {"speaker": "Moving on to energy. We're", "text": "seeing record storage deployments."},
    {"speaker": "Martin Viecha", "text": "Let's go to analyst questions. The first question is from Adam Jonas from Morgan Stanley."},
    {"speaker": "Adam Jonas", "text": "How is the ramp?"},
    {"speaker": "Elon Musk", "text": "The ramp is hard."},
]


def test_hosted_call():
    r, t = roles(HOSTED)
    assert r["Vaibhav Taneja"] == "cfo"
    assert r["Adam Jonas"] == "analyst"
    assert r["Elon Musk"] == "other_exec"  # never titled in-call; build fills it from other calls
    assert "Moving on to energy. We're" not in r
    assert "storage deployments" in t.text.iloc[2]
    assert t.section.tolist() == ["prepared"] * 4 + ["qa"] * 2


def test_clean_speaker():
    assert clean_speaker("A - Marvin D. Burkett") == "Marvin D. Burkett"
    assert clean_speaker("Sheryl Kara Sandberg - Facebook, Inc.") == "Sheryl Kara Sandberg"


def test_sentences_drop_operator_and_flag_forward_looking():
    s = to_sentences("META_2019Q4", MODERN)
    assert "operator" not in set(s.speaker_role)
    assert s.sentence_idx.tolist() == list(range(len(s)))
    fwd = dict(zip(s.text, s.is_forward_looking))
    assert fwd["We expect growth to continue."] and fwd["Our outlook is unchanged."]
    assert not fwd["Revenue grew 25%."]
