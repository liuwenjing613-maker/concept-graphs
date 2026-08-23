from scripts.find_first_replay_divergence import first_divergence


def test_first_divergence_reports_the_earliest_changed_decision():
    reference = [
        {"obs_uid": "a", "natural_match": None},
        {"obs_uid": "b", "natural_match": 0, "applied_match": 0},
        {"obs_uid": "c", "natural_match": 1},
    ]
    replayed = [
        {"obs_uid": "a", "natural_match": None},
        {"obs_uid": "b", "natural_match": 1, "applied_match": 1},
        {"obs_uid": "c", "natural_match": 2},
    ]
    result = first_divergence(reference, replayed)
    assert result["index"] == 1
    assert result["obs_uid"] == "b"
    assert "natural_match" in result["differences"]


def test_equal_traces_have_no_divergence():
    trace = [{"obs_uid": "a", "natural_match": None}]
    assert first_divergence(trace, trace) is None
