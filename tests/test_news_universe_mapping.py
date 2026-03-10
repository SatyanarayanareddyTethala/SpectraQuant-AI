import pandas as pd
from spectraquant.news.universe_builder import load_universe_mapping, match_articles_to_universe


def test_mapping_company_and_alias(tmp_path):
    u = tmp_path / "u.csv"
    pd.DataFrame({"symbol": ["TCS.NS"], "company_name": ["Tata Consultancy Services"]}).to_csv(u, index=False)
    a = tmp_path / "a.csv"
    pd.DataFrame({"ticker": ["TCS.NS"], "alias": ["Tata Consulting"]}).to_csv(a, index=False)
    m = load_universe_mapping(str(u), str(a))
    arts = [{"title": "Tata Consulting wins deal", "description": "", "content": "", "source_name": "x", "published_at_utc": "2024-01-01T00:00:00Z"}]
    hits = match_articles_to_universe(arts, m)
    assert hits[0][0]["ticker"] == "TCS.NS"


def test_reject_ambiguous_false_positive(tmp_path):
    u = tmp_path / "u.csv"
    pd.DataFrame({"symbol": ["ABC.NS"], "company_name": ["Alpha Beta"]}).to_csv(u, index=False)
    m = load_universe_mapping(str(u), None)
    arts = [{"title": "Alpha announces policy", "description": "", "content": "", "source_name": "x", "published_at_utc": "2024-01-01T00:00:00Z"}]
    hits = match_articles_to_universe(arts, m)
    assert 0 not in hits
