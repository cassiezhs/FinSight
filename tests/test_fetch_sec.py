from data_ingestiion.fetch_sec import _normalize_text, _find_best_section


def test_find_best_section_skips_short_table_of_contents_candidate():
    html = """
    <html><body>
      <p>Item 1A. Risk Factors Item 1B.</p>
      <h1>Part I</h1>
      <h2>Item 1A. Risk Factors</h2>
      <p>Actual risk disclosure starts here and contains enough text to be selected.
      The company faces operational, market, credit, regulatory, cybersecurity,
      liquidity, supplier, concentration, and competitive risks across regions.</p>
      <h2>Item 1B. Unresolved Staff Comments</h2>
    </body></html>
    """
    orig, lower = _normalize_text(html)

    section = _find_best_section(
        orig,
        lower,
        start_label="item 1a",
        end_labels=["item 1b"],
        prefer_after_label="part i",
        min_chars=80,
    )

    assert "Actual risk disclosure starts here" in section
    assert "Unresolved Staff Comments" not in section


def test_find_best_section_returns_empty_when_label_missing():
    orig, lower = _normalize_text("<p>Part I. Business overview only.</p>")

    assert _find_best_section(orig, lower, "item 7", ["item 8"]) == ""

