from app.training.dataset_loader import _parse_hf_dataset_id


def test_parse_hf_dataset_id_colon():
    assert _parse_hf_dataset_id("Salesforce/wikitext:wikitext-103-raw-v1") == (
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
    )


def test_parse_hf_dataset_id_plain():
    assert _parse_hf_dataset_id("imdb") == ("imdb", None)
