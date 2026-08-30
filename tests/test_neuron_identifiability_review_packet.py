from neurobench.experiments.neuron_identifiability.review_packet import _tsv


def test_review_tsv_uses_tab_delimiter(tmp_path):
    path=tmp_path/"x.tsv"; path.write_text("a\tb\n1\t2\n")
    assert _tsv(path)==[{"a":"1","b":"2"}]
