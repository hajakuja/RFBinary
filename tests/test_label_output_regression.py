from rfbd.labels import ID_TO_LABEL


def test_inference_label_strings_are_exact():
    assert ID_TO_LABEL[0] == "no_drone"
    assert ID_TO_LABEL[1] == "drone"
