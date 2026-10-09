"""Unit tests for the cleaning pipeline's normalization and quality filters."""

from collection.common.clean import is_repetitive, normalize_text, passes_filters, script_ratios


def test_nukta_precomposed_and_decomposed_normalize_equal():
    precomposed = "फ़"  # ZA (precomposed)
    decomposed = "य़"[:0] + "फ़"  # nukta letters are composition-excluded
    # Directly build both spellings of qa (ka + nukta) and confirm NFC agrees.
    combining = "क़"  # QA precomposed
    decomposed_form = "क" + "़"  # ka + nukta, decomposed spelling
    assert normalize_text(combining) == normalize_text(decomposed_form)


def test_control_characters_stripped():
    assert normalize_text("नमस्ते\x00\x01दुनिया") == "नमस्तेदुनिया"


def test_stray_zwj_between_spaces_removed():
    text = "शब्द ‍ शब्द"
    assert "‍" not in normalize_text(text)


def test_short_document_dropped():
    ok, reason = passes_filters("बहुत छोटा")
    assert not ok and reason == "short"


def test_devanagari_document_passes():
    text = (
        "यह एक हिन्दी लेख है जो आज की मुख्य घटनाओं के बारे में जानकारी देता है। "
        "शहर में आज सुबह भारी बारिश हुई जिससे यातायात प्रभावित हुआ। "
        "स्थानीय प्रशासन ने नागरिकों को सतर्क रहने की सलाह दी है। "
        "मौसम विभाग ने अगले दो दिनों तक और बारिश होने की संभावना जताई है। "
        "इस बीच बाजार में सब्जियों के दाम में भी बढ़ोतरी देखी गई। "
        "व्यापारियों का कहना है कि आपूर्ति में कमी के कारण ऐसा हो रहा है। "
        "प्रशासन ने आश्वासन दिया है कि जल्द ही स्थिति सामान्य हो जाएगी। "
        "स्कूलों में भी एहतियात के तौर पर छुट्टी की घोषणा की गई है। "
    )
    ok, reason = passes_filters(text)
    assert ok and reason is None


def test_english_document_fails_script_filter():
    text = "This is an English article with no Devanagari content at all. " * 10
    ok, reason = passes_filters(text)
    assert not ok and reason == "script"


def test_script_ratios_pure_devanagari():
    dev, lat = script_ratios("नमस्ते दुनिया")
    assert dev == 1.0 and lat == 0.0


def test_repeated_line_flagged_as_repetitive():
    text = "\n".join(["यह एक सामान्य पंक्ति है।"] * 10)
    assert is_repetitive(text)


def test_repeated_ngram_flagged_as_repetitive():
    # A short repeated phrase interleaved with distinct filler words -- catches the
    # case a per-line check misses (varied lines that each still contain the same
    # boilerplate 3-gram, e.g. a share-button phrase re-inserted after every
    # paragraph). Each repeat cycle here is short enough that the repeated 3-gram
    # itself, not just the surrounding words, dominates the ngram distribution.
    filler = " ".join(["क्लिक करें अभी"] * 20)
    assert is_repetitive(filler)


def test_varied_prose_not_flagged_as_repetitive():
    text = (
        "यह एक हिन्दी लेख है जो आज की मुख्य घटनाओं के बारे में जानकारी देता है। "
        "शहर में आज सुबह भारी बारिश हुई जिससे यातायात प्रभावित हुआ। "
        "स्थानीय प्रशासन ने नागरिकों को सतर्क रहने की सलाह दी है। "
        "मौसम विभाग ने अगले दो दिनों तक और बारिश होने की संभावना जताई है। "
    )
    assert not is_repetitive(text)


def test_document_dropped_for_repetition_before_dedup():
    text = "\n".join(["नमस्ते दुनिया यह एक परीक्षण पंक्ति है।"] * 10)
    ok, reason = passes_filters(text)
    assert not ok and reason == "repetitive"
