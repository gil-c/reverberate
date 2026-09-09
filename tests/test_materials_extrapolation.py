def test_the_octaves_under_125_hz_follow_the_label_s_construction_family() -> None:
    from reverberate.materials.extrapolation import extend_low_bands, low_band_family

    assert low_band_family("shell") == "panel"
    assert low_band_family("carpet") == "porous"
    assert low_band_family("fireplace") == "massive"
    assert low_band_family("credenza") == "hold"
    # A light shell rises to the panel floors under 125 Hz; a heavy one keeps its own value.
    assert extend_low_bands("shell", 0.15) == (0.30, 0.25, 0.20)
    assert extend_low_bands("shell", 0.40) == (0.40, 0.40, 0.40)
    # A thin porous layer halves per octave on the way down.
    assert extend_low_bands("carpet", 0.14) == (0.07, 0.035, 0.0175)
    # Massive and unknown labels hold 125 Hz, which is what the tables do.
    assert extend_low_bands("fireplace", 0.01) == (0.01, 0.01, 0.01)
    assert extend_low_bands("credenza", 0.15) == (0.15, 0.15, 0.15)
