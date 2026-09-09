"""Spatial output: ambisonic encoding of the solver's field, and binaural decoding.

Roadmap section 7 fixes ambisonics as the intermediate format, decoded offline,
because it is the only representation that allows head rotation and a change of
head morphology without resimulating. Until this package existed every receiver
was a bare omnidirectional point and every WAV said so.

The package is pure: numpy and scipy only, no file IO, no network. Every
convention that could be silently wrong is stated once, in :mod:`.sh`, and
locked by a test that would fail on the other choice.
"""
