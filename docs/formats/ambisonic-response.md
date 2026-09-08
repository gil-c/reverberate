# The ambisonic response on disk

Status: current. Implemented by `reverberate.spatial`. Decided in
[ADR 0008](../adr/0008-ambisonic-intermediate-and-binaural-decoding.md).

The point receiver format in
[`impulse-response.md`](impulse-response.md) is unchanged and still applies to
loose receivers in a room. This document covers the other case: one expansion
about one point, whose channels are not positions.

## Conventions, stated once because each has a silent opposite

| quantity | choice | what the other choice does |
| --- | --- | --- |
| channel order | ACN, `n^2 + n + m` | reorders channels |
| normalisation | N3D internally, SN3D on ambiX export | a per order gain |
| Condon-Shortley phase | **absent** | flips odd `m` |
| frame | `x` front, `y` left, `z` up | mirrors the room |
| transform | `numpy.fft.rfft`, so a delay is `e^{-i omega tau}` | |
| outgoing wave | `e^{-ikr} / r`, Hankel of the **second** kind | flips odd orders |
| plane wave | `b_nm = Y_nm(s)`, so a unit plane wave has `b_00 = 1` | |
| yaw | rotates the **field**, counter clockwise from above | turns the head the wrong way |

The scene is Y up with positions `(x, height, z)`. The frame change to the
ambisonic frame is `reverberate.response.to_sofa_coordinates` and no other
function, because two rotations is how a dataset acquires a mirrored axis.

The check that catches all of them at once is the direction of arrival of the
windowed direct sound against the geometry, `spatial.validate`.

## Files

| file | what it is | who reads it |
| --- | --- | --- |
| `ambisonic.wav` | ambiX: ACN, SN3D, one channel per coefficient | any ambisonic tool |
| `ambisonic.sofa` | AES69 `SingleRoomSRIR`, receiver type `spherical harmonics` | anyone, with the provenance |
| `binaural.sofa` | `SingleRoomSRIR`, two receivers, one measurement per head yaw | a listening test or a training pipeline |
| `binaural_*.wav` | the anechoic clip decoded through the head | a person |

One measurement, not one per channel: an ambisonic array **is** one rigid
array, which is the opposite of ADR 0005's loose points, and AES69 has a
receiver type that says exactly that. `ReceiverDescriptions` carries the ACN
index with its degree and order, so a reader who has only the file can tell
which channel is which.

The binaural file's head orientation travels as `ListenerView` rather than in
a file name, and its two receivers are named `left ear` and `right ear`.

## Levels

**One gain per file, applied to every channel, and recorded.** The direction in
an ambisonic signal is carried entirely by the ratios between its channels, so
a per channel normalisation would destroy the thing the format exists for. This
is the same rule `reverberate.audio.write_wav` already enforces between
receivers, for the same reason.

## What has to travel with the response

- **The effective order per frequency**, measured on the array that recorded
  it, not the nominal order. Below about 500 Hz the high orders are physically
  absent and the report says where.
- **The air absorption parameters**, temperature and relative humidity, or the
  statement that none was applied. At 16 kHz the coefficient varies by a factor
  of 1.9 across ordinary indoor humidity.
- **The head**, and how it was decoded: the model, the order, the magnitude
  least squares cut-on, and whether the covariance constraint was applied. A
  binaural rendering that does not say how it was decoded cannot be compared
  with another.
