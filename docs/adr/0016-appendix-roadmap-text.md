# 0016, appendix: wording proposed for the roadmap

The roadmap is the owner's file and is not versioned. ADR 0016 changes what
several of its sections say; this page proposes the words, section by
section, for the owner to paste, change or refuse. Nothing here is the
roadmap's text: each block is a proposal, and the section titles are the
only thing taken from it.

Two of these sections were already out of step before ADR 0016, since ADR
0008 (order 7) and ADR 0014 (the mirror). They are brought up to date in the
same pass because a reader of section 15 will look for them.

## Header, "What changed"

Add a paragraph:

> **ADR 0016 adds a second deliverable: moving scenes for Clarify.** A scene
> is a recipe (sources on stations and rails, a free listener, twenty
> minutes), traced once on a rented card into a pack and rendered by one
> signal engine, which is also the audit's. The wave solver answers under
> 1 kHz and the mirror of ADR 0014 above. The scope is one scene on
> hssd_0076 until the owner has validated it by ear and every stage is at
> its minimum cost. Sections 1, 1.2, 3, 7.1, 7.3, 13, 14, 15 and 16 change
> with it.

## 1. Mission

Replace the paragraph on the single engine:

> **Two engines, joined at 1 kHz.** A finite difference wave solver under
> the crossover, where a room is its resonances; above it a geometric
> mirror of that solver, calibrated on it and judged against it (ADR 0014).
> The wave solver remains the reference: the mirror is only as good as the
> wave field it was fitted on, and section 9 judges both.

Add to the deliverables:

> and an engine that renders a moving scene from a recipe, for training and
> for its own audit (ADR 0016).

## 1.1 The consumer

Add a fourth requirement:

> - **Movement.** Sources and listener move over sequences of twenty
>   minutes. The signal at the listener is rendered from a recipe; the
>   recipe is what is stored.

## 1.2 The geometric engine is removed

Retitle "1.2 Third party geometric engines are removed" and add at the end:

> **The question reopened, for an engine of the project's own.** ADR 0014
> records a mirror written here, so no licence stands in the way, and
> joined to the wave field in the frequency domain, so the two formalisms
> meet on a measured seam and not on an assumed one. What this section
> removed stays removed.

## 3. Hard constraints

Amend constraint 1:

> 1. **Determinism.** Same seed and same inputs produce byte identical
>    outputs across processes on one kind of device. **Between a processor
>    and a card the outputs agree to 1e-6 of the peak and are not byte
>    identical**; that tolerance is tested, and nothing may rely on more.

## 7.1 The order that physics supports

Replace the closing paragraph on the order by band:

> **Order 7 is what is delivered**, from an array fitted at order 10 per
> band (ADR 0008, ADR 0015). Under 1 kHz order 7 is also what lets a field
> be translated: an expansion holds while `k d` stays near 3.5, `d` being
> the distance from its centre, 0.20 m at 1 kHz, and only inside the ball
> free of surfaces and sources (ADR 0016).

## 7.3 Dataset augmentation

Add under the table:

> **Movement is not an augmentation.** A moving source is a solve per
> position under 1 kHz, 8 cm apart, and a path trace per step above. A
> moving listener is free under 1 kHz, by translation between solved cells,
> and a path trace per step above. Only the head's rotation is free in the
> sense of this table.

## 13. Work queue

Add two items:

> ### W44. How far a solved field can be moved **[DELIVERED]**
>
> Leave-one-out on the 437 points of hssd_0076 and a line of 341 points
> 2 cm apart. Averaging two neighbours fails from 500 Hz at 0.40 m; an
> order 7 expansion translates while `k d` stays near 3.5; two neighbours fused by plane
> waves hold 1 kHz at 0.40 m; a translation fails when a surface or the
> source is nearer than the distance translated. 2.95 USD for the night,
> the line itself 2.19 USD at 1.74 USD/h.
> `data/runs/w44_clarify_interpolation/`.
>
> ### W45. One moving scene, generated and audited **[IN PROGRESS]**
>
> ADR 0016. Recipe, trace, pack, signal engine, audit in the application;
> one twenty minute scene on hssd_0076. **Gate: the owner listens.** Then
> every stage's cost to its minimum, with a ledger per stage. No second
> scene and no dataset before both.

## 14. Budget

Add at the head of the section:

> **This section prices the static dataset of three bands and does not
> price moving scenes.** ADR 0016 sets no dataset budget: its objective is
> the minimum cost of one scene, stage by stage, measured. A budget for
> scenes is written once that ledger exists.

## 15. Explicit non goals

Replace the line on a second solver:

> - **No third solver.** The wave solver and its mirror (ADR 0014) are the
>   two. The wave solver itself may be rewritten where that lowers the cost
>   and reproduces the reference within a stated tolerance (ADR 0016).

Replace the line on real time simulation:

> - **No simulation in real time.** The acoustics of a scene are traced
>   offline. The signal engine that plays a traced scene runs at about the
>   speed of the sound it writes, on a laptop for the audit and on the
>   training card; it computes no acoustics. No GUI in the production chain:
>   the application audits, it does not produce. No game engine.

Keep the line on machine learning and add:

> The signal engine is built so that a training process can call it; the
> training is not here.

## 16. Open questions

Add:

> - **Why a translated field keeps a floor of -20 to -30 dB** at every
>   spacing, 2 cm included. The solver's sub-cell receiver placement is the
>   hypothesis. Not tested.
> - **Whether the mirror's calibration holds seated**, and from a source
>   position other than the one it was fitted on.
> - **What a source position costs under 1 kHz** once only the cells in use
>   are written, and whether that needs the project's own solver.
> - **Whether a voice needs its directivity under 1 kHz.** The wave solve is
>   omnidirectional; a directional one is a monopole and three dipoles, four
>   solves a position.
> - **Whether the low band can be translated next to a near voice**, every
>   measured translation within 0.8 m of the source having failed.

Remove the question on whether order 3 is enough: order 7 is delivered.
