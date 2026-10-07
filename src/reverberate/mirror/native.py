"""The moving trace's inner loops in C: one source text for a host's cores and for a card.

:mod:`reverberate.mirror.moving` sieves and validates (step, image) pairs on
flat ``numpy`` arrays: every pair pays every test's temporaries, and a block
of a million pairs is a gigabyte of them. The same tests are written here a
pair at a time, in C, leaving at the first test a pair fails. On a host the
text is compiled once by the machine's own compiler and called through
``ctypes``, which lets go of the interpreter's lock for the length of the
call; on a card the same text is compiled by ``cupy`` under the project's
kernel options, a thread a pair (:data:`DEVICE_SOURCE`).

**The arithmetic is the twin's, operation for operation.** Every quantity
that decides whether a path exists is three products and two sums in the
order :func:`reverberate.mirror.ism.dot3` writes them, a division, a square
root or a comparison: each is one correctly rounded operation, the same
bits on any processor, provided none is fused with its neighbour. The text
is therefore compiled with contraction off (``-ffp-contract=off`` on a
host, ``--fmad=false`` on a card), and what the twin takes from a
transcendental function (the tree's beam, an arc sine and an arc cosine) is
not in the text at all: :mod:`reverberate.mirror.moving` asks ``numpy`` for
it, of the few pairs that passed everything else. A leg that grazes a
facet's edge is then decided the same way by the twin, by a host's core and
by a card.

**Nothing is installed.** The shared object is built from this module's own
text into a directory named by the text, the flags and the compiler
(:func:`library`), written whole or not at all, so that the processes of a
machine build it once between them. A machine without a compiler, or with
``REVERBERATE_NO_NATIVE`` set, takes the ``numpy`` twin, which the tests
hold this text against.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CARD_VARIABLE",
    "CORE_SOURCE",
    "DEVICE_SOURCE",
    "FLAGS",
    "HOST_SOURCE",
    "Held",
    "available",
    "device_trace_jobs",
    "disabled",
    "library",
    "on_cards",
    "why_not",
]

#: Set to ``1`` for a worker that holds a card to sieve and validate on it. Off until a
#: card has shown the tables equal: the text is the host's, and no card has run it yet.
CARD_VARIABLE = "REVERBERATE_PATHS_ON_CARD"

#: Flags of the host's build. Contraction off: a product and a sum are never one operation.
FLAGS = ("-O2", "-fPIC", "-shared", "-std=c99", "-ffp-contract=off", "-fno-fast-math")

#: The tests themselves: plain C that is also C++, so that a card compiles the same lines.
#: ``RV_FN`` is what a function is declared with and ``RV_INF`` an infinity, both given by
#: the text this one is appended to.
CORE_SOURCE = r"""
typedef struct {
    long long facets;
    const double* normals;        /* [facet, 3] */
    const double* offsets;        /* [facet] */
    const unsigned char* both;    /* [facet] */
    const double* rectangle;      /* [facet, 10] */
    const double* frame;          /* [facet, 9] */
    const long long* shape;       /* [facet, 2] */
    const long long* base;        /* [facet] */
    const long long* bucket_offsets;
    const long long* bucket_members;
    const double* reflector_v0;   /* [triangle, 3] */
    const double* reflector_e1;
    const double* reflector_e2;
    const double* occluder_v0;
    const double* occluder_e1;
    const double* occluder_e2;
    const long long* cell_offsets;
    const long long* cell_members;
    double grid_origin[3];
    long long grid_shape[3];
    double grid_cell;
} RvScene;

typedef struct {
    long long count;
    long long width;
    const double* positions;      /* [image, 3] */
    const int* order;             /* [image] */
    const int* parent;            /* [image] */
    const int* sequence;          /* [image, width] */
    const double* rotation;       /* [image, 9] */
    const double* translation;    /* [image, 3] */
} RvTree;

/* ``ism.dot3``: (a0 b0 + a1 b1) + a2 b2. */
RV_FN double rv_dot(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

RV_FN double rv_norm(const double* a) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

/* ``moving._segment_triangles``: one segment against one triangle. */
RV_FN int rv_segment_triangle(
    const double* o, const double* d, const double* v0, const double* e1, const double* e2,
    double low, double high)
{
    double px = d[1] * e2[2] - d[2] * e2[1];
    double py = d[2] * e2[0] - d[0] * e2[2];
    double pz = d[0] * e2[1] - d[1] * e2[0];
    double det = e1[0] * px + e1[1] * py + e1[2] * pz;
    if (fabs(det) < 1e-12) return 0;
    double inv = 1.0 / det;
    double tx = o[0] - v0[0];
    double ty = o[1] - v0[1];
    double tz = o[2] - v0[2];
    double u = (tx * px + ty * py + tz * pz) * inv;
    double qx = ty * e1[2] - tz * e1[1];
    double qy = tz * e1[0] - tx * e1[2];
    double qz = tx * e1[1] - ty * e1[0];
    double v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
    double t = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv;
    return (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0) && (t >= low) && (t <= high);
}

/* ``moving._inside_facet``: a short probe along the normal against the point's bucket. */
RV_FN int rv_inside_facet(const RvScene* s, const double* point, long long facet) {
    const double* n = s->normals + 3 * facet;
    const double* fr = s->frame + 9 * facet;
    const long long* sh = s->shape + 2 * facet;
    double pu = rv_dot(point, fr + 3);
    double pv = rv_dot(point, fr + 6);
    long long iu = (long long)floor((pu - fr[0]) / fr[2]);
    long long iv = (long long)floor((pv - fr[1]) / fr[2]);
    if (iu < 0) iu = 0;
    if (iu > sh[0] - 1) iu = sh[0] - 1;
    if (iv < 0) iv = 0;
    if (iv > sh[1] - 1) iv = sh[1] - 1;
    long long bucket = s->base[facet] + iu * sh[1] + iv;
    double start[3], direction[3];
    for (int k = 0; k < 3; ++k) {
        start[k] = point[k] - 1e-4 * n[k];
        double stop = point[k] + 1e-4 * n[k];
        direction[k] = stop - start[k];
    }
    for (long long m = s->bucket_offsets[bucket]; m < s->bucket_offsets[bucket + 1]; ++m) {
        long long tri = s->bucket_members[m];
        if (rv_segment_triangle(
                start, direction, s->reflector_v0 + 3 * tri, s->reflector_e1 + 3 * tri,
                s->reflector_e2 + 3 * tri, 0.0, 1.0))
            return 1;
    }
    return 0;
}

/* ``moving._blocked``: the segment a -> b, shrunk by epsilon at both ends, against the
   occluders of the cells it crosses, a cell at a time. */
RV_FN int rv_blocked(const RvScene* s, const double* a, const double* b, double epsilon) {
    double direction[3];
    for (int k = 0; k < 3; ++k) direction[k] = b[k] - a[k];
    double length = rv_norm(direction);
    if (!(length > 2.0 * epsilon)) return 0;
    double strict = epsilon / length;
    double high = 1.0 - strict;
    long long cell[3], last[3], step[3];
    double t_max[3], t_delta[3];
    for (int k = 0; k < 3; ++k) {
        long long top = s->grid_shape[k] - 1;
        long long c = (long long)floor((a[k] - s->grid_origin[k]) / s->grid_cell);
        if (c < 0) c = 0;
        if (c > top) c = top;
        cell[k] = c;
        long long e = (long long)floor((a[k] + direction[k] - s->grid_origin[k]) / s->grid_cell);
        if (e < 0) e = 0;
        if (e > top) e = top;
        last[k] = e;
        step[k] = (direction[k] > 0.0 ? 1 : 0) - (direction[k] < 0.0 ? 1 : 0);
        if (step[k] != 0) {
            double boundary =
                s->grid_origin[k] + (double)(cell[k] + (step[k] > 0 ? 1 : 0)) * s->grid_cell;
            t_max[k] = (boundary - a[k]) / direction[k];
            t_delta[k] = s->grid_cell / fabs(direction[k]);
        } else {
            t_max[k] = RV_INF;
            t_delta[k] = RV_INF;
        }
    }
    long long limit = (s->grid_shape[0] + s->grid_shape[1] + s->grid_shape[2]) * 3 + 1;
    for (long long n = 0; n < limit; ++n) {
        long long flat = (cell[0] * s->grid_shape[1] + cell[1]) * s->grid_shape[2] + cell[2];
        for (long long m = s->cell_offsets[flat]; m < s->cell_offsets[flat + 1]; ++m) {
            long long tri = s->cell_members[m];
            if (rv_segment_triangle(
                    a, direction, s->occluder_v0 + 3 * tri, s->occluder_e1 + 3 * tri,
                    s->occluder_e2 + 3 * tri, strict, high))
                return 1;
        }
        if (cell[0] == last[0] && cell[1] == last[1] && cell[2] == last[2]) return 0;
        int axis = 0;
        if (t_max[1] < t_max[axis]) axis = 1;
        if (t_max[2] < t_max[axis]) axis = 2;
        if (t_max[axis] > 1.0) return 0;
        cell[axis] += step[axis];
        if (cell[axis] < 0 || cell[axis] >= s->grid_shape[axis]) return 0;
        t_max[axis] += t_delta[axis];
    }
    return 0;
}

/* ``moving._walk`` for one pair: whether the unfolded path of ``image`` from ``near`` can
   cross its facets. With ``source`` the images are the step's own, through their affine
   maps; without, they stand where the tree's anchor puts them. */
RV_FN int rv_sieve(
    const RvScene* s, const RvTree* t, long long image, const double* listener,
    const double* source, double margin)
{
    long long current = image;
    double reach = margin;
    double near[3] = {listener[0], listener[1], listener[2]};
    long long order = t->order[image];
    for (long long step = 0; step <= t->width; ++step) {
        long long bounce = order - 1 - step;
        if (bounce < 0) return 1;
        long long facet = t->sequence[image * t->width + bounce];
        double far[3];
        if (source) {
            const double* turn = t->rotation + 9 * current;
            const double* shift = t->translation + 3 * current;
            for (int k = 0; k < 3; ++k)
                far[k] = turn[3 * k] * source[0] + turn[3 * k + 1] * source[1]
                    + turn[3 * k + 2] * source[2] + shift[k];
        } else {
            for (int k = 0; k < 3; ++k) far[k] = t->positions[3 * current + k];
        }
        const double* n = s->normals + 3 * facet;
        double ha = rv_dot(near, n) - s->offsets[facet];
        double hb = rv_dot(far, n) - s->offsets[facet];
        double difference = ha - hb;
        double span = fabs(difference);
        int bounded = span > 1e-9;
        double safe = bounded ? span : 1.0;
        double along_leg = ha / (bounded ? difference : 1.0);
        double leg[3], crossing[3];
        for (int k = 0; k < 3; ++k) {
            leg[k] = far[k] - near[k];
            crossing[k] = near[k] + along_leg * leg[k];
        }
        double along = reach / safe;
        double moved = reach * (1.0 + rv_norm(leg) / safe);
        const double* r = s->rectangle + 10 * facet;
        double u = rv_dot(crossing, r + 4) - r[0];
        double v = rv_dot(crossing, r + 7) - r[1];
        if (bounded
            && !(along_leg > -along && along_leg < 1.0 + along && u >= -moved
                 && u <= r[2] + moved && v >= -moved && v <= r[3] + moved))
            return 0;
        if (bounded) {
            for (int k = 0; k < 3; ++k) near[k] = crossing[k];
            reach = moved > margin ? moved : margin;
        } else {
            /* A crossing nothing bounds says nothing of the next leg either. */
            for (int k = 0; k < 3; ++k) near[k] = far[k];
            reach = RV_INF;
        }
        current = t->parent[current];
    }
    return 1;
}

/* ``moving._validate`` for one pair, all but the tree's beam: the air side of each facet
   and the window of the region where the source stands, the walk back from the listener
   through each facet's triangles, and every leg against the occluders. ``chain`` is
   ``[width + 1, 3]``, the images level by level from the source; ``image`` and ``first``
   are the last of them and the path's first point after the source. */
RV_FN int rv_validate(
    const RvScene* s, const int* sequence, long long order, long long width,
    const double* source, const double* listener,
    int has_region, const double* lo, const double* hi, double reach, double epsilon,
    double* chain, double* image, double* first)
{
    double position[3] = {source[0], source[1], source[2]};
    double points[3 * (RV_WIDTH + 1)];
    for (int k = 0; k < 3; ++k) chain[k] = position[k];
    for (long long level = 0; level < order; ++level) {
        long long facet = sequence[level];
        const double* n = s->normals + 3 * facet;
        double height = rv_dot(position, n) - s->offsets[facet];
        int ok = (height > 0.0) || s->both[facet];
        double twice = 2.0 * height;
        for (int k = 0; k < 3; ++k) position[k] = position[k] - twice * n[k];
        if (has_region) {
            double gap[3];
            for (int k = 0; k < 3; ++k) {
                double below = lo[k] - position[k];
                double above = position[k] - hi[k];
                double g = below > above ? below : above;
                gap[k] = g > 0.0 ? g : 0.0;
            }
            ok = ok && (rv_norm(gap) <= reach);
        }
        if (!ok) return 0;
        for (int k = 0; k < 3; ++k) chain[3 * (level + 1) + k] = position[k];
    }
    for (long long level = order; level < width; ++level)
        for (int k = 0; k < 3; ++k) chain[3 * (level + 1) + k] = position[k];
    for (int k = 0; k < 3; ++k) points[k] = listener[k];
    for (long long step = 0; step < order; ++step) {
        long long facet = sequence[order - 1 - step];
        const double* far = chain + 3 * (order - step);
        const double* start = points + 3 * step;
        const double* n = s->normals + 3 * facet;
        double ha = rv_dot(start, n) - s->offsets[facet];
        double hb = rv_dot(far, n) - s->offsets[facet];
        double denominator = ha - hb;
        if (!(fabs(denominator) > 1e-12)) return 0;
        double t = ha / denominator;
        if (!(t > 0.0 && t < 1.0)) return 0;
        double* hit = points + 3 * (step + 1);
        for (int k = 0; k < 3; ++k) hit[k] = start[k] + t * (far[k] - start[k]);
        if (!rv_inside_facet(s, hit, facet)) return 0;
    }
    for (long long step = 0; step <= order; ++step) {
        const double* end = step == order ? chain : points + 3 * (step + 1);
        if (rv_blocked(s, points + 3 * step, end, epsilon)) return 0;
    }
    for (int k = 0; k < 3; ++k) {
        first[k] = points[3 * order + k];
        image[k] = chain[3 * order + k];
    }
    return 1;
}
"""

#: What a host's compiler is given: the tests, then the loops over jobs and pairs.
HOST_SOURCE = (
    r"""
#include <math.h>
#include <stddef.h>
#define RV_FN static inline
#define RV_INF ((double)INFINITY)
#define RV_WIDTH 16
"""
    + CORE_SOURCE
    + r"""
/* The widest facet sequence the text holds a path of. */
long long rv_width(void) { return RV_WIDTH; }

/* A listener cell's short list: the images of the tree whose unfolded path from ``near``
   can cross its facets from anywhere within ``margin``. Returns how many; ``out`` holds
   them in the tree's order. */
long long rv_short_list(
    const RvScene* s, const RvTree* t, const double* near, double margin, long long* out)
{
    long long kept = 0;
    for (long long image = 0; image < t->count; ++image)
        if (rv_sieve(s, t, image, near, NULL, margin)) out[kept++] = image;
    return kept;
}

/* Jobs ``first`` to ``jobs`` of a block, each a source, a listener and a short list:
   every image of the list sieved and, when it passes, validated. A path is a row of the
   outputs, job by job and within a job in the list's order. Returns the first job not
   done: ``jobs``, or the job whose rows did not fit in ``capacity``, none of them
   written. ``counts`` are the rows written and the pairs the sieve kept. */
long long rv_trace_jobs(
    const RvScene* s, const RvTree* t, long long first, long long jobs,
    const double* source, const double* listener,
    const long long* list_start, const long long* list_stop, const long long* lists,
    double margin, int has_region, const double* lo, const double* hi, double reach,
    double epsilon, long long capacity,
    long long* out_job, long long* out_image, double* out_at, double* out_first,
    double* out_chain, long long* counts)
{
    long long written = 0, sieved = 0;
    long long stride = 3 * (t->width + 1);
    double chain[3 * (RV_WIDTH + 1)], at[3], leaves[3];
    for (long long job = first; job < jobs; ++job) {
        long long written_before = written, sieved_before = sieved;
        const double* from = source + 3 * job;
        const double* ears = listener + 3 * job;
        int full = 0;
        for (long long k = list_start[job]; k < list_stop[job]; ++k) {
            long long image = lists[k];
            if (!rv_sieve(s, t, image, ears, from, margin)) continue;
            ++sieved;
            if (!rv_validate(
                    s, t->sequence + image * t->width, t->order[image], t->width, from, ears,
                    has_region, lo, hi, reach, epsilon, chain, at, leaves))
                continue;
            if (written >= capacity) { full = 1; break; }
            out_job[written] = job;
            out_image[written] = image;
            for (int c = 0; c < 3; ++c) {
                out_at[3 * written + c] = at[c];
                out_first[3 * written + c] = leaves[c];
            }
            for (long long c = 0; c < stride; ++c) out_chain[stride * written + c] = chain[c];
            ++written;
        }
        if (full) {
            counts[0] = written_before;
            counts[1] = sieved_before;
            return job;
        }
    }
    counts[0] = written;
    counts[1] = sieved;
    return jobs;
}

/* Pairs given one by one: a source, a listener, a facet sequence and its order each.
   ``alive`` says which are paths but for the beam; their rows of the outputs are filled. */
void rv_validate_pairs(
    const RvScene* s, long long pairs, long long width,
    const double* source, const double* listener, const int* sequence, const int* order,
    int has_region, const double* lo, const double* hi, double reach, double epsilon,
    unsigned char* alive, double* out_at, double* out_first, double* out_chain)
{
    long long stride = 3 * (width + 1);
    for (long long p = 0; p < pairs; ++p)
        alive[p] = (unsigned char)rv_validate(
            s, sequence + p * width, order[p], width, source + 3 * p, listener + 3 * p,
            has_region, lo, hi, reach, epsilon, out_chain + stride * p, out_at + 3 * p,
            out_first + 3 * p);
}

/* Segments a -> b, each against the occluders. */
void rv_blocked_many(
    const RvScene* s, long long segments, const double* a, const double* b, double epsilon,
    unsigned char* out)
{
    for (long long k = 0; k < segments; ++k)
        out[k] = (unsigned char)rv_blocked(s, a + 3 * k, b + 3 * k, epsilon);
}
"""
)

#: What a card is given: the same tests, and two launches, a thread a pair. The sieve
#: marks the pairs worth validating; the validation fills the rows of those that are
#: paths. Compiled under :data:`reverberate.compute.KERNEL_OPTIONS` (no fused
#: multiply-add), so each test is the host's to the bit.
DEVICE_SOURCE = (
    r"""
#define RV_FN __device__ __forceinline__
#define RV_INF (__longlong_as_double(0x7ff0000000000000LL))
#define RV_WIDTH 16
#ifndef NULL
#define NULL 0
#endif
"""
    + CORE_SOURCE
    + r"""
__device__ __forceinline__ void rv_scene_of(
    RvScene* s, long long facets,
    const double* normals, const double* offsets, const unsigned char* both,
    const double* rectangle, const double* frame, const long long* shape,
    const long long* base, const long long* bucket_offsets, const long long* bucket_members,
    const double* reflector_v0, const double* reflector_e1, const double* reflector_e2,
    const double* occluder_v0, const double* occluder_e1, const double* occluder_e2,
    const long long* cell_offsets, const long long* cell_members,
    const double* grid_origin, const long long* grid_shape, double grid_cell)
{
    s->facets = facets;
    s->normals = normals; s->offsets = offsets; s->both = both;
    s->rectangle = rectangle; s->frame = frame; s->shape = shape; s->base = base;
    s->bucket_offsets = bucket_offsets; s->bucket_members = bucket_members;
    s->reflector_v0 = reflector_v0; s->reflector_e1 = reflector_e1;
    s->reflector_e2 = reflector_e2;
    s->occluder_v0 = occluder_v0; s->occluder_e1 = occluder_e1; s->occluder_e2 = occluder_e2;
    s->cell_offsets = cell_offsets; s->cell_members = cell_members;
    for (int k = 0; k < 3; ++k) {
        s->grid_origin[k] = grid_origin[k];
        s->grid_shape[k] = grid_shape[k];
    }
    s->grid_cell = grid_cell;
}

#define RV_SCENE_ARGS \
    long long facets, \
    const double* normals, const double* offsets, const unsigned char* both, \
    const double* rectangle, const double* frame, const long long* shape, \
    const long long* base, const long long* bucket_offsets, const long long* bucket_members, \
    const double* reflector_v0, const double* reflector_e1, const double* reflector_e2, \
    const double* occluder_v0, const double* occluder_e1, const double* occluder_e2, \
    const long long* cell_offsets, const long long* cell_members, \
    const double* grid_origin, const long long* grid_shape, double grid_cell
#define RV_SCENE_PASS \
    facets, normals, offsets, both, rectangle, frame, shape, base, bucket_offsets, \
    bucket_members, reflector_v0, reflector_e1, reflector_e2, occluder_v0, occluder_e1, \
    occluder_e2, cell_offsets, cell_members, grid_origin, grid_shape, grid_cell

/* A thread a (job, image) pair: whether the sieve keeps it. */
extern "C" __global__ void rv_sieve_pairs(
    RV_SCENE_ARGS,
    long long count, long long width, const double* positions, const int* order,
    const int* parent, const int* sequence, const double* rotation,
    const double* translation,
    const double* source, const double* listener,
    const long long* pair_job, const long long* pair_image, long long pairs, double margin,
    unsigned char* kept)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    RvScene s;
    rv_scene_of(&s, RV_SCENE_PASS);
    RvTree t;
    t.count = count; t.width = width; t.positions = positions; t.order = order;
    t.parent = parent; t.sequence = sequence; t.rotation = rotation;
    t.translation = translation;
    long long job = pair_job[p];
    kept[p] = (unsigned char)rv_sieve(
        &s, &t, pair_image[p], listener + 3 * job, source + 3 * job, margin);
}

/* A thread a pair the sieve kept: whether it is a path but for the beam, and its rows. */
extern "C" __global__ void rv_validate_kept(
    RV_SCENE_ARGS,
    long long width, const int* order, const int* sequence,
    const double* source, const double* listener,
    const long long* pair_job, const long long* pair_image, long long pairs,
    int has_region, const double* lo, const double* hi, double reach, double epsilon,
    unsigned char* alive, double* out_at, double* out_first, double* out_chain)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    RvScene s;
    rv_scene_of(&s, RV_SCENE_PASS);
    long long job = pair_job[p];
    long long image = pair_image[p];
    alive[p] = (unsigned char)rv_validate(
        &s, sequence + image * width, order[image], width, source + 3 * job,
        listener + 3 * job, has_region, lo, hi, reach, epsilon,
        out_chain + 3 * (width + 1) * p, out_at + 3 * p, out_first + 3 * p);
}
"""
)

#: The widest facet sequence the text is compiled for (``RV_WIDTH``).
MAX_WIDTH = 16

_POINTER = ctypes.c_void_p


class _Scene(ctypes.Structure):
    _fields_ = [
        ("facets", ctypes.c_longlong),
        *[
            (name, _POINTER)
            for name in (
                "normals",
                "offsets",
                "both",
                "rectangle",
                "frame",
                "shape",
                "base",
                "bucket_offsets",
                "bucket_members",
                "reflector_v0",
                "reflector_e1",
                "reflector_e2",
                "occluder_v0",
                "occluder_e1",
                "occluder_e2",
                "cell_offsets",
                "cell_members",
            )
        ],
        ("grid_origin", ctypes.c_double * 3),
        ("grid_shape", ctypes.c_longlong * 3),
        ("grid_cell", ctypes.c_double),
    ]


class _Tree(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_longlong),
        ("width", ctypes.c_longlong),
        ("positions", _POINTER),
        ("order", _POINTER),
        ("parent", _POINTER),
        ("sequence", _POINTER),
        ("rotation", _POINTER),
        ("translation", _POINTER),
    ]


#: What the scene's arrays must be for the text to read them: name, type, columns.
SCENE_ARRAYS: tuple[tuple[str, Any, int], ...] = (
    ("normals", np.float64, 3),
    ("offsets", np.float64, 0),
    ("both", np.bool_, 0),
    ("rectangle", np.float64, 10),
    ("frame", np.float64, 9),
    ("shape", np.int64, 2),
    ("base", np.int64, 0),
    ("bucket_offsets", np.int64, 0),
    ("bucket_members", np.int64, 0),
    ("reflector_v0", np.float64, 3),
    ("reflector_e1", np.float64, 3),
    ("reflector_e2", np.float64, 3),
    ("occluder_v0", np.float64, 3),
    ("occluder_e1", np.float64, 3),
    ("occluder_e2", np.float64, 3),
    ("cell_offsets", np.int64, 0),
    ("cell_members", np.int64, 0),
)
TREE_ARRAYS: tuple[tuple[str, Any, int], ...] = (
    ("positions", np.float64, 3),
    ("order", np.int32, 0),
    ("parent", np.int32, 0),
    ("sequence", np.int32, -1),
    ("rotation", np.float64, 9),
    ("translation", np.float64, 3),
)

_state: dict[str, Any] = {"off": 0, "library": None, "why": None, "tried": False}
#: Held by whoever builds the library, the first time it is asked for, and by whoever
#: turns the text off or on. The threads of a process's first trace all ask at once:
#: without it the first said it had tried before it had built, and the others were
#: told there was no library and went on with the ``numpy`` twin (the defect of
#: :mod:`reverberate.render.native`, found there on the CI's Linux).
_FIRST = threading.Lock()


def _as(array: Any, dtype: Any) -> np.ndarray:
    """``array`` as the text reads it: this type, one block of memory. No copy when it is."""
    return np.ascontiguousarray(array, dtype=dtype)


def _address(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def _compiler() -> str | None:
    named = os.environ.get("CC")
    if named and shutil.which(named.split()[0]):
        return named
    for name in ("cc", "gcc", "clang"):
        if shutil.which(name):
            return name
    return None


def _cache_root() -> Path:
    named = os.environ.get("REVERBERATE_NATIVE_CACHE")
    if named:
        return Path(named)
    home = os.environ.get("XDG_CACHE_HOME")
    root = Path(home) if home else Path.home() / ".cache"
    return root / "reverberate" / "native"


def _build() -> ctypes.CDLL:
    """The shared object of :data:`HOST_SOURCE`: found under its name, or built there."""
    compiler = _compiler()
    if compiler is None:
        raise OSError("no C compiler on this machine (cc, gcc, clang, or CC)")
    version = subprocess.run(
        [*compiler.split(), "--version"], capture_output=True, text=True, check=False
    ).stdout
    name = hashlib.sha256(
        "\n".join([HOST_SOURCE, " ".join(FLAGS), compiler, version]).encode()
    ).hexdigest()[:24]
    root = _cache_root()
    target = root / f"moving_{name}.so"
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            source = Path(scratch) / "moving.c"
            source.write_text(HOST_SOURCE)
            built = Path(scratch) / "moving.so"
            done = subprocess.run(
                [*compiler.split(), *FLAGS, str(source), "-o", str(built), "-lm"],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0 or not built.is_file():
                raise OSError(f"{compiler} did not build the moving trace: {done.stderr[-800:]}")
            # Whole or not at all: another process of the machine may be building it too.
            built.replace(target)
    held = ctypes.CDLL(str(target))
    held.rv_width.restype = ctypes.c_longlong
    if int(held.rv_width()) != MAX_WIDTH:
        raise OSError(f"{target} was built for another width than {MAX_WIDTH}")
    held.rv_short_list.restype = ctypes.c_longlong
    held.rv_short_list.argtypes = [
        ctypes.POINTER(_Scene),
        ctypes.POINTER(_Tree),
        _POINTER,
        ctypes.c_double,
        _POINTER,
    ]
    held.rv_trace_jobs.restype = ctypes.c_longlong
    held.rv_trace_jobs.argtypes = [
        ctypes.POINTER(_Scene),
        ctypes.POINTER(_Tree),
        ctypes.c_longlong,
        ctypes.c_longlong,
        *[_POINTER] * 5,
        ctypes.c_double,
        ctypes.c_int,
        _POINTER,
        _POINTER,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_longlong,
        *[_POINTER] * 6,
    ]
    held.rv_validate_pairs.restype = None
    held.rv_validate_pairs.argtypes = [
        ctypes.POINTER(_Scene),
        ctypes.c_longlong,
        ctypes.c_longlong,
        *[_POINTER] * 4,
        ctypes.c_int,
        _POINTER,
        _POINTER,
        ctypes.c_double,
        ctypes.c_double,
        *[_POINTER] * 4,
    ]
    held.rv_blocked_many.restype = None
    held.rv_blocked_many.argtypes = [
        ctypes.POINTER(_Scene),
        ctypes.c_longlong,
        _POINTER,
        _POINTER,
        ctypes.c_double,
        _POINTER,
    ]
    return held


def library() -> ctypes.CDLL | None:
    """The host's build of the text, made at the first call; ``None`` where there is none."""
    if _state["off"] or os.environ.get("REVERBERATE_NO_NATIVE"):
        return None
    if not _state["tried"]:
        with _FIRST:
            if not _state["tried"]:
                try:
                    _state["library"] = _build()
                except OSError as error:
                    _state["why"] = str(error)
                # Said last: until then another thread waits here and does not go on
                # without the library.
                _state["tried"] = True
    found: ctypes.CDLL | None = _state["library"]
    return found


def available() -> bool:
    """Whether the moving trace runs its inner loops in C in this process."""
    return library() is not None


def why_not() -> str | None:
    """Why the text is not in use here, for a log; ``None`` when it is."""
    if os.environ.get("REVERBERATE_NO_NATIVE"):
        return "REVERBERATE_NO_NATIVE is set"
    if _state["off"]:
        return "turned off by the caller"
    library()
    why: str | None = _state["why"]
    return why


@contextlib.contextmanager
def disabled() -> Iterator[None]:
    """The ``numpy`` twin for the length of a block: what the tests hold the text against.

    For the whole process, every thread of it, and counted: blocks nest.
    """
    with _FIRST:
        _state["off"] += 1
    try:
        yield
    finally:
        with _FIRST:
            _state["off"] -= 1


class Held:
    """A scene's arrays, or a tree's, as the text reads them, kept alive beside their struct."""

    def __init__(self, struct: Any, arrays: dict[str, np.ndarray]) -> None:
        self.struct = struct
        self.arrays = arrays

    @property
    def pointer(self) -> Any:
        return ctypes.byref(self.struct)


def scene_of(host: dict[str, np.ndarray]) -> Held:
    """The scene's arrays (:attr:`reverberate.mirror.moving.MovingScene.host`) for the text."""
    arrays = {name: _as(host[name], dtype) for name, dtype, _ in SCENE_ARRAYS}
    struct = _Scene()
    struct.facets = int(arrays["normals"].shape[0])
    for name in arrays:
        setattr(struct, name, _address(arrays[name]))
    for k in range(3):
        struct.grid_origin[k] = float(np.asarray(host["grid_origin"], dtype=float)[k])
        struct.grid_shape[k] = int(np.asarray(host["grid_shape"])[k])
    struct.grid_cell = float(host["grid_cell"])
    return Held(struct, arrays)


def tree_of(fields: dict[str, Any], width: int) -> Held:
    """A candidate tree's arrays for the text; ``sequence`` is ``[image, width]``."""
    if width > MAX_WIDTH:
        raise ValueError(f"a facet sequence of {width} is longer than the text's {MAX_WIDTH}")
    arrays = {name: _as(fields[name], dtype) for name, dtype, _ in TREE_ARRAYS}
    arrays["rotation"] = arrays["rotation"].reshape(-1, 9)
    struct = _Tree()
    struct.count = int(arrays["order"].shape[0])
    struct.width = int(width)
    for name in arrays:
        setattr(struct, name, _address(arrays[name]))
    return Held(struct, arrays)


def short_list(scene: Held, tree: Held, near: np.ndarray, margin: float) -> np.ndarray:
    """:func:`reverberate.mirror.moving._walk` of a whole tree from one listener's anchor."""
    held = library()
    assert held is not None
    near = _as(near, np.float64).reshape(3)
    out = np.empty(int(tree.struct.count), dtype=np.int64)
    kept = held.rv_short_list(scene.pointer, tree.pointer, _address(near), margin, _address(out))
    return out[: int(kept)].copy()


def trace_jobs(
    scene: Held,
    tree: Held,
    source: np.ndarray,
    listener: np.ndarray,
    list_start: np.ndarray,
    list_stop: np.ndarray,
    lists: np.ndarray,
    *,
    margin: float,
    region: tuple[np.ndarray, np.ndarray] | None,
    reach: float,
    epsilon: float,
    capacity: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Every job's short list sieved and validated, all but the beam.

    ``source`` and ``listener`` are ``[job, 3]``; job ``j`` reads images
    ``lists[list_start[j]:list_stop[j]]``. Returns the paths' jobs, their
    images in the tree, the images' positions, the paths' first points
    after the source and their chains ``[path, width + 1, 3]``, job by job
    and within a job in the list's order; and how many pairs the sieve kept.
    """
    held = library()
    assert held is not None
    source = _as(source, np.float64).reshape(-1, 3)
    listener = _as(listener, np.float64).reshape(-1, 3)
    list_start = _as(list_start, np.int64)
    list_stop = _as(list_stop, np.int64)
    lists = _as(lists, np.int64)
    jobs = int(source.shape[0])
    width = int(tree.struct.width)
    lo = _as(np.zeros(3) if region is None else region[0], np.float64).reshape(3)
    hi = _as(np.zeros(3) if region is None else region[1], np.float64).reshape(3)
    parts: list[tuple[np.ndarray, ...]] = []
    counts = np.zeros(2, dtype=np.int64)
    sieved = 0
    first = 0
    while first < jobs:
        out_job = np.empty(capacity, dtype=np.int64)
        out_image = np.empty(capacity, dtype=np.int64)
        out_at = np.empty((capacity, 3))
        out_first = np.empty((capacity, 3))
        out_chain = np.empty((capacity, width + 1, 3))
        reached = int(
            held.rv_trace_jobs(
                scene.pointer,
                tree.pointer,
                first,
                jobs,
                _address(source),
                _address(listener),
                _address(list_start),
                _address(list_stop),
                _address(lists),
                margin,
                0 if region is None else 1,
                _address(lo),
                _address(hi),
                reach,
                epsilon,
                capacity,
                _address(out_job),
                _address(out_image),
                _address(out_at),
                _address(out_first),
                _address(out_chain),
                _address(counts),
            )
        )
        written = int(counts[0])
        sieved += int(counts[1])
        if reached == first:
            # One job's paths alone are more than the rows given: more rows.
            capacity *= 4
            continue
        parts.append(
            (
                out_job[:written],
                out_image[:written],
                out_at[:written],
                out_first[:written],
                out_chain[:written],
            )
        )
        first = reached
    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, width + 1, 3)), 0
    job, image, at, leaves, chain = (np.concatenate(column) for column in zip(*parts, strict=True))
    return job, image, at, leaves, chain, sieved


def on_cards() -> bool:
    """Whether a worker that holds a card runs the text on it (:data:`CARD_VARIABLE`)."""
    return os.environ.get(CARD_VARIABLE, "") not in ("", "0")


def device_trace_jobs(
    xp: Any,
    scene: dict[str, Any],
    tree: dict[str, Any],
    source: np.ndarray,
    listener: np.ndarray,
    list_start: np.ndarray,
    list_stop: np.ndarray,
    lists: np.ndarray,
    *,
    margin: float,
    region: tuple[np.ndarray, np.ndarray] | None,
    reach: float,
    epsilon: float,
    threads: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """:func:`trace_jobs` on a card: one launch sieves every pair, one validates what it kept.

    ``scene`` and ``tree`` are the arrays on the card, typed as
    :data:`SCENE_ARRAYS` and :data:`TREE_ARRAYS` say. The answer is the
    host's, row for row: the same text decides, a thread a pair, and the
    rows come home in the pairs' order.
    """
    from reverberate.compute import raw_kernel, to_numpy

    sieve = raw_kernel(DEVICE_SOURCE, "rv_sieve_pairs")
    validate = raw_kernel(DEVICE_SOURCE, "rv_validate_kept")
    width = int(tree["sequence"].shape[1])
    if width > MAX_WIDTH:
        raise ValueError(f"a facet sequence of {width} is longer than the text's {MAX_WIDTH}")
    counts = np.asarray(list_stop, dtype=np.int64) - np.asarray(list_start, dtype=np.int64)
    jobs = int(counts.shape[0])
    total = int(counts.sum())
    empty = np.zeros(0, dtype=np.int64)
    nothing = (empty, empty, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, width + 1, 3)), 0)
    if total == 0:
        return nothing
    job_host = np.repeat(np.arange(jobs, dtype=np.int64), counts)
    first = np.concatenate([[0], np.cumsum(counts)])[:-1]
    rank = np.arange(total, dtype=np.int64) - np.repeat(first, counts)
    image_host = np.asarray(lists, dtype=np.int64)[np.asarray(list_start)[job_host] + rank]
    pair_job, pair_image = xp.asarray(job_host), xp.asarray(image_host)
    at = xp.asarray(np.ascontiguousarray(source, dtype=np.float64))
    ears = xp.asarray(np.ascontiguousarray(listener, dtype=np.float64))
    given = (
        np.int64(scene["normals"].shape[0]),
        *[scene[name] for name, _, _ in SCENE_ARRAYS],
        scene["grid_origin"],
        scene["grid_shape"],
        np.float64(float(to_numpy(scene["grid_cell"]))),
    )
    kept = xp.zeros(total, dtype=xp.uint8)
    sieve(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *given,
            np.int64(tree["order"].shape[0]),
            np.int64(width),
            tree["positions"],
            tree["order"],
            tree["parent"],
            tree["sequence"],
            tree["rotation"],
            tree["translation"],
            at,
            ears,
            pair_job,
            pair_image,
            np.int64(total),
            np.float64(margin),
            kept,
        ),
    )
    index = xp.flatnonzero(kept)
    sieved = int(index.shape[0])
    if sieved == 0:
        return nothing
    pair_job, pair_image = pair_job[index], pair_image[index]
    lo = xp.asarray(np.ascontiguousarray(np.zeros(3) if region is None else region[0], float))
    hi = xp.asarray(np.ascontiguousarray(np.zeros(3) if region is None else region[1], float))
    alive = xp.zeros(sieved, dtype=xp.uint8)
    out_at = xp.zeros((sieved, 3))
    out_first = xp.zeros((sieved, 3))
    out_chain = xp.zeros((sieved, width + 1, 3))
    validate(
        ((sieved + threads - 1) // threads,),
        (threads,),
        (
            *given,
            np.int64(width),
            tree["order"],
            tree["sequence"],
            at,
            ears,
            pair_job,
            pair_image,
            np.int64(sieved),
            np.int32(0 if region is None else 1),
            lo,
            hi,
            np.float64(reach),
            np.float64(epsilon),
            alive,
            out_at,
            out_first,
            out_chain,
        ),
    )
    paths = xp.flatnonzero(alive)
    return (
        to_numpy(pair_job[paths]).astype(np.int64),
        to_numpy(pair_image[paths]).astype(np.int64),
        to_numpy(out_at[paths]),
        to_numpy(out_first[paths]),
        to_numpy(out_chain[paths]),
        sieved,
    )


def validate_pairs(
    scene: Held,
    source: np.ndarray,
    listener: np.ndarray,
    sequence: np.ndarray,
    order: np.ndarray,
    *,
    region: tuple[np.ndarray, np.ndarray] | None,
    reach: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:func:`reverberate.mirror.moving._validate` of given pairs, all but the beam.

    Returns which pairs are alive, and for those their images, first points
    and chains; the rows of the others are not to be read.
    """
    held = library()
    assert held is not None
    source = _as(source, np.float64).reshape(-1, 3)
    listener = _as(listener, np.float64).reshape(-1, 3)
    sequence = _as(sequence, np.int32)
    order = _as(order, np.int32)
    pairs, width = int(sequence.shape[0]), int(sequence.shape[1])
    if width > MAX_WIDTH:
        raise ValueError(f"a facet sequence of {width} is longer than the text's {MAX_WIDTH}")
    lo = _as(np.zeros(3) if region is None else region[0], np.float64).reshape(3)
    hi = _as(np.zeros(3) if region is None else region[1], np.float64).reshape(3)
    alive = np.zeros(pairs, dtype=np.uint8)
    at = np.zeros((pairs, 3))
    leaves = np.zeros((pairs, 3))
    chain = np.zeros((pairs, width + 1, 3))
    held.rv_validate_pairs(
        scene.pointer,
        pairs,
        width,
        _address(source),
        _address(listener),
        _address(sequence),
        _address(order),
        0 if region is None else 1,
        _address(lo),
        _address(hi),
        reach,
        epsilon,
        _address(alive),
        _address(at),
        _address(leaves),
        _address(chain),
    )
    return alive.astype(bool), at, leaves, chain


def blocked(scene: Held, a: np.ndarray, b: np.ndarray, epsilon: float) -> np.ndarray:
    """:func:`reverberate.mirror.moving._blocked` of segments ``a -> b``."""
    held = library()
    assert held is not None
    a = _as(a, np.float64).reshape(-1, 3)
    b = _as(b, np.float64).reshape(-1, 3)
    out = np.zeros(int(a.shape[0]), dtype=np.uint8)
    held.rv_blocked_many(
        scene.pointer, int(a.shape[0]), _address(a), _address(b), epsilon, _address(out)
    )
    return out.astype(bool)
