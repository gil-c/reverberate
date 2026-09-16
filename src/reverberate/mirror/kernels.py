"""The CUDA sources of the mirror: the paths of the image sources, and the rays.

Both kernels are transcriptions of their numpy twins, operation for
operation, in double precision, compiled under the project's kernel options
(no fused multiply-add). What the twin computes with ``np.cross``,
``einsum`` over three components and ``np.sqrt`` is written out here in the
same order, so a hit found on the card is the hit the twin finds and a
histogram count on the card is the twin's count.

Layouts shared with :mod:`reverberate.mirror.engine`: a triangle is nine
doubles ``a b c``; a uniform grid is an origin, a cell size, a shape and
compressed rows of member indices; the image tree is positions, orders,
parents and sequences of ``width`` facets; a path's points are
``width + 2`` triples.
"""

from __future__ import annotations

__all__ = ["PATHS_KERNEL", "RAYS_KERNEL"]

_COMMON = r"""
#define INF (__longlong_as_double(0x7ff0000000000000LL))

__device__ __forceinline__ double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void cross3(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/* Moller and Trumbore, as the twin's ``ray_triangles``: the segment o -> o + d
   hits the triangle when u, v are inside and t lies in [lo, hi]. */
__device__ __forceinline__ bool segment_hits(
    const double* o, const double* d, const double* tri, double lo, double hi, double* t_out)
{
    const double* v0 = tri;
    double e1[3], e2[3], p[3], tv[3], q[3];
    for (int k = 0; k < 3; ++k) { e1[k] = tri[3 + k] - v0[k]; e2[k] = tri[6 + k] - v0[k]; }
    cross3(d, e2, p);
    double det = dot3(e1, p);
    if (fabs(det) < 1e-12) return false;
    double inv = 1.0 / det;
    for (int k = 0; k < 3; ++k) tv[k] = o[k] - v0[k];
    double u = dot3(tv, p) * inv;
    cross3(tv, e1, q);
    double v = dot3(d, q) * inv;
    double t = dot3(e2, q) * inv;
    if (u < 0.0 || v < 0.0 || u + v > 1.0) return false;
    if (t < lo || t > hi) return false;
    *t_out = t;
    return true;
}

struct Grid {
    double origin[3];
    double cell;
    int shape[3];
};

__device__ __forceinline__ void cell_of(const Grid& g, const double* p, int* c) {
    for (int k = 0; k < 3; ++k) {
        int i = (int)floor((p[k] - g.origin[k]) / g.cell);
        if (i < 0) i = 0;
        if (i > g.shape[k] - 1) i = g.shape[k] - 1;
        c[k] = i;
    }
}

__device__ __forceinline__ long long flat_of(const Grid& g, const int* c) {
    return ((long long)c[0] * g.shape[1] + c[1]) * (long long)g.shape[2] + c[2];
}

/* The twin's ``cells_along``: the cells a segment a -> b passes through, in
   order, handed to ``visit`` one at a time; returns false when the walk
   stopped early because ``visit`` said so. */
template <typename Visit>
__device__ __forceinline__ void walk_cells(const Grid& g, const double* a, const double* b, Visit visit)
{
    double dir[3];
    for (int k = 0; k < 3; ++k) dir[k] = b[k] - a[k];
    int cell[3], last[3];
    cell_of(g, a, cell);
    cell_of(g, b, last);
    if (!visit(flat_of(g, cell), 0.0)) return;
    double length2 = dot3(dir, dir);
    if (length2 == 0.0) return;
    int step[3];
    double t_max[3], t_delta[3];
    for (int k = 0; k < 3; ++k) {
        step[k] = dir[k] > 0.0 ? 1 : (dir[k] < 0.0 ? -1 : 0);
        double boundary = g.origin[k] + (cell[k] + (step[k] > 0 ? 1 : 0)) * g.cell;
        t_max[k] = step[k] != 0 ? (boundary - a[k]) / dir[k] : INF;
        t_delta[k] = step[k] != 0 ? g.cell / fabs(dir[k]) : INF;
    }
    int limit = 3 * (g.shape[0] + g.shape[1] + g.shape[2]);
    for (int n = 0; n < limit; ++n) {
        if (cell[0] == last[0] && cell[1] == last[1] && cell[2] == last[2]) break;
        int axis = 0;
        if (t_max[1] < t_max[axis]) axis = 1;
        if (t_max[2] < t_max[axis]) axis = 2;
        if (t_max[axis] > 1.0) break;
        double entered = t_max[axis];
        cell[axis] += step[axis];
        if (cell[axis] < 0 || cell[axis] >= g.shape[axis]) break;
        t_max[axis] += t_delta[axis];
        if (!visit(flat_of(g, cell), entered)) return;
    }
}
"""

PATHS_KERNEL = (
    _COMMON
    + r"""
/* Whether the segment a -> b, shrunk by epsilon at both ends, is cut by an
   occluder in the cells it crosses. The twin's ``_segment_hits``. */
__device__ bool leg_blocked(
    const Grid& g, const long long* offsets, const int* members, const double* tris,
    const double* a, const double* b, double epsilon)
{
    double d[3];
    for (int k = 0; k < 3; ++k) d[k] = b[k] - a[k];
    double length = sqrt(dot3(d, d));
    if (length <= 2.0 * epsilon) return false;
    double strict = epsilon / length;
    bool blocked = false;
    walk_cells(g, a, b, [&](long long cell, double) {
        long long start = offsets[cell], stop = offsets[cell + 1];
        for (long long m = start; m < stop; ++m) {
            double t;
            if (segment_hits(a, d, tris + (long long)members[m] * 9, strict, 1.0 - strict, &t)) {
                blocked = true;
                return false;
            }
        }
        return true;
    });
    return blocked;
}

extern "C" __global__ void paths(
    const double* __restrict__ tree_pos, const int* __restrict__ tree_order,
    const int* __restrict__ tree_parent, const int* __restrict__ tree_seq, int width,
    const double* __restrict__ facet_normal, const double* __restrict__ facet_offset,
    const int* __restrict__ facet_start, const int* __restrict__ facet_count,
    const double* __restrict__ reflector_tris,
    const double* __restrict__ occ_tris,
    const double* __restrict__ grid_origin, double grid_cell, const int* __restrict__ grid_shape,
    const long long* __restrict__ cell_offsets, const int* __restrict__ cell_members,
    const double* __restrict__ receivers, const double* __restrict__ source,
    const int* __restrict__ pair_image, const int* __restrict__ pair_receiver, int npairs,
    double epsilon,
    unsigned char* __restrict__ valid_out, double* __restrict__ points_out, int write_points)
{
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= npairs) return;
    Grid g;
    for (int k = 0; k < 3; ++k) { g.origin[k] = grid_origin[k]; g.shape[k] = grid_shape[k]; }
    g.cell = grid_cell;
    int image = pair_image[pair];
    const double* receiver = receivers + (long long)pair_receiver[pair] * 3;
    double near[3], far[3], hit[3];
    for (int k = 0; k < 3; ++k) { near[k] = receiver[k]; far[k] = tree_pos[(long long)image * 3 + k]; }
    int current = image;
    int order = tree_order[image];
    bool alive = true;
    double* points = write_points ? points_out + (long long)pair * (width + 2) * 3 : 0;
    if (points) {
        for (int s = 0; s < width + 2; ++s) for (int k = 0; k < 3; ++k) points[s * 3 + k] = receiver[k];
        for (int k = 0; k < 3; ++k) points[k] = source[k];
    }
    for (int step = 0; step < width && alive; ++step) {
        int bounce = order - 1 - step;
        if (bounce < 0) break;
        int facet = tree_seq[(long long)image * width + bounce];
        const double* n = facet_normal + (long long)facet * 3;
        double offset = facet_offset[facet];
        /* The twin's ``_plane_crossing``. */
        double ha = dot3(near, n) - offset;
        double hb = dot3(far, n) - offset;
        double denominator = ha - hb;
        if (fabs(denominator) <= 1e-12) { alive = false; break; }
        double t = ha / denominator;
        if (!(t > 0.0 && t < 1.0)) { alive = false; break; }
        for (int k = 0; k < 3; ++k) hit[k] = near[k] + t * (far[k] - near[k]);
        /* Inside the facet: a probe through the crossing along the normal
           against the facet's own triangles, as the twin does it. */
        double pa[3], pd[3];
        for (int k = 0; k < 3; ++k) { pa[k] = hit[k] - 1e-4 * n[k]; pd[k] = 2e-4 * n[k]; }
        bool inside = false;
        int start = facet_start[facet], count = facet_count[facet];
        for (int i = 0; i < count && !inside; ++i) {
            double tt;
            inside = segment_hits(pa, pd, reflector_tris + (long long)(start + i) * 9, 0.0, 1.0, &tt);
        }
        if (!inside) { alive = false; break; }
        if (leg_blocked(g, cell_offsets, cell_members, occ_tris, near, hit, epsilon)) { alive = false; break; }
        if (points) {
            int slot = order - step;
            for (int k = 0; k < 3; ++k) points[slot * 3 + k] = hit[k];
        }
        for (int k = 0; k < 3; ++k) near[k] = hit[k];
        current = tree_parent[current];
        for (int k = 0; k < 3; ++k) far[k] = tree_pos[(long long)current * 3 + k];
    }
    if (alive && leg_blocked(g, cell_offsets, cell_members, occ_tris, near, far, epsilon)) alive = false;
    valid_out[pair] = alive ? 1 : 0;
}
"""
)

RAYS_KERNEL = (
    _COMMON
    + r"""
#define SCALE 1099511627776.0   /* 2^40, HISTOGRAM_SCALE */

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* The twin's ``hash_uniform``: 53 bits of the hash over 2^53. */
__device__ __forceinline__ double uniform_of(
    unsigned long long seed, unsigned long long ray, unsigned long long bounce, unsigned long long draw)
{
    unsigned long long key = mix64(seed * 0x9E3779B97F4A7C15ULL + ray);
    unsigned long long state = mix64(key ^ (bounce * 0xD6E8FEB86659FD93ULL) ^ (draw * 0xA0761D6478BD642FULL));
    return (double)(state >> 11) / 9007199254740992.0;
}

/* The twin's ``harmonics3``: N3D real harmonics to order 3, ACN order. */
__device__ __forceinline__ void harmonics3(double x, double y, double z, double* out) {
    double s3 = sqrt(3.0), s5 = sqrt(5.0), s7 = sqrt(7.0), s15 = sqrt(15.0);
    out[0] = 1.0;
    out[1] = s3 * y;
    out[2] = s3 * z;
    out[3] = s3 * x;
    out[4] = s15 * x * y;
    out[5] = s15 * y * z;
    out[6] = s5 * 0.5 * (3.0 * z * z - 1.0);
    out[7] = s15 * x * z;
    out[8] = s15 * 0.5 * (x * x - y * y);
    out[9] = s7 * sqrt(5.0 / 8.0) * y * (3.0 * x * x - y * y);
    out[10] = s7 * sqrt(15.0) * x * y * z;
    out[11] = s7 * sqrt(3.0 / 8.0) * y * (5.0 * z * z - 1.0);
    out[12] = s7 * 0.5 * z * (5.0 * z * z - 3.0);
    out[13] = s7 * sqrt(3.0 / 8.0) * x * (5.0 * z * z - 1.0);
    out[14] = s7 * sqrt(15.0) * 0.5 * z * (x * x - y * y);
    out[15] = s7 * sqrt(5.0 / 8.0) * x * (x * x - 3.0 * y * y);
}

extern "C" __global__ void rays(
    unsigned long long seed, int ray_start, int ray_count, int total_rays,
    const double* __restrict__ source, double reach, int bins, double bin_length,
    double receiver_radius, int nrec, const double* __restrict__ receivers,
    const long long* __restrict__ rec_offsets, const int* __restrict__ rec_members,
    const double* __restrict__ occ_tris, const short* __restrict__ tri_label,
    const double* __restrict__ grid_origin, double grid_cell, const int* __restrict__ grid_shape,
    const long long* __restrict__ cell_offsets, const int* __restrict__ cell_members,
    const double* __restrict__ absorption, const double* __restrict__ scattering, int bands,
    int channels, double floor_energy,
    long long* __restrict__ energy_out, long long* __restrict__ moments_out, long long* __restrict__ hits_out)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= ray_count) return;
    int ray = ray_start + local;
    Grid g;
    for (int k = 0; k < 3; ++k) { g.origin[k] = grid_origin[k]; g.shape[k] = grid_shape[k]; }
    g.cell = grid_cell;
    /* The twin's ``ray_directions``: rejection in the cube, draws 0, 1, 2, ... */
    double dir[3] = {0.0, 0.0, 1.0};
    for (int attempt = 0; attempt < 64; ++attempt) {
        int base = 3 * attempt;
        double u = uniform_of(seed, ray, 0, base);
        double v = uniform_of(seed, ray, 0, base + 1);
        double w = uniform_of(seed, ray, 0, base + 2);
        double p[3] = {2.0 * u - 1.0, 2.0 * v - 1.0, 2.0 * w - 1.0};
        double norm2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if (norm2 <= 1.0 && norm2 > 1e-8) {
            double s = sqrt(norm2);
            for (int k = 0; k < 3; ++k) dir[k] = p[k] / s;
            break;
        }
    }
    double pos[3];
    for (int k = 0; k < 3; ++k) pos[k] = source[k];
    double carried[8];
    for (int b = 0; b < bands; ++b) carried[b] = 1.0 / (double)total_rays;
    double travelled = 0.0;
    int last_triangle = -1;
    for (int bounce = 0; bounce < 100000; ++bounce) {
        double remaining = reach - travelled;
        double end[3];
        for (int k = 0; k < 3; ++k) end[k] = pos[k] + dir[k] * remaining;
        /* Nearest hit over the cells along the segment, as the twin's union. */
        double best_t = INF;
        int best_tri = -1;
        double scaled[3] = {dir[0] * remaining, dir[1] * remaining, dir[2] * remaining};
        walk_cells(g, pos, end, [&](long long cell, double entered) {
            /* A hit already found before this cell's entry cannot be bettered. */
            if (best_t <= entered * remaining) return false;
            long long start = cell_offsets[cell], stop = cell_offsets[cell + 1];
            for (long long m = start; m < stop; ++m) {
                int tri = cell_members[m];
                if (tri == last_triangle) continue;
                double t;
                /* The twin tests the unit direction with t > 1e-6 in metres;
                   here the segment parameter is over the whole remaining
                   reach, so the bound scales. */
                if (segment_hits(pos, scaled, occ_tris + (long long)tri * 9, 1e-6 / remaining, INF, &t)) {
                    double metres = t * remaining;
                    if (metres < best_t) { best_t = metres; best_tri = tri; }
                }
            }
            return true;
        });
        double segment = best_t < remaining ? best_t : remaining;
        /* Receivers whose sphere the segment enters: the twin's ``_sphere_crossings``,
           over the receivers listed in the cells the segment crosses, each counted in
           the cell its entry point lies in. */
        double harmonics[16];
        bool have_harmonics = false;
        double seg_end[3];
        for (int k = 0; k < 3; ++k) seg_end[k] = pos[k] + dir[k] * segment;
        walk_cells(g, pos, seg_end, [&](long long cell, double) {
            long long start = rec_offsets[cell], stop = rec_offsets[cell + 1];
            for (long long m = start; m < stop; ++m) {
                int r = rec_members[m];
                const double* c = receivers + (long long)r * 3;
                double rel[3] = {c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]};
                double along = dot3(rel, dir);
                double rel2 = dot3(rel, rel);
                double perp2 = rel2 - along * along;
                bool inside_r = perp2 <= receiver_radius * receiver_radius;
                double half = sqrt(fmax(receiver_radius * receiver_radius - perp2, 0.0));
                double entry = along - half;
                bool starts_inside = rel2 <= receiver_radius * receiver_radius;
                if (starts_inside) entry = 0.0;
                bool crossed = inside_r && entry >= 0.0 && entry <= segment && !(starts_inside && along < 0.0);
                if (!crossed) continue;
                /* Once per segment: in the cell of the entry point only. */
                double e[3] = {pos[0] + dir[0] * entry, pos[1] + dir[1] * entry, pos[2] + dir[2] * entry};
                int ec[3];
                cell_of(g, e, ec);
                if (flat_of(g, ec) != cell) continue;
                int at = (int)((travelled + entry) / bin_length);
                if (at >= bins) continue;
                if (!have_harmonics) {
                    /* Arrival direction, scene frame to ambisonic: (x, y, z) -> (x, -z, y). */
                    harmonics3(-dir[0], dir[2], -dir[1], harmonics);
                    have_harmonics = true;
                }
                long long base = ((long long)r * bins + at) * bands;
                for (int b = 0; b < bands; ++b) {
                    long long count = llrint(carried[b] * SCALE);
                    atomicAdd((unsigned long long*)(energy_out + base + b), (unsigned long long)count);
                    long long mbase = (base + b) * channels;
                    for (int ch = 0; ch < channels; ++ch) {
                        long long w = llrint(carried[b] * harmonics[ch] * SCALE);
                        atomicAdd((unsigned long long*)(moments_out + mbase + ch), (unsigned long long)w);
                    }
                }
                atomicAdd((unsigned long long*)(hits_out + (long long)r * bins + at), 1ULL);
            }
            return true;
        });
        if (best_tri < 0 || travelled + best_t >= reach) break;
        travelled += best_t;
        for (int k = 0; k < 3; ++k) pos[k] = pos[k] + dir[k] * best_t;
        int label = tri_label[best_tri];
        bool any = false;
        for (int b = 0; b < bands; ++b) {
            carried[b] = carried[b] * (1.0 - absorption[label * bands + b]);
            if (carried[b] >= floor_energy) any = true;
        }
        if (!any) break;
        const double* tri = occ_tris + (long long)best_tri * 9;
        double e1[3], e2[3], n[3];
        for (int k = 0; k < 3; ++k) { e1[k] = tri[3 + k] - tri[k]; e2[k] = tri[6 + k] - tri[k]; }
        cross3(e1, e2, n);
        double nn = sqrt(dot3(n, n));
        if (nn < 1e-12) nn = 1e-12;
        for (int k = 0; k < 3; ++k) n[k] = n[k] / nn;
        if (dot3(n, dir) > 0.0) for (int k = 0; k < 3; ++k) n[k] = -n[k];
        double kind = uniform_of(seed, ray, bounce + 1, 0);
        if (kind < scattering[label]) {
            /* The twin's ``_lambert``. */
            double helper[3] = {1.0, 0.0, 0.0};
            if (fabs(n[0]) >= 0.9) { helper[0] = 0.0; helper[1] = 1.0; }
            double t1[3], t2[3];
            cross3(n, helper, t1);
            double t1n = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
            for (int k = 0; k < 3; ++k) t1[k] = t1[k] / t1n;
            cross3(n, t1, t2);
            bool found = false;
            for (int attempt = 0; attempt < 64 && !found; ++attempt) {
                int base = 1 + 2 * attempt;
                double a = 2.0 * uniform_of(seed, ray, bounce + 1, base) - 1.0;
                double bb = 2.0 * uniform_of(seed, ray, bounce + 1, base + 1) - 1.0;
                double r2 = a * a + bb * bb;
                if (r2 <= 1.0) {
                    double z = sqrt(1.0 - r2);
                    for (int k = 0; k < 3; ++k) dir[k] = a * t1[k] + bb * t2[k] + z * n[k];
                    found = true;
                }
            }
            if (!found) for (int k = 0; k < 3; ++k) dir[k] = n[k];
        } else {
            double dn = dot3(dir, n);
            for (int k = 0; k < 3; ++k) dir[k] = dir[k] - 2.0 * dn * n[k];
        }
        last_triangle = best_tri;
    }
}
"""
)
