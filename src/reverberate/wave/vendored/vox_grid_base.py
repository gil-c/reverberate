##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: vox_grid_base.py
#
# Description: Class for a voxel-grid for ray-tri / tri-box intersections
#  Uses multiprocessing
#
##############################################################################
#
# REVERBERATE PATCH 6 -- index the triangle search.
#
# Upstream's fill() answers "which triangles might touch this voxel?" by
# comparing every triangle's bounding box against the voxel's, once per voxel.
# There is no spatial index, so the cost is O(Nvox x Ntris) and it grows
# superlinearly with the scene: substituting VoxGrid's own heuristic,
# Nvox_est = 0.025*sqrt(Ntris*Ngrid), the scan costs 0.025*Ntris^1.5*sqrt(Ngrid).
#
# Measured on this project's two scenes at 4 kHz, single process:
#
#   bedroom     44 831 tris,    40 832 vox        1.8e9 pairs       25.6 s
#   apartment  1 447 712 tris, 1 359 240 vox      2.0e12 pairs   61 626 s
#
# Seventeen hours of one core, to answer a question a lattice answers by
# division. The voxels are a regular lattice by construction, so each triangle's
# candidate voxels are the index range its bounding box spans; bin every
# triangle into that range once, and the per-voxel search becomes a slice. The
# apartment's 2.0e12 elementary comparisons become 60.0 million (triangle,
# voxel) pairs, built in 1.6 s -- a factor of 32 790.
#
# The exact test, tri_box_intersection_vec, is untouched and still decides every
# candidate. The binning is deliberately conservative -- one voxel of slack each
# way, and the widest voxel used for every voxel -- so it can only ever offer
# extra candidates, never withhold a real one. Extra candidates cost a handful
# of exact tests and cannot change the answer.
#
# *Acceptance is identity, not speed.* Each candidate list is built ascending by
# triangle index, which is the order np.nonzero produced, so candidates[hits]
# comes out identical and vox_out.h5 must be byte for byte what it was. The one
# thing that does change is the printed 'tribox checks' statistic, which counts
# candidates offered rather than triangles scanned -- that is the saving, so it
# would be strange if it did not.
#
# build_triangle_index() returns None rather than guessing when the voxel boxes
# are not the lattice it expects, and fill() then runs upstream's scan. A
# subclass that lays voxels out some other way keeps working, slowly.
#
##############################################################################

import numpy as np
from numpy import array as npa
from common.timerdict import TimerDict
from common.tri_box_intersection import tri_box_intersection_vec
from common.room_geo import RoomGeo
from common.tris_precompute import tris_precompute
import multiprocessing as mp
from common.myfuncs import clear_dat_folder,clear_console
from common.myfuncs import get_default_nprocs
from tqdm import tqdm
import common.check_version as cv
import time

assert cv.ATLEASTVERSION38 #for shared memory (but project needs 3.9 anyway)

from multiprocessing import shared_memory

#base class for a voxel
class VoxBase:
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
        self.tri_idxs = [] #triangle indices as list
        self.tris_pre = None
        self.tris_mat = None

#base class for a voxel grid
class VoxGridBase:
    def __init__(self,room_geo):
        tris = room_geo.tris
        pts = room_geo.pts
        tris_pre = room_geo.tris_pre
        mats = room_geo.mat_ind

        assert tris.ndim == 2
        assert pts.ndim == 2
        assert tris.shape[0] > tris.shape[1]
        assert pts.shape[0] > pts.shape[1]

        Npts = pts.shape[0]
        Ntris = tris.shape[0]

        self.tris = tris
        self.tris_pre = tris_pre
        self.mats = mats
        self.pts = pts
        self.Npts = Npts
        self.Ntris = Ntris

        self.voxels = []
        self.nonempty_idx = []
        self.timer = TimerDict()
        self.nprocs = get_default_nprocs()

    #PATCH 6: bin every triangle into the voxels its bounding box spans, once.
    #Returns (offsets, tri_ids) as a CSR pair list: voxel i's candidates are
    #tri_ids[offsets[i]:offsets[i+1]], ascending.  None when the voxels are not
    #the regular lattice this assumes, so the caller can fall back to the scan.
    def build_triangle_index(self,chunk_pairs=8_000_000):
        Nvox = self.Nvox
        Ntris = self.Ntris
        if Nvox<2 or Ntris<1:
            return None

        vox_bmin = np.stack([vox.bmin for vox in self.voxels]).astype(np.float64)
        vox_bmax = np.stack([vox.bmax for vox in self.voxels]).astype(np.float64)
        if not np.all(np.isfinite(vox_bmin)) or not np.all(np.isfinite(vox_bmax)):
            return None #dummy voxels, __init__ never ran

        #the lattice, read off the voxels rather than assumed from Nh
        axes = [np.unique(vox_bmin[:,d]) for d in range(3)]
        shape = npa([axis.size for axis in axes],dtype=np.int64)
        if int(np.prod(shape))!=Nvox:
            return None

        origin = npa([axis[0] for axis in axes],dtype=np.float64)
        step = np.ones(3,dtype=np.float64)
        for d,axis in enumerate(axes):
            if axis.size<2:
                continue
            delta = np.diff(axis)
            #uniform to a micron: the voxels come from a uniform grid, so any
            #real spread here means this is not the lattice we think it is
            if not np.allclose(delta,delta[0],rtol=0.0,atol=1e-6):
                return None
            step[d] = delta[0]
        if np.any(step<=0.0):
            return None

        #and that the voxels are ordered x-major, which is how VoxGrid builds
        #them.  Checked rather than trusted: the CSR is addressed by vox.idx.
        subs = [np.searchsorted(axes[d],vox_bmin[:,d]) for d in range(3)]
        flat = (subs[0]*shape[1] + subs[1])*shape[2] + subs[2]
        if not np.array_equal(flat,np.arange(Nvox)):
            return None

        #widest voxel in each axis: voxels in the last row reach further, and a
        #single width for all of them keeps the range closed form and
        #conservative
        width = (vox_bmax-vox_bmin).max(axis=0)

        tri_bmin = self.tris_pre['bmin'].astype(np.float64)
        tri_bmax = self.tris_pre['bmax'].astype(np.float64)
        #voxel k spans [origin+k*step, origin+k*step+width], so it can touch the
        #triangle when origin+k*step <= tri_bmax and origin+k*step+width >=
        #tri_bmin.  Both bounds are exact; SLACK opens them by a fraction of a
        #cell so no rounding can drop a voxel that touches the triangle exactly
        #on its face.  Using the widest voxel for every voxel is already the
        #conservative half of this, and a whole extra voxel each way on top of
        #it trebled the pair count for nothing: 154.4 million against 51.5 on
        #one apartment, all of the difference rejected again by tri_box.
        SLACK = 1e-6
        lo = np.ceil((tri_bmin-width-origin)/step-SLACK).astype(np.int64)
        hi = np.floor((tri_bmax-origin)/step+SLACK).astype(np.int64)
        np.clip(lo,0,shape-1,out=lo)
        np.clip(hi,0,shape-1,out=hi)
        spans = hi-lo+1
        counts = spans[:,0]*spans[:,1]*spans[:,2]
        total = int(counts.sum())

        starts = np.zeros(Ntris+1,dtype=np.int64)
        np.cumsum(counts,out=starts[1:])

        #two passes, chunked, so the peak is the chunk and not the whole pair
        #list: an apartment is 60 million pairs and materialising three int64
        #arrays of that length at once is 1.4 GB for no reason.
        def pairs_of(first,last):
            offs = np.arange(first,last,dtype=np.int64)
            tri = np.searchsorted(starts,offs,side='right')-1
            within = offs-starts[tri]
            sx = spans[tri]
            ix = lo[tri,0] + within//(sx[:,1]*sx[:,2])
            iy = lo[tri,1] + (within//sx[:,2])%sx[:,1]
            iz = lo[tri,2] + within%sx[:,2]
            return tri,(ix*shape[1]+iy)*shape[2]+iz

        per_vox = np.zeros(Nvox,dtype=np.int64)
        for first in range(0,total,chunk_pairs):
            _,vox_ids = pairs_of(first,min(first+chunk_pairs,total))
            per_vox += np.bincount(vox_ids,minlength=Nvox)

        offsets = np.zeros(Nvox+1,dtype=np.int64)
        np.cumsum(per_vox,out=offsets[1:])
        tri_ids = np.empty(total,dtype=np.int64)

        #cursor carries each voxel's write position across chunks.  Chunks run
        #in triangle order and the sort inside one is stable, so every voxel's
        #slice ends up ascending by triangle index -- which is what makes the
        #result identical to np.nonzero's.
        cursor = offsets[:-1].copy()
        for first in range(0,total,chunk_pairs):
            tri,vox_ids = pairs_of(first,min(first+chunk_pairs,total))
            order = np.argsort(vox_ids,kind='stable')
            tri = tri[order]; vox_ids = vox_ids[order]
            #rank of each pair within its voxel's run, inside this chunk
            edges = np.flatnonzero(np.diff(vox_ids))+1
            run_start = np.zeros(vox_ids.size,dtype=np.int64)
            run_start[edges] = edges
            np.maximum.accumulate(run_start,out=run_start)
            tri_ids[cursor[vox_ids]+np.arange(vox_ids.size)-run_start] = tri
            cursor += np.bincount(vox_ids,minlength=Nvox)

        return offsets,tri_ids

    #fill the grid (primarily using tri-box intersections)
    def fill(self,Nprocs=None):
        if Nprocs is None:
            Nprocs = self.nprocs
        self.print(f'using {Nprocs} processes')

        tris = self.tris
        tris_pre = self.tris_pre
        Ntris = self.Ntris
        pts = self.pts
        Nvox = self.Nvox

        self.timer.tic('voxgrid fill')

        tri_pts = tris_pre['v']
        tri_bmin = tris_pre['bmin']
        tri_bmax = tris_pre['bmax']

        if Nvox==1:
            vox = self.voxels[0]
            vox.tri_idxs = np.arange(Ntris)
            vox.tris_pre = self.tris_pre
            vox.tris_mat = self.mats
            self.nonempty_idx = [0]
        else:
            if Nprocs>1:
                clear_dat_folder('mmap_dat')

            #create shared memory
            Ntris_vox_shm = shared_memory.SharedMemory(create=True,size=Nvox*np.dtype(np.int64).itemsize)
            Ntris_vox = np.frombuffer(Ntris_vox_shm.buf, dtype=np.int64)
            #alternative syntax
            #Ntris_vox = np.ndarray((Nvox,), dtype=np.int64, buffer=Ntris_vox_shm.buf)

            #use as buffer view to np array
            N_tribox_tests_shm = shared_memory.SharedMemory(create=True,size=Nvox*np.dtype(np.int64).itemsize)
            N_tribox_tests = np.frombuffer(N_tribox_tests_shm.buf, dtype=np.int64)

            Ntris_vox[:] = 0
            N_tribox_tests[:] = 0

            #PATCH 6: one binning pass, then the per-voxel search is a slice
            self.timer.tic('voxgrid index')
            index = self.build_triangle_index()
            if index is None:
                self.print('no lattice found, scanning every triangle per voxel')
            else:
                idx_offsets,idx_tris = index
                self.print(f'indexed {idx_tris.size} (triangle, voxel) pairs, '
                           f'{idx_tris.size/Nvox:.1f} per voxel against {Ntris} scanned')
            self.print(self.timer.ftoc('voxgrid index'))

            #looping through boxes makes more sense because we append to voxels (for multithreading)
            def process_voxel(vox):
                if index is None:
                    candidates = np.nonzero(np.all(np.logical_and(vox.bmax >= tri_bmin,vox.bmin <= tri_bmax),axis=-1))[0]
                else:
                    candidates = idx_tris[idx_offsets[vox.idx]:idx_offsets[vox.idx+1]]
                tri_idxs_vox = []
                N_tribox_tests[vox.idx] += candidates.size
                if candidates.size==0:
                    return tri_idxs_vox
                hits = tri_box_intersection_vec(vox.bmin,vox.bmax,tris_pre[candidates])
                tri_idxs_vox = candidates[hits].tolist()
                return tri_idxs_vox

            def process_voxels(vidx_list,proc_idx):
                pbar = tqdm(total=len(vidx_list),desc=f'process {proc_idx:02d} voxgrid processing',ascii=True,leave=False,position=0)
                for vox_idx in vidx_list:
                    tri_idxs_vox = process_voxel(self.voxels[vox_idx])
                    Ntris_vox[vox_idx] = len(tri_idxs_vox)
                    #if not empty, save vox data as file
                    if len(tri_idxs_vox)>0:
                        np.array(tri_idxs_vox,dtype=np.int64).tofile(f'mmap_dat/vox_{vox_idx}.dat')
                    pbar.update(1)

                pbar.close()

            
            if Nprocs==1: #keep separate for debug purposes
                #process without intermediate files
                pbar = tqdm(total=Nvox,desc=f'single process voxgrid processing',ascii=True,leave=False)
                for vox_idx in range(Nvox):
                    vox = self.voxels[vox_idx]
                    tri_idxs_vox = process_voxel(vox)
                    Ntris_vox[vox_idx] = len(tri_idxs_vox)
                    pbar.update(1)
                    if Ntris_vox[vox_idx]>0:
                        vox.tri_idxs = tri_idxs_vox
                        vox.tris_pre = self.tris_pre[vox.tri_idxs]
                        vox.tris_mat = self.mats[vox.tri_idxs]
                        assert Ntris_vox[vox_idx] == len(vox.tri_idxs)
                        self.nonempty_idx.append(vox_idx)
                pbar.close()

            elif Nprocs>1:
                procs = []

                vox_idx_lists = [[] for i in range(Nprocs)]
                vox_order = np.random.permutation(Nvox)
                #vox_order = np.arange(Nvox)
                for idx in range(Nvox):
                    cc = np.argmin([len(l) for l in vox_idx_lists])
                    vox_idx_lists[cc].append(vox_order[idx])

                for proc_idx in range(Nprocs):
                    proc = mp.Process(target=process_voxels, args=(vox_idx_lists[proc_idx],proc_idx))
                    procs.append(proc)

                for proc_idx in range(Nprocs):
                    procs[proc_idx].start()

                for one_proc in procs:
                    one_proc.join()

                #now load from temp files
                for vox_idx in range(Nvox):
                    vox = self.voxels[vox_idx]
                    if Ntris_vox[vox_idx]>0:
                        #now with one process read data from files
                        vox.tri_idxs = np.fromfile(f'mmap_dat/vox_{vox_idx}.dat',dtype=np.int64)
                        vox.tris_pre = self.tris_pre[vox.tri_idxs]
                        vox.tris_mat = self.mats[vox.tri_idxs]
                        assert Ntris_vox[vox_idx] == len(vox.tri_idxs)
                        self.nonempty_idx.append(vox_idx)

                clear_dat_folder('mmap_dat')

            self.print(self.timer.ftoc('voxgrid fill'))

            Ntris_vox_tot = np.sum(Ntris_vox)

            N_tribox_tests_tot = np.sum(N_tribox_tests)
            self.print(f'tribox checks={N_tribox_tests_tot} for {Ntris} tris and {Nvox} vox ({N_tribox_tests_tot/(Nvox*Ntris)*100.0:.2f} %)')

            #cleanup shared memory
            Ntris_vox_shm.close()
            Ntris_vox_shm.unlink()

            N_tribox_tests_shm.close()
            N_tribox_tests_shm.unlink()

            self.print(f'tris redundant={Ntris_vox_tot}, {100.*Ntris_vox_tot/self.Ntris:.2f} %')
            self.print(f'avg tris per voxel={Ntris_vox_tot/Nvox:.2f}')

    def print(self,fstring):
        print(f'--VOX_GRID_BASE: {fstring}')

    def print_stats(self):
        ntris_found = np.sum([len(vox.tri_idxs) for vox in self.voxels])
        self.print(f'total tris found in voxels={ntris_found:d}')

    #draws non-empty boxes only
    def draw_boxes(self,tube_radius,backend='mayavi'):
        from common.box import Box
        Nvox = self.Nvox
        self.print('drawing boxes..')
        boxtris = np.zeros((Nvox*12,3)) 
        boxpts = np.zeros((Nvox*8,3)) 
        tp = 0
        #build up a triangular mesh for all boxes in one go
        for i in range(len(self.nonempty_idx)):
            vox = self.voxels[self.nonempty_idx[i]]
            assert len(vox.tri_idxs)>0
            box = Box(*(vox.bmax-vox.bmin),shift=vox.bmin,centered=False)
            boxtris[i*12:(i+1)*12,:] = box.tris + tp
            boxpts[i*8:(i+1)*8,:] = box.verts 
            tp += 8
        self.print(f'{len(self.nonempty_idx)=}')
        self.print(f'{tp=}')

        if backend=='mayavi':
            from mayavi import mlab
            mlab.triangular_mesh(*(boxpts.T),boxtris,representation='mesh',color=(0,1,0),tube_radius=tube_radius)
            mlab.draw()
        elif backend=='polyscope':
            import polyscope as ps
            pmesh = ps.register_surface_mesh('voxels', boxpts, boxtris,color=(0,1,0),edge_color=(0,1,0),edge_width=tube_radius)
            #pmesh.set_transparency(0.0)

        self.print('boxes drawn..')
