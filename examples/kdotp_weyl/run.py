#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import asyncio
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import z2pack

import phasemap as pm
import nodefinder as nf

import matplotlib.pyplot as plt

from split import Hamilton_split
from splitting_fct import gap_fct

logging.getLogger("z2pack").setLevel(logging.WARNING)

FEATURE_SIZE = 1e-4


async def phase_func(splitting, loop, executor):
    return await loop.run_in_executor(executor, get_num_nodes, splitting)


def get_num_nodes(splitting):
    print("Calculating splitting:", splitting)
    search_res = nf.search.run(
        partial(gap_fct, splitting=splitting),
        limits=[(-0.5, 0.5)] * 3,
        initial_mesh_size=5,
        refinement_stencil=None,
        feature_size=FEATURE_SIZE,
        gap_threshold=1e-5,
        periodic=False,
        use_fake_potential=True,
        nelder_mead_kwargs={"fprime_cutoff": 100},
    )
    identify_res = nf.identify.run(search_res)
    weyl_count = 0
    for res in identify_res:
        try:
            if abs(get_chern(splitting=splitting, k=res.shape.position)) > 0.5:
                weyl_count += 1
        except AttributeError:
            pass
    return weyl_count


def get_chern(splitting, k):
    H = partial(Hamilton_split, splitting=splitting)
    res = z2pack.surface.run(
        system=z2pack.hm.System(H),
        surface=z2pack.shape.Sphere(k, FEATURE_SIZE),
        pos_tol=1e-3,
        min_neighbour_dist=1e-8,
        iterator=range(50, 401, 8),
    )
    return z2pack.invariant.chern(res)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(max_workers=4)
    res = pm.run(
        lambda pos: phase_func([pos[0], 0, pos[1]], loop=loop, executor=executor),
        limits=[(-0.3, 0.3)] * 2,
        mesh=3,
        num_steps=5,
        save_file="res.json",
        load=True,
    )
    pm.plot.boxes(res)
    plt.savefig("result.pdf", bbox_inches="tight")
