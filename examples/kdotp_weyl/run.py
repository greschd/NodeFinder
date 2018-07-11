#!/usr/bin/env python

import logging
from functools import partial

import z2pack
logging.getLogger('z2pack').setLevel(logging.WARNING)

import phasemap as pm
import nodefinder as nf
import matplotlib.pyplot as plt
from fsc.async_tools import limit_parallel

from split import Hamilton_split
from splitting_fct import gap_fct

FEATURE_SIZE = 1e-4


@limit_parallel(1)
async def get_num_nodes(splitting):
    print('Calculating splitting:', splitting)
    search_res = await nf.search.run_async(
        partial(gap_fct, splitting=splitting),
        limits=[(-0.5, 0.5)] * 3,
        initial_mesh_size=5,
        refinement_mesh_size=0,
        feature_size=FEATURE_SIZE,
        gap_threshold=1e-5,
        periodic=False,
        use_fake_potential=True,
        nelder_mead_kwargs={'fprime_cutoff': 100}
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
        iterator=range(50, 401, 8)
    )
    return z2pack.invariant.chern(res)


if __name__ == '__main__':
    res = pm.run(
        lambda pos: get_num_nodes([pos[0], 0, pos[1]]),
        limits=[(-0.3, 0.3)] * 2,
        mesh=3,
        num_steps=4,
        save_file='res.json',
        load=True,
    )
    pm.plot.boxes(res)
    plt.savefig('result.pdf', bbox_inches='tight')
