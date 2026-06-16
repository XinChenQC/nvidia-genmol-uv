# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import safe as sf
import datamol as dm
from contextlib import suppress
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# Run RDKit slicing in an isolated process so a C++ SIGSEGV in FragmentOnBonds
# can't kill the worker. Use 'forkserver', NOT 'fork': the GPU worker already
# has CUDA initialized, and forking such a process crashes the child at startup
# (PyTorch's pthread_atfork handlers touch the now-invalid CUDA context). The
# forkserver's server is a freshly exec'd Python with no CUDA, and unlike a
# naive 'spawn' it does not re-import the serverless entrypoint. Preload only
# the lightweight RDKit/SAFE deps (no torch) so worker startup stays fast.
try:
    _MP_CTX = multiprocessing.get_context('forkserver')
    _MP_CTX.set_forkserver_preload(['genmol.utils.utils_chem'])
except Exception:
    _MP_CTX = multiprocessing.get_context('spawn')

# https://github.com/datamol-io/safe/blob/main/safe/sample.py
# https://github.com/jensengroup/GB_GA/blob/master/crossover.py
def safe_to_smiles(safe_str, fix=True):
    if fix:
        safe_str = '.'.join([frag for frag in safe_str.split('.')
                             if sf.decode(frag, ignore_errors=True) is not None])
    return sf.decode(safe_str, canonical=True, ignore_errors=True)


def filter_by_substructure(sequences, substruct):
    substruct = sf.utils.standardize_attach(substruct)
    substruct = Chem.DeleteSubstructs(Chem.MolFromSmarts(substruct), Chem.MolFromSmiles('*'))
    substruct = Chem.MolFromSmarts(Chem.MolToSmiles(substruct))
    return sf.utils.filter_by_substructure_constraints(sequences, substruct)


_FAILED = object()


def _run_isolated(fn, *args, timeout=30):
    # Run one crashy RDKit call in its own subprocess. A C++ SIGSEGV there
    # surfaces as BrokenProcessPool; we return _FAILED so the caller can skip
    # just that molecule instead of losing the whole batch (or killing the
    # worker with exit code 139).
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=_MP_CTX) as ex:
            return ex.submit(fn, *args).result(timeout=timeout)
    except Exception:
        return _FAILED


def _slice_one(smiles, query_smarts):
    """Slice one molecule and return its linker as SMILES (or None)."""
    slicer = sf.utils.MolSlicer(require_ring_system=False)
    query = dm.from_smarts(query_smarts)
    out = slicer(dm.to_mol(smiles), query)
    linker = out[1]
    return Chem.MolToSmiles(linker) if linker is not None else None


def _link_one(linker_smiles, prefix, suffix):
    """Link one linker fragment with the two attachment fragments -> SMILES list."""
    slicer = sf.utils.MolSlicer(require_ring_system=False)
    linker = dm.to_mol(linker_smiles)
    return [x for x in slicer.link_fragments(linker, prefix, suffix) if x]


def mix_sequences(prefix_sequences, suffix_sequences, prefix, suffix, num_samples=1):
    # MolSlicer -> FragmentOnBonds can SIGSEGV (uncatchable by try/except) in
    # RDKit's C++ layer on some generated molecules. Each slicing/linking call
    # runs in its own isolated subprocess so a single crashing molecule only
    # drops itself, instead of taking down the whole batch (0 molecules) or the
    # worker (exit code 139). Same algorithm as upstream, just per-call isolated.
    linkers = []
    for x in prefix_sequences:
        r = _run_isolated(_slice_one, x, prefix)
        if r is not _FAILED and r:
            linkers.append(r)
    for x in suffix_sequences:
        r = _run_isolated(_slice_one, x, suffix)
        if r is not _FAILED and r:
            linkers.append(r)

    linked = []
    for n_linked, linker in enumerate(linkers):
        r = _run_isolated(_link_one, linker, prefix, suffix)
        if r is not _FAILED and r:
            linked.extend(r)
        if n_linked > num_samples:
            break
        linked = [x for x in linked if x]
    return linked[:num_samples]


def cut(smiles):
    def cut_nonring(mol):
        if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
            return None

        bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

        try:
            return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            return None
        
    mol = Chem.MolFromSmiles(smiles)
    frags = set()
    # non-ring cut
    for _ in range(3):
        frags_nonring = cut_nonring(mol)
        if frags_nonring is not None:
            frags |= set([Chem.MolToSmiles(f) for f in frags_nonring])
    return frags


class Slicer:
    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        # non-ring single bonds
        bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]'))
        for bond in bonds:
            yield bond
