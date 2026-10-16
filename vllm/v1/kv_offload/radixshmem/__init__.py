# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV offloading to a node-wide shared CPU pool indexed by a radix tree.

``RadixShmemConnector`` implements ``KVConnectorBase_V1`` with its own
scheduler, worker and metadata under ``kv_connector/v1/radixshmem``.
``RadixShmemBackend`` supplies the shared pool, manager and GPU copies.
The scheduler passes prefixes, chunk coordinates, leases and reservations;
no generic OffloadingManager key protocol is used. Cache-manager rules,
byte-layout conversion, copy handlers and event formatting are shared with vLLM.
The pool and the prefix index are provided by the ``radixshmem`` package and
are shared by every DP rank on the node, so a prefix offloaded by one rank is
a hit for all of them. Sliding-window and align-mode mamba groups get their
own slot pools next to the full-attention tree; with ``remote`` enabled,
prefixes found on other nodes are pulled into the local pool first.
"""
