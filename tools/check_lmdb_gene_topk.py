#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import lmdb
import msgpack


@dataclass
class Counters:
    total: int = 0
    raw_ge_topk: int = 0
    raw_lt_topk: int = 0
    mapped_ge_topk: int = 0
    mapped_lt_topk: int = 0
    mapped_zero: int = 0
    decode_fail: int = 0
    missing_key: int = 0


def iter_lmdb_paths(lmdb_path: str) -> List[str]:
    if os.path.isdir(lmdb_path):
        shards = [
            os.path.join(lmdb_path, p)
            for p in os.listdir(lmdb_path)
            if p.endswith('.db') and os.path.isdir(os.path.join(lmdb_path, p))
        ]
        return sorted(shards)
    return [lmdb_path]


def load_record(txn, idx: int):
    key10 = f'{idx:010d}'.encode('ascii')
    value = txn.get(key10)
    if value is None:
        key9 = f'{idx:09d}'.encode('ascii')
        value = txn.get(key9)
    if value is None:
        return None, 'missing_key'
    try:
        data = msgpack.unpackb(value, raw=False)
        return data, None
    except Exception:
        try:
            data = json.loads(value.decode('utf-8'))
            return data, None
        except Exception:
            return None, 'decode_fail'


def build_mapping(lmdb_vocab_path: Optional[str], scgpt_vocab_path: str) -> Optional[Dict[int, int]]:
    if not lmdb_vocab_path:
        return None
    with open(scgpt_vocab_path, 'r') as f:
        scgpt_vocab = json.load(f)
    gene_vocab = {gene: int(idx) for gene, idx in scgpt_vocab.items()}
    with open(lmdb_vocab_path, 'r') as f:
        lmdb_vocab = json.load(f)
    lmdb_id2gene = {v: k for k, v in lmdb_vocab.items()}
    mapping = {
        lmdb_id: gene_vocab[gene_name]
        for lmdb_id, gene_name in lmdb_id2gene.items()
        if gene_name in gene_vocab
    }
    return mapping


def summarize(name: str, counters: Counters):
    t = max(counters.total, 1)
    print(f'[{name}] total={counters.total}')
    print(f'  raw_len>=topk: {counters.raw_ge_topk} ({counters.raw_ge_topk / t:.2%})')
    print(f'  raw_len<topk : {counters.raw_lt_topk} ({counters.raw_lt_topk / t:.2%})')
    print(f'  mapped_len>=topk: {counters.mapped_ge_topk} ({counters.mapped_ge_topk / t:.2%})')
    print(f'  mapped_len<topk : {counters.mapped_lt_topk} ({counters.mapped_lt_topk / t:.2%})')
    print(f'  mapped_len==0   : {counters.mapped_zero} ({counters.mapped_zero / t:.2%})')
    print(f'  decode_fail={counters.decode_fail}, missing_key={counters.missing_key}')


def main():
    parser = argparse.ArgumentParser(
        description='Check whether LMDB gene_ids provide enough valid top-k tokens before/after vocab mapping.'
    )
    parser.add_argument('--lmdb-path', required=True, help='LMDB shard dir or single .db path')
    parser.add_argument('--topk', type=int, default=1200, help='Target token length (default: 1200)')
    parser.add_argument('--lmdb-vocab-path', default=None, help='LMDB vocab json (optional)')
    parser.add_argument('--scgpt-vocab-path', required=True, help='scGPT vocab json')
    parser.add_argument('--max-shards', type=int, default=0, help='0 means all shards')
    parser.add_argument('--samples-per-shard', type=int, default=0, help='0 means all samples in shard')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    mapping = build_mapping(args.lmdb_vocab_path, args.scgpt_vocab_path)
    if mapping is None:
        print('Mapping mode: disabled (raw gene_ids only)')
    else:
        print(f'Mapping mode: enabled, mapped ids={len(mapping)}')

    shards = iter_lmdb_paths(args.lmdb_path)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    print(f'Shards to inspect: {len(shards)}')

    global_cnt = Counters()

    for shard in shards:
        shard_cnt = Counters()
        env = lmdb.open(shard, readonly=True, lock=False)
        with env.begin() as txn:
            len_bytes = txn.get(b'__len__')
            n = int(len_bytes.decode('utf-8')) if len_bytes is not None else txn.stat()['entries']
            if args.samples_per_shard > 0 and args.samples_per_shard < n:
                indices = random.sample(range(n), args.samples_per_shard)
            else:
                indices = range(n)

            for idx in indices:
                shard_cnt.total += 1
                data, err = load_record(txn, idx)
                if err == 'missing_key':
                    shard_cnt.missing_key += 1
                    continue
                if err == 'decode_fail':
                    shard_cnt.decode_fail += 1
                    continue

                gene_ids_lmdb = list(data.get('gene_ids', []))
                raw_len = len(gene_ids_lmdb)
                if raw_len >= args.topk:
                    shard_cnt.raw_ge_topk += 1
                else:
                    shard_cnt.raw_lt_topk += 1

                if mapping is not None:
                    mapped_ids = [mapping[g] for g in gene_ids_lmdb if g in mapping]
                    mapped_len = len(mapped_ids)
                else:
                    mapped_len = raw_len

                if mapped_len >= args.topk:
                    shard_cnt.mapped_ge_topk += 1
                else:
                    shard_cnt.mapped_lt_topk += 1
                if mapped_len == 0:
                    shard_cnt.mapped_zero += 1

        env.close()

        global_cnt.total += shard_cnt.total
        global_cnt.raw_ge_topk += shard_cnt.raw_ge_topk
        global_cnt.raw_lt_topk += shard_cnt.raw_lt_topk
        global_cnt.mapped_ge_topk += shard_cnt.mapped_ge_topk
        global_cnt.mapped_lt_topk += shard_cnt.mapped_lt_topk
        global_cnt.mapped_zero += shard_cnt.mapped_zero
        global_cnt.decode_fail += shard_cnt.decode_fail
        global_cnt.missing_key += shard_cnt.missing_key

    summarize('GLOBAL', global_cnt)


if __name__ == '__main__':
    main()
