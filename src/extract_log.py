# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import timeit
import numpy as np
import argparse
from .lib.utils import makedirs, is_complex
csv.field_size_limit(sys.maxsize)

click_field_name = ["date", "format", "paper", "ip", "mode", "uid", "session",
                    "port", "id", "useragent", "usercookies"]

query_field_name = ["date", "query", "ip", "referer", "mode", "num_results",
                    "results", "uid", "session", "port", "overlength", "id",
                    "useragent", "usercookies"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate log and feature')
    parser.add_argument('-m', type=int, help='max pos to be estimated')
    parser.add_argument('query_path', help='query path')
    parser.add_argument('click_path', help='click path')
    parser.add_argument('feat_path', help='feat path')
    parser.add_argument('log_dir', help='click log dir')

    args = parser.parse_args()
    start = timeit.default_timer()

    M = args.m
    query_path = args.query_path
    click_path = args.click_path
    feat_path = args.feat_path
    log0_path = os.path.join(args.log_dir, 'train.log0.txt')
    log1_path = os.path.join(args.log_dir, 'train.log1.txt')
    log2_path = os.path.join(args.log_dir, 'train.log2.txt')
    uid2qid = {}
    click_set = set()

    with open(click_path, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=click_field_name)
        for row in reader:
            if row['format'] == 'abs':
                uid = row['uid']
                if uid not in uid2qid:
                    uid2qid[uid] = len(uid2qid)
                qid = uid2qid[uid]
                click_set.add((qid, row['paper']))

    query_doc2id = {}
    makedirs(args.log_dir)
    makedirs(os.path.dirname(args.feat_path))
    with open(query_path, 'r') as fin, open(feat_path, 'w') as fout_feat, open(log0_path, 'w') as log0, open(log1_path, 'w') as log1, open(log2_path, 'w') as log2:
        reader = csv.DictReader(fin, delimiter='\t', quoting=csv.QUOTE_NONE,
                                fieldnames=query_field_name)
        for row in reader:
            uid = row['uid']
            if uid not in uid2qid:
                continue
            query = row['query']
            feat = int(is_complex(query))

            qid = uid2qid[uid]
            query_doc2id[qid] = {}
            fout_feat.write('qid:{} 1:{}\n'.format(qid, feat))
            toks = row['results'].split('*')
            num_ranker = int(toks[0])
            selected_ranker_id = int(toks[1])
            selected_rank = toks[3 + selected_ranker_id]
            selected_length = len(selected_rank)
            max_length = min(selected_length, M)
            for ranker_id in range(num_ranker):
                logger = eval('log{}'.format(ranker_id))
                rank = toks[3 + ranker_id]
                docs = rank.split(',')[1:]
                for rk in range(min(max_length, len(docs))):
                    doc = docs[rk]
                    if doc not in query_doc2id[qid]:
                        query_doc2id[qid][doc] = len(query_doc2id[qid])
                    doc_id = query_doc2id[qid][doc]
                    clicked = 0
                    if (qid, doc) in click_set:
                        clicked = 1
                    logger.write('{} qid:{} {}\n'.format(clicked, qid, doc_id))
