# -*- coding: utf-8 -*-
import numpy as np

def load_query(path):
    queries = []
    with open(path, 'r') as fin:
        for line_ in fin:
            line = line_.strip()
            toks = line.split(' ', 2)
            assert len(toks) == 3
            rel = int(toks[0])
            qid = int(toks[1].split(':')[1])
            if not queries or not queries[-1].equal_qid(qid):
                queries.append(Query(qid, (rel, toks[2])))
            else:
                queries[-1].append((rel, toks[2]))
    return queries

def load_feat(path):
    queries = []
    with open(path, 'r') as fin:
        for line_ in fin:
            line = line_.strip()
            toks = line.split(' ')
            qid = int(toks[0].split(':')[1])
            feat = np.asarray([float(tok.split(':')[1]) for tok in toks[1:]])
            queries.append(Query(qid, feat=feat))
    return queries

def load_prop(path):
    p = []
    with open(path, 'r') as fin:
        for line_ in fin:
            line = line_.strip()
            toks = line.split(' ')
            qid = int(toks[0].split(':')[1])
            prop = np.asarray([float(tok) for tok in toks[1:]])
            p.append(prop)
    p = np.array(p)
    return p

def load_log(path):
    logs = []
    with open(path, 'r') as fin:
        for line_ in fin:
            line = line_.strip()
            toks = line.split(' ')
            assert len(toks) == 3
            delta = int(toks[0])
            qid = int(toks[1].split(':')[1])
            doc_id = int(toks[2])
            if not logs or not logs[-1].equal_qid(qid):
                logs.append(Query(qid, (doc_id, delta)))
            else:
                logs[-1].append((doc_id, delta))
    return logs

def dump_query(queries, path):
    with open(path, 'w') as fout:
        for query in queries:
            for doc in query._docs:
                rel, feature = doc
                fout.write('{} qid:{} {}\n'.format(rel, query._qid, feature))

def dump_feat(queries, path):
    with open(path, 'w') as fout:
        for query in queries:
            fout.write('qid:{}'.format(query._qid))
            for idx, val in enumerate(query._feat, start=1):
                fout.write(' {}:{}'.format(idx, val))
            fout.write('\n')

class Query(object):

    def __init__(self, qid, doc=None, feat=None, prop=None):
        self._qid = qid
        self._feat = feat
        self._prop = prop
        self._docs = []
        self._inited = False
        if not doc is None:
            self._docs.append(doc)

    def append(self, doc):
        self._docs.append(doc)
        return len(self._docs)

    def equal_qid(self, qid):
        return qid == self._qid

    def get_rel_cnt(self):
        if not self._inited:
            self._inited = True
            self._rel_cnt = 0
            for doc in self._docs:
                if doc[0] == 1:
                    self._rel_cnt += 1

        return self._rel_cnt
