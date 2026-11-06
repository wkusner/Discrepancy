"""Builds literature/hs_tau_table.json from the Hardin-Sloane table as read from https://neilsloane.com/sphdesigns/
(via an automated page summary on 2026-11-03).  TRANSCRIPTION IS UNVERIFIED: compare with the source page before
relying on any entry.  Entries are facts (integers/labels) attributed to the source; tau(N) values are the
authors' *conjectured* maxima except where the proof code says otherwise (see sources)."""
import json, re
RAW = """1 0 V1 inf; 2 1 V1 inf; 3 1 V1 [2,3]; 4 2 V1 [3,3]; 5 1 V1 [2,3]; 6 3 V1 [3,4]; 7 2 M90 [3]; 8 3 V1 [3,4]; 9 2 M90 [2,3]; 10 3 B93 [2+,10];
11 3 B93 [2,3]+; 12 5 V1 [3,5]; 13 3 B93 [4]; 14 4 HS92 [2,3]+; 15 3 B93 [2,5]; 16 5 HS92 [3,3]+; 17 4 HS92 [2,3]+; 18 5 R95 [2+,6]; 19 4 HS92 [3]; 20 5 V1 [3,5];
21 4 HS92 [2,3]; 22 5 R95 [2+,10]; 23 5 V2 [2,3]+; 24 7 McL63 [3,4]+; 25 5 V1 [2,5]+; 26 6 V3 [2,3]+; 27 5 R95 [2,3]; 28 6 V3 [2+,4]; 29 6 V3 [2]+; 30 7 V1 [3,4]+;
31 6 V3 [5]+; 32 7 V1 [3,4]+; 33 6 V3 [2,3]+; 34 7 V3 [2,4]+; 35 6 V3 [2,5]+; 36 8 V3 [3,3]+; 37 7 V3 [3]+; 38 7 V3 [3,4]+; 39 7 V3 [2,3]+; 40 8 V3 [3,3]+;
41 7 V3 [2,3]+; 42 8 V3 [2,4]+; 43 7 V3 [6]+; 44 8 V3 [3,3]+; 45 8 V3 [2]+; 46 8 V3 [2,4]+; 47 8 V3 [2,3]+; 48 9 V1 [3,4]+; 49 8 V3 [4]+; 50 9 V3 [2,6]+;
51 8 V3 [2,3]+; 52 9 V3 [3,3]+; 53 8 V3 [2,3]+; 54 9 V3 [3,4]+; 55 9 V3 [2]+; 56 9 V3 [3+,4]; 57 9 V3 [2,3]+; 58 9 V3 [2,4]+; 59 9 V3 [2,3]+; 60 10 V3 [3,3]+;
61 9 V3 [6]+; 62 10 V3 [2,3]+; 63 9 V3 [2,7]+; 64 10 V3 [3,3]+; 65 10 V3 [2]+; 66 10 V3 [2,4]+; 67 10 V3 [2]+; 68 10 V3 [2+,4]; 69 10 V3 [4]+; 70 11 V3 [2,5]+;
71 10 V3 [2,3+]; 72 11 V3 [3,5]+; 73 10 V3 [4]+; 74 11 V3 [2,6]+; 75 11 V3 [2]+; 76 11 V3 [3,3]+; 77 11 V3 [4]+; 78 11 V3 [3,4]+; 79 11 V3 [2]+; 80 11 V3 [3,5]+;
81 11 V3 [4]+; 82 11 V3 [2+,10+]; 83 11 V3 [2,3]+; 84 12 V3 [3,3]+; 85 11 V3 [2,5]+; 86 12 V3 [2,2]+; 87 12 V3 [1]+; 88 12 V3 [3,3]+; 89 12 V3 [2]+; 90 12 V3 [2,4]+;
91 12 V3 [2]+; 92 12 V3 [3,3]+; 93 12 V3 [4]+; 94 13 V3 [2+,2+]; 95 12 V3 [2]+; 96 13 V3 [3,3]+; 97 12 V3 [4]+; 98 13 V3 [2,4]+; 99 12 V3 [2]; 100 13 V3 [3,3]+"""
rows = []
for item in re.split(r'[;\n]', RAW):
    item = item.strip()
    if not item:
        continue
    N, tau, code, grp = item.split()
    rows.append(dict(N=int(N), tau=int(tau), proof_code=code, group_coxeter=grp))
assert [r['N'] for r in rows] == list(range(1, 101))
json.dump(dict(
    source="Hardin and Sloane, tables of spherical designs in 3 dimensions, https://neilsloane.com/sphdesigns/ "
           "(see also arXiv:math/0207211)",
    read_on="2026-11-03", transcription_status="UNVERIFIED (read through an automated summary of the page)",
    meaning="tau(N) = largest t for which an N-point t-design on S^2 is known to the authors; conjectured to be "
            "the maximum, with proof codes: V1 algebraic existence proof, V2 interval proof (N=23 only), "
            "V3 numerical solution with discrepancy <= 1e-26 (no proof); citation codes (M90 Mimura 1990, B93 Bajnok "
            "1993, HS92 Hardin-Sloane 1992, R95 Reznick 1995, McL63 McLaren 1963) name the first proof of existence "
            "of some design with these parameters.",
    rows=rows), open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'hs_tau_table.json'), 'w'), indent=1)
print(len(rows), "rows")
