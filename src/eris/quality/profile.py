import sys

import duckdb

con = duckdb.connect()
for t in sys.argv[1:]:
    con.execute(f"create or replace view v as select * exclude (_source_file,_ingested_at) from 'data/bronze/{t}.parquet'")
    n = con.execute("select count(*) from v").fetchone()[0]
    cols = [r[0] for r in con.execute("describe v").fetchall()]
    print(f"\n=== {t} rows={n}")
    for c in cols:
        nd, nn = con.execute(f'select count(distinct "{c}"), count(*)-count("{c}") from v').fetchone()
        line = f"  {c:26s} d={nd:<8} null={nn/n:5.1%}"
        if nd <= 30:
            line += "  " + ", ".join(f"{a}:{b}" for a, b in con.execute(f'select "{c}", count(*) from v group by 1 order by 2 desc').fetchall())
        else:
            line += "  e.g. " + " | ".join(str(r[0])[:40] for r in con.execute(f'select distinct "{c}" from v where "{c}" is not null limit 3').fetchall())
        print(line)
