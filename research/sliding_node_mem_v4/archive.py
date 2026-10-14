"""Persistent byte-preserving event archive, independent of neural admission.

Raw ingestion supplies the original bytes; indexing never reserializes them.
Queries use a strict ingestion cursor, including for equal-time records. Existing
prepared training arrays cannot recreate original telemetry and must be labelled
as prepared records when archived through this API.
"""
import hashlib
import math
import sqlite3
import json
from pathlib import Path


class EventArchive:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, time REAL NOT NULL, source INTEGER NOT NULL,
                destination INTEGER NOT NULL, format TEXT NOT NULL,
                sha256 TEXT NOT NULL, payload BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS source_time ON events(source,time,id);
            CREATE INDEX IF NOT EXISTS destination_time ON events(destination,time,id);
        ''')

    def append(self, event_id, time, source, destination, payload, format='raw'):
        if not isinstance(payload, bytes) or not math.isfinite(time):
            raise ValueError('Original byte payload and finite timestamp required')
        values = (event_id, time, source, destination, format, hashlib.sha256(payload).hexdigest(), payload)
        old = self.db.execute('SELECT * FROM events WHERE id=?', (event_id,)).fetchone()
        if old is not None:
            if old != values:
                raise ValueError('An event ID cannot overwrite different evidence')
            return
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)', values)

    def retrieve(self, entity, before_id, before_time, limit=8):
        if limit < 1:
            raise ValueError('Positive retrieval bound required')
        # Union avoids duplicate self-events; time AND cursor prevent future reads.
        return self.db.execute('''SELECT * FROM events WHERE id IN (
            SELECT id FROM events WHERE source=? AND time<=? AND id<?
            UNION SELECT id FROM events WHERE destination=? AND time<=? AND id<?)
            ORDER BY time DESC,id DESC LIMIT ?''',
            (entity, before_time, before_id, entity, before_time, before_id, limit)).fetchall()

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()


def ingest_jsonl(source, destination):
    """Archive normalized telemetry envelopes while retaining each exact line.

    Required envelope keys: event_id, time (seconds), src, dst. All other fields
    are preserved in the original bytes, including unknown attributes. The
    archive is independent of model training, rarity and novelty admission.
    """
    archive = EventArchive(destination)
    count = 0
    try:
        with Path(source).open('rb') as stream:
            for payload in stream:
                if not payload.strip():
                    continue
                record = json.loads(payload)
                archive.append(int(record['event_id']), float(record['time']),
                               int(record['src']), int(record['dst']), payload)
                count += 1
                if count % 10000 == 0:
                    archive.commit()
    finally:
        archive.close()
    return count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps({'archived_records': ingest_jsonl(args.source, args.output)}))

