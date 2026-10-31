"use client";

import { useEffect, useState } from "react";
import styles from "./detection.module.css";

type Recorded = {
  kind: string; command: string | null; executable_path: string | null;
  file_path: string | null; file_name: string | null; pid: string | null;
  parent_pid: string | null; user: string | null; host: string | null;
  uuid: string | null; hash: string | null; local_ip: string | null;
  local_port: string | null; remote_ip: string | null; remote_port: string | null;
};
type Detail = {
  id: number; recorded: Recorded;
  activity: { events: number; first_seen: number | null; last_seen: number | null };
  operations: { relation: string; events: number }[];
  related_entities: (Recorded & { id: number; relation: string; events: number })[];
  limitations: string[];
};
const value = (v: string | null) => v || "Not recorded";
const timestamp = (v: number | null) => v == null ? "Not recorded" : new Date(v * 1000).toISOString().replace("T", " ").replace(".000Z", " UTC");

export function RecordedEntityDetails({ dataset, nodeId }: { dataset: string; nodeId: number }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/graph-datasets/${encodeURIComponent(dataset)}/nodes/${nodeId}`, { signal: controller.signal })
      .then(async response => {
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Unable to load recorded entity details");
        return body as Detail;
      })
      .then(setDetail)
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [dataset, nodeId]);
  if (error) return <p className={styles.recordedNotice} role="alert">{error}</p>;
  if (!detail) return <p role="status" className={styles.recordedLoading}>Loading recorded entity information…</p>;
  const r = detail.recorded;
  const process = r.kind === "subject" || r.kind === "process";
  const executed = detail.related_entities.filter(n => n.kind === "file" && n.relation === "EXECUTE" && n.file_path);
  return <section className={styles.recordedInfo} aria-label="Recorded entity information">
    <h4>Recorded identity</h4>
    {process && <>
      <label>Recorded command<code>{value(r.command)}</code></label>
      <label>Executable path<code>{value(r.executable_path)}</code></label>
      <dl><div><dt>Process ID (PID)</dt><dd>{value(r.pid)}</dd></div>
        {r.parent_pid && <div><dt>Parent PID</dt><dd>{r.parent_pid}</dd></div>}
        {r.user && <div><dt>User</dt><dd>{r.user}</dd></div>}</dl>
    </>}
    {r.kind === "file" && <><label>File name<code>{value(r.file_name)}</code></label><label>Recorded file path<code>{value(r.file_path)}</code></label></>}
    {(r.remote_ip || r.local_ip) && <dl>
      <div><dt>Local endpoint</dt><dd>{r.local_ip ? `${r.local_ip}${r.local_port ? ':' + r.local_port : ''}` : "Not recorded"}</dd></div>
      <div><dt>Remote endpoint</dt><dd>{r.remote_ip ? `${r.remote_ip}${r.remote_port ? ':' + r.remote_port : ''}` : "Not recorded"}</dd></div>
    </dl>}
    {r.host && <label>Recorded host<code>{r.host}</code></label>}
    {r.uuid && <label>Original UUID<code>{r.uuid}</code></label>}
    {r.hash && <label>Recorded hash / identifier<code>{r.hash}</code></label>}
    {executed.length > 0 && <div className={styles.executedFiles}><h4>Observed EXECUTE file paths</h4>
      {executed.map(n => <div key={n.id}><code>{n.file_path}</code><small>File #{n.id} · {n.events.toLocaleString()} recorded events</small></div>)}
      <p>Paths linked by actual EXECUTE events, shown separately from the process’s own metadata.</p>
    </div>}
    {detail.limitations.map(message => <p className={styles.recordedNotice} key={message}>{message}</p>)}
    <h4>Recorded activity</h4><dl>
      <div><dt>Events involving this entity</dt><dd>{detail.activity.events.toLocaleString()}</dd></div>
      <div><dt>First observed</dt><dd>{timestamp(detail.activity.first_seen)}</dd></div>
      <div><dt>Last observed</dt><dd>{timestamp(detail.activity.last_seen)}</dd></div>
    </dl>
    <div className={styles.relations} aria-label="Observed relation types">
      <table><thead><tr><th>Operation</th><th>Events</th></tr></thead><tbody>
        {detail.operations.map(operation => <tr key={operation.relation}><th scope="row">{operation.relation}</th><td>{operation.events.toLocaleString()}</td></tr>)}
      </tbody></table>
    </div>
    {detail.related_entities.length > 0 && <details><summary>Related entities and file operations</summary>
      <ul className={styles.relatedEntities}>{detail.related_entities.map(n => <li key={`${n.id}:${n.relation}`}><strong>{n.relation}</strong><code>{n.file_path || n.command || n.remote_ip || n.file_name || `${n.kind} · name not recorded`}</code><small>Entity #{n.id} · {n.events.toLocaleString()} events</small></li>)}</ul>
      <p>Up to 30 observed entity/operation pairs from this dataset’s scored event split.</p>
    </details>}
  </section>;
}
