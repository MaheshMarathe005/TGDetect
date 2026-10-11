"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  CircleCheck,
  FileText,
  Globe,
  Loader2,
  Network,
  Search,
  ShieldAlert,
  Terminal,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import styles from "./detection.module.css";

type Entity = {
  id: number;
  score: number;
  alert: boolean;
  benchmark_member: boolean | null;
  ip: string | null;
  prediction: string;
  attributes: Record<string, unknown>;
};
type Event = {
  id: string;
  src: number;
  dst: number;
  time: number;
  relation_name: string;
  evidence: number;
};
type Dataset = {
  id: string;
  name: string;
  status: string;
  stage?: string;
  progress: number;
  events?: number;
  nodes?: number;
  error?: string;
  scoring?: string;
};
type EntityOptions = {
  items: { label: string; count: number; alert: number }[];
  available_kinds: string[];
  truncated: boolean;
};
async function readJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}
type Graph = {
  dataset: Dataset;
  nodes: Entity[];
  events: Event[];
  counts: { predicted_malicious: number; predicted_benign: number };
  threshold: number;
  scope: string;
  ip_scope: string;
  truncated: boolean;
  seeds: number[];
};
const label = (n: Entity) =>
  n.ip ||
  String(
    n.attributes.cmd_line ||
      n.attributes.path ||
      n.attributes.host ||
      n.attributes.hostname ||
      n.attributes.name ||
      `${n.attributes.type || "entity"} #${n.id}`,
  );
function nodeIcon(n: Entity) {
  return n.ip
    ? Globe
    : String(n.attributes.type) === "subject"
      ? Terminal
      : FileText;
}

function layout(nodes: Entity[], edges: Event[]) {
  const width = 1100,
    height = 700;
  const points = new Map(
    nodes.map((n, i) => [
      n.id,
      {
        x: width / 2 + Math.cos(i * 2.399) * (100 + Math.sqrt(i) * 55),
        y: height / 2 + Math.sin(i * 2.399) * (90 + Math.sqrt(i) * 38),
      },
    ]),
  );
  const unique = [
    ...new Map(
      edges.map((e) => [
        `${Math.min(e.src, e.dst)}:${Math.max(e.src, e.dst)}`,
        e,
      ]),
    ).values(),
  ];
  // Deterministic spring layout: real relationships, no synthetic graph edges.
  for (let tick = 0; tick < 300; tick++) {
    const force = new Map(nodes.map((n) => [n.id, { x: 0, y: 0 }]));
    for (let i = 0; i < nodes.length; i++)
      for (let j = i + 1; j < nodes.length; j++) {
        const a = points.get(nodes[i].id)!,
          b = points.get(nodes[j].id)!;
        const dx = a.x - b.x,
          dy = a.y - b.y,
          d = Math.max(20, Math.hypot(dx, dy)),
          push = 18000 / (d * d);
        // Reserve room for the IP/name plate as well as the circular icon.
        if (Math.abs(dx) < 195 && Math.abs(dy) < 110) {
          const overlapX = 195 - Math.abs(dx),
            overlapY = 110 - Math.abs(dy);
          if (overlapX < overlapY) {
            const shift = (Math.sign(dx) || 1) * overlapX * 0.4;
            force.get(nodes[i].id)!.x += shift;
            force.get(nodes[j].id)!.x -= shift;
          } else {
            const shift = (Math.sign(dy) || 1) * overlapY * 0.4;
            force.get(nodes[i].id)!.y += shift;
            force.get(nodes[j].id)!.y -= shift;
          }
        }
        force.get(nodes[i].id)!.x += (dx / d) * push;
        force.get(nodes[i].id)!.y += (dy / d) * push;
        force.get(nodes[j].id)!.x -= (dx / d) * push;
        force.get(nodes[j].id)!.y -= (dy / d) * push;
      }
    for (const e of unique) {
      const a = points.get(e.src),
        b = points.get(e.dst);
      if (!a || !b || e.src === e.dst) continue;
      const dx = b.x - a.x,
        dy = b.y - a.y,
        d = Math.hypot(dx, dy) || 1,
        pull = (d - 165) * 0.025;
      force.get(e.src)!.x += (dx / d) * pull;
      force.get(e.src)!.y += (dy / d) * pull;
      force.get(e.dst)!.x -= (dx / d) * pull;
      force.get(e.dst)!.y -= (dy / d) * pull;
    }
    for (const n of nodes) {
      const p = points.get(n.id)!,
        f = force.get(n.id)!;
      p.x = Math.max(
        100,
        Math.min(
          width - 100,
          p.x + Math.max(-5, Math.min(5, f.x + (width / 2 - p.x) * 0.003)),
        ),
      );
      p.y = Math.max(
        70,
        Math.min(
          height - 75,
          p.y + Math.max(-5, Math.min(5, f.y + (height / 2 - p.y) * 0.003)),
        ),
      );
    }
  }
  return points;
}
export function DetectionGraph({
  investigate,
}: {
  investigate: (node: number) => void;
}) {
  const [data, setData] = useState<Graph | null>(null),
    [loading, setLoading] = useState(true),
    [error, setError] = useState("");
  const [search, setSearch] = useState(""),
    [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all"),
    [selected, setSelected] = useState<number | null>(null),
    [zoom, setZoom] = useState(1),
    [truth, setTruth] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]),
    [dataset, setDataset] = useState("cadets"),
    [kind, setKind] = useState("ip"),
    [value, setValue] = useState(""),
    [anchor, setAnchor] = useState<number | null>(null),
    [options, setOptions] = useState<EntityOptions | null>(null),
    [hops, setHops] = useState(2),
    [upload, setUpload] = useState<Dataset | null>(null),
    [uploadError, setUploadError] = useState(""),
    [uploading, setUploading] = useState(false);
  useEffect(() => {
    readJSON<{ items: Dataset[] }>("/api/graph-datasets")
      .then((d) => {
        setDatasets(d.items);
        const pending = d.items.find((x) =>
          ["queued", "processing"].includes(x.status),
        );
        if (pending) setUpload(pending);
      })
      .catch((e) => setError(e.message));
  }, []);
  function changeDataset(id: string) {
    setDataset(id);
    setKind("ip");
    setValue("");
    setAnchor(null);
    setSelected(null);
    setSearch("");
    setQuery("");
    setFilter("all");
    setData(null);
  }
  useEffect(() => {
    const controller = new AbortController();
    setOptions(null);
    const params = new URLSearchParams({ kind, q: query, prediction: filter });
    readJSON<EntityOptions>(
      `/api/graph-datasets/${dataset}/entities?${params}`,
      { signal: controller.signal },
    )
      .then((d) => {
        if (!d.available_kinds.includes(kind) && d.available_kinds.length) {
          setKind(d.available_kinds[0]);
          return;
        }
        setOptions(d);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [dataset, kind, query, filter]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setData(null);
    const params = new URLSearchParams({
      kind,
      value,
      prediction: filter,
      limit: "24",
      hops: String(hops),
    });
    if (anchor !== null) params.set("node", String(anchor));
    readJSON<Graph>(`/api/graph-datasets/${dataset}/graph?${params}`, {
      signal: controller.signal,
    })
      .then(setData)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [dataset, kind, value, filter, anchor, hops]);
  useEffect(() => {
    if (!upload || !["queued", "processing"].includes(upload.status)) return;
    const controller = new AbortController();
    const timer = setInterval(
      () =>
        readJSON<Dataset>(`/api/graph-datasets/${upload.id}`, {
          signal: controller.signal,
        })
          .then((d) => {
            setUpload(d);
            if (d.status === "ready") {
              setDatasets((old) => [...old.filter((x) => x.id !== d.id), d]);
              changeDataset(d.id);
            }
            if (d.status === "failed")
              setUploadError(d.error || "Scoring failed");
          })
          .catch((e) => {
            if (!controller.signal.aborted) setUploadError(e.message);
          }),
      1000,
    );
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, [upload?.id, upload?.status]);
  async function uploadFile(file: File) {
    setUploading(true);
    setUploadError("");
    setUpload(null);
    try {
      const d = await readJSON<Dataset>(
        `/api/graph-datasets?name=${encodeURIComponent(file.name)}`,
        {
          method: "POST",
          body: file,
          headers: { "Content-Type": "application/octet-stream" },
        },
      );
      setUpload(d);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }
  function selectNode(id: number) {
    setSelected(id);
    setAnchor(id);
  }
  const shown = data?.nodes || [];
  const ids = new Set(shown.map((n) => n.id));
  const events = useMemo(
    () =>
      data?.events.filter(
        (e) =>
          shown.some((n) => n.id === e.src) &&
          shown.some((n) => n.id === e.dst),
      ) || [],
    [data, shown],
  );
  const positions = useMemo(() => layout(shown, events), [shown, events]);
  const groups = new Map<
    string,
    { event: Event; count: number; relations: Set<string> }
  >();
  for (const e of events) {
    const key = `${e.src}:${e.dst}`,
      old = groups.get(key);
    groups.set(key, {
      event: !old || e.evidence > old.event.evidence ? e : old.event,
      count: (old?.count || 0) + 1,
      relations: new Set([...(old?.relations || []), e.relation_name]),
    });
  }
  const focus = data?.nodes.find((n) => n.id === selected);
  const neighborIds = new Set([
    selected,
    ...events
      .filter((e) => e.src === selected || e.dst === selected)
      .flatMap((e) => [e.src, e.dst]),
  ]);
  function exportGraph() {
    if (!data) return;
    const url = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify(
            { ...data, visible_node_ids: [...ids], filter },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "cadets-detection-graph.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section
      className={styles.workspace}
      aria-label="Detection graph workspace"
    >
      <div className={styles.heading}>
        <div>
          <p>V3 NODE CLASSIFICATION</p>
          <h2>Malicious & benign detection graph</h2>
          <span>
            Upload endpoint logs, run V3 detection, and explore connected IPs,
            hosts and entities.
          </span>
        </div>
        <button onClick={exportGraph} disabled={!data}>
          <ArrowDownToLine size={15} />
          Export graph
        </button>
      </div>
      <div className={styles.uploadPanel}>
        <label>
          Dataset
          <select
            aria-label="Graph dataset"
            value={dataset}
            onChange={(e) => changeDataset(e.target.value)}
          >
            {datasets
              .filter((d) => d.status === "ready")
              .map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
          </select>
        </label>
        <label className={styles.uploadButton}>
          {uploading ? "Uploading…" : "Upload logs / dataset"}
          <input
            aria-label="Upload graph logs"
            type="file"
            accept=".json,.jsonl,.ndjson,.csv"
            disabled={
              uploading ||
              (!!upload && ["queued", "processing"].includes(upload.status))
            }
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void uploadFile(file);
              e.target.value = "";
            }}
          />
        </label>
        <details>
          <summary>Supported log format</summary>
          <p>
            JSON array, JSONL or CSV. Each event needs time/timestamp, relation,
            and src/dst endpoints (IDs, IPs or objects). Optional src_ip/dst_ip,
            src_host/dst_host and endpoint attributes supply graph labels.
            Supported relations: CONNECT, EXECUTE, OPEN, READ, RECVFROM,
            RECVMSG, SENDMSG, SENDTO, WRITE. Up to 100,000 events / 50 MB.
            Unknown formats are rejected with the failing row; predictions and
            labels in uploads are ignored.
          </p>
          <pre>
            {JSON.stringify(
              {
                timestamp: "2026-10-11T10:00:00Z",
                relation: "CONNECT",
                src: {
                  id: "workstation",
                  host: "workstation-01",
                  type: "subject",
                  cmd_line: "curl",
                },
                dst: { id: "server", remote_ip: "192.0.2.10", type: "netflow" },
              },
              null,
              2,
            )}
          </pre>
        </details>
        {upload && (
          <div role="status" className={styles.uploadStatus}>
            <span>
              {upload.name} · {upload.stage || upload.status}
            </span>
            <progress max={1} value={upload.progress} />
          </div>
        )}
        {uploadError && <p role="alert">{uploadError}</p>}
      </div>
      <div className={styles.stats}>
        <div className={styles.red}>
          <ShieldAlert size={20} />
          <span>
            Predicted malicious
            <strong>{data?.counts.predicted_malicious ?? "—"}</strong>
          </span>
          <small>Red nodes · score above threshold</small>
        </div>
        <div className={styles.green}>
          <CircleCheck size={20} />
          <span>
            Predicted benign
            <strong>{data?.counts.predicted_benign ?? "—"}</strong>
          </span>
          <small>Green nodes · score at or below threshold</small>
        </div>
        <div>
          <Globe size={20} />
          <span>
            Recorded IP entities
            <strong>{data?.nodes.filter((n) => n.ip).length ?? "—"}</strong>
          </span>
          <small>Counts describe this displayed sample</small>
        </div>
      </div>
      <div className={styles.toolbar}>
        <label>
          Entity type
          <select
            aria-label="Graph entity type"
            value={kind}
            onChange={(e) => {
              setKind(e.target.value);
              setValue("");
              setAnchor(null);
              setSelected(null);
              setQuery("");
              setSearch("");
            }}
          >
            {[...new Set([kind, ...(options?.available_kinds || [])])].map(
              (k) => (
                <option key={k} value={k}>
                  {k === "ip"
                    ? "IP addresses"
                    : k === "host"
                      ? "Hosts"
                      : k === "entity"
                        ? "All entity names"
                        : k === "process"
                          ? "Processes"
                          : "Files"}
                </option>
              ),
            )}
          </select>
        </label>
        <label>
          Starting entity
          <select
            aria-label="Graph starting entity"
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setAnchor(null);
              setSelected(null);
            }}
          >
            <option value="">
              Highest-scoring {filter === "all" ? "entity" : filter + " entity"}
            </option>
            {options?.items.map((o) => (
              <option key={o.label} value={o.label}>
                {o.label} ({o.count} {o.count === 1 ? "entity" : "entities"})
              </option>
            ))}
          </select>
        </label>
        {options?.truncated && (
          <small>First 100 matches; search to narrow.</small>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            setQuery(search.trim());
          }}
        >
          <Search size={16} />
          <input
            aria-label="Search graph entities"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={`Search ${kind}s in this dataset`}
          />
          <button>Find entities</button>
        </form>
        <label>
          Show
          <select
            aria-label="Detection class filter"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setSelected(null);
              setAnchor(null);
              setValue("");
            }}
          >
            <option value="all">All starting entities</option>
            <option value="malicious">Malicious + connected context</option>
            <option value="benign">Benign + connected context</option>
          </select>
        </label>
        <label>
          Connections
          <select
            aria-label="Graph connection depth"
            value={hops}
            onChange={(e) => setHops(Number(e.target.value))}
          >
            <option value={1}>1 hop</option>
            <option value={2}>2 hops</option>
            <option value={3}>3 hops</option>
          </select>
        </label>
        <button
          onClick={() => {
            setAnchor(null);
            setValue("");
            setSelected(null);
          }}
        >
          Reset focus
        </button>
        <button
          aria-label="Zoom detection graph out"
          onClick={() => setZoom(Math.max(0.6, zoom - 0.15))}
        >
          <ZoomOut size={16} />
        </button>
        <button
          aria-label="Zoom detection graph in"
          onClick={() => setZoom(Math.min(1.8, zoom + 0.15))}
        >
          <ZoomIn size={16} />
        </button>
      </div>
      <p className={styles.note}>
        {anchor !== null
          ? `Focused on entity #${anchor}.`
          : value
            ? `Focused on ${kind}: ${value}.`
            : "Showing the highest-scoring starting entity and its connections."}{" "}
        Class filters choose starting entities; connected benign and malicious
        nodes remain visible. Click any red or green node to load its connected
        neighborhood.
      </p>
      {data?.dataset.scoring && (
        <p className={styles.note}>
          Predictions were computed from this upload using the CADETS-trained V3
          model. Detection accuracy on other log domains has not been validated.
        </p>
      )}
      {loading ? (
        <div className={styles.state} role="status">
          <Loader2 size={20} />
          Loading observed network neighborhoods…
        </div>
      ) : error ? (
        <div role="alert" className={styles.state}>
          {error}
        </div>
      ) : (
        <>
          <div className={`${styles.body} ${focus ? styles.withDetails : ""}`}>
            <div
              className={styles.canvas}
              role="region"
              aria-label="Malicious and benign node graph"
              tabIndex={0}
            >
              <svg
                viewBox="0 0 1100 700"
                width={1100 * zoom}
                height={700 * zoom}
                role="img"
                aria-label="Red predicted malicious nodes and green predicted benign nodes. Network nodes display recorded IPs."
              >
                <defs>
                  <pattern
                    id="detection-dots"
                    width="26"
                    height="26"
                    patternUnits="userSpaceOnUse"
                  >
                    <circle cx="1" cy="1" r="1" fill="#263651" />
                  </pattern>
                  <marker
                    id="detection-arrow"
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto"
                  >
                    <path d="M0 0L10 5L0 10z" fill="#6d82a4" />
                  </marker>
                </defs>
                <rect width="1100" height="700" fill="url(#detection-dots)" />
                {[...groups.values()].map(({ event: e, count, relations }) => {
                  const a = positions.get(e.src)!,
                    b = positions.get(e.dst)!,
                    dx = b.x - a.x,
                    dy = b.y - a.y,
                    d = Math.hypot(dx, dy) || 1;
                  return (
                    <g
                      key={`${e.src}:${e.dst}`}
                      opacity={
                        selected === null ||
                        e.src === selected ||
                        e.dst === selected
                          ? 1
                          : 0.12
                      }
                    >
                      <path
                        d={
                          e.src === e.dst
                            ? `M${a.x - 14},${a.y - 20}C${a.x - 65},${a.y - 75} ${a.x + 65},${a.y - 75} ${a.x + 14},${a.y - 20}`
                            : `M${a.x + (dx / d) * 27},${a.y + (dy / d) * 27}L${b.x - (dx / d) * 29},${b.y - (dy / d) * 29}`
                        }
                        stroke={
                          e.src === selected || e.dst === selected
                            ? "#b5a1ed"
                            : "#4c6488"
                        }
                        strokeWidth="1.6"
                        fill="none"
                        markerEnd="url(#detection-arrow)"
                      />
                      <title>
                        {[...relations].join(", ")} · {count} observed events
                      </title>
                    </g>
                  );
                })}
                {shown.map((n) => {
                  const p = positions.get(n.id)!,
                    Icon = nodeIcon(n),
                    color = n.alert ? "#fb7185" : "#34d399";
                  return (
                    <g
                      key={n.id}
                      data-prediction={n.prediction}
                      data-node-id={n.id}
                      role="button"
                      tabIndex={0}
                      aria-label={`${n.alert ? "Predicted malicious" : "Predicted benign"} ${label(n)}`}
                      className={styles.node}
                      transform={`translate(${p.x},${p.y})`}
                      opacity={
                        selected === null || neighborIds.has(n.id) ? 1 : 0.2
                      }
                      onClick={() => selectNode(n.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          selectNode(n.id);
                        }
                      }}
                    >
                      <circle
                        r={selected === n.id ? 34 : 29}
                        fill="none"
                        stroke={color}
                        opacity=".16"
                        strokeWidth="6"
                      />
                      <circle
                        r="25"
                        fill={n.alert ? "#3d2338" : "#153a36"}
                        stroke={color}
                        strokeWidth={selected === n.id ? 3 : 1.8}
                      />
                      <Icon x={-10} y={-10} size={20} color={color} />
                      <rect
                        x="-91"
                        y="34"
                        width="182"
                        height="35"
                        rx="7"
                        fill="#142139"
                        stroke="#2b3d5b"
                      />
                      <text
                        textAnchor="middle"
                        y="48"
                        className={styles.nodeLabel}
                      >
                        {label(n).length > 26
                          ? "…" + label(n).slice(-25)
                          : label(n)}
                      </text>
                      <text
                        textAnchor="middle"
                        y="61"
                        className={styles.nodeMeta}
                      >
                        {n.alert ? "MALICIOUS" : "BENIGN"} prediction · #{n.id}
                      </text>
                      <title>
                        {label(n)} · evidence {n.score.toFixed(4)} ·{" "}
                        {n.ip
                          ? "recorded remote IP"
                          : "IP not recorded for this entity"}
                      </title>
                    </g>
                  );
                })}
              </svg>
              {!shown.length && (
                <div className={styles.empty}>
                  No matching entities in this graph sample.
                </div>
              )}
            </div>
            {focus && (
              <aside
                className={styles.details}
                aria-label="Detection node details"
              >
                <div>
                  <span className={focus.alert ? styles.red : styles.green}>
                    {focus.alert ? (
                      <ShieldAlert size={22} />
                    ) : (
                      <CircleCheck size={22} />
                    )}
                  </span>
                  <button
                    aria-label="Close node details"
                    onClick={() => setSelected(null)}
                  >
                    <X size={16} />
                  </button>
                </div>
                <p className={focus.alert ? styles.red : styles.green}>
                  {focus.alert ? "PREDICTED MALICIOUS" : "PREDICTED BENIGN"}
                </p>
                <h3>{label(focus)}</h3>
                <dl>
                  <div>
                    <dt>Entity ID</dt>
                    <dd>#{focus.id}</dd>
                  </div>
                  <div>
                    <dt>Recorded remote IP</dt>
                    <dd>{focus.ip || "Not recorded"}</dd>
                  </div>
                  <div>
                    <dt>Entity type</dt>
                    <dd>{String(focus.attributes.type || "unknown")}</dd>
                  </div>
                  <div>
                    <dt>V3 evidence score</dt>
                    <dd>{focus.score.toFixed(4)}</dd>
                  </div>
                  <div>
                    <dt>Alert threshold</dt>
                    <dd>{data!.threshold.toFixed(4)}</dd>
                  </div>
                </dl>
                <p className={styles.decision}>
                  {focus.score.toFixed(4)} {focus.alert ? ">" : "≤"}{" "}
                  {data!.threshold.toFixed(4)} →{" "}
                  {focus.alert ? "malicious prediction" : "benign prediction"}
                </p>
                {dataset === "cadets" && (
                  <button
                    className={styles.investigate}
                    onClick={() => investigate(focus.id)}
                  >
                    Investigate entity <ArrowRight size={15} />
                  </button>
                )}
                {dataset !== "cadets" && (
                  <div className={styles.note}>
                    Connected events shown here were scored from this upload.
                    Select another node to follow its neighborhood.
                  </div>
                )}
                <label className={styles.truth}>
                  <input
                    type="checkbox"
                    checked={truth}
                    onChange={(e) => setTruth(e.target.checked)}
                  />
                  Show benchmark label separately
                </label>
                {truth && (
                  <p className={styles.note}>
                    Benchmark:{" "}
                    {focus.benchmark_member === null
                      ? "not supplied"
                      : focus.benchmark_member
                        ? "malicious node listed"
                        : "not listed as malicious"}
                    . This label did not determine the node color.
                  </p>
                )}
                <details>
                  <summary>Observed attributes</summary>
                  <pre>{JSON.stringify(focus.attributes, null, 2)}</pre>
                </details>
              </aside>
            )}
          </div>
          <div className={styles.legend}>
            <span>
              <i className={styles.redDot} />
              Predicted malicious
            </span>
            <span>
              <i className={styles.greenDot} />
              Predicted benign
            </span>
            <span>
              <Globe size={14} />
              IP / network
            </span>
            <span>
              <Terminal size={14} />
              Process
            </span>
            <span>
              <FileText size={14} />
              File
            </span>
            <span>Click a node to inspect its decision</span>
          </div>
          <p className={styles.note}>
            {data?.scope} {data?.ip_scope}{" "}
            {data?.truncated
              ? "Additional entities/events were omitted by the display bounds."
              : ""}
          </p>
        </>
      )}
    </section>
  );
}
