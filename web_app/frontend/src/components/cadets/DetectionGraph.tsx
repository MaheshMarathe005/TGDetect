"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  CircleCheck,
  Loader2,
  Network,
  Search,
  ShieldAlert,
  TriangleAlert,
  X,
} from "lucide-react";
import styles from "./detection.module.css";
import { Graph3D } from "./Graph3D";
import { RecordedEntityDetails } from "./RecordedEntityDetails";
import { nodeRisk, RISK_COLORS, WARNING_RATIO } from "./graph-risk";

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
  prediction_rule?: string;
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
    [truth, setTruth] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]),
    [dataset, setDataset] = useState("cadets"),
    [kind, setKind] = useState("ip"),
    [value, setValue] = useState(""),
    [anchor, setAnchor] = useState<number | null>(null),
    [options, setOptions] = useState<EntityOptions | null>(null),
    [hops, setHops] = useState(3),
    [nodeLimit, setNodeLimit] = useState(240),
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
      limit: String(nodeLimit),
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
  }, [dataset, kind, value, filter, anchor, hops, nodeLimit]);
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
  const shown = useMemo(() => data?.nodes || [], [data]);
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
  const focus = data?.nodes.find((n) => n.id === selected);
  const clustered = Boolean(data?.dataset.prediction_rule?.includes("cluster"));
  const riskCounts = { high: 0, medium: 0, low: 0, unknown: 0 };
  for (const node of shown) riskCounts[nodeRisk(node, data?.threshold ?? NaN, !clustered).level]++;
  const focusRisk = focus ? nodeRisk(focus, data?.threshold ?? NaN, !clustered) : null;
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
          <h2>Explore the entity constellation</h2>
          <span>
            A fixed 3D map of connected IPs, processes and files. Orbit the graph to explore every relationship.
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
        <div className={styles.red}><ShieldAlert size={20}/><span>Predicted malicious<strong>{data ? riskCounts.high : "—"}</strong></span>
          <small>Red · {clustered ? "selected by anomaly clustering" : `score > ${data?.threshold.toFixed(6) ?? "threshold"}`}</small></div>
        <div className={styles.orange}><TriangleAlert size={20}/><span>Medium · near threshold<strong>{data ? riskCounts.medium : "—"}</strong></span>
          <small>{clustered ? "Warning band unavailable for clustering decisions" : `Orange · 80–100% of threshold · at or below the alert cutoff`}</small></div>
        <div className={styles.green}><CircleCheck size={20}/><span>Predicted benign · low score<strong>{data ? riskCounts.low : "—"}</strong></span>
          <small>Green · {clustered ? "not selected by anomaly clustering" : "score < 80% of threshold"}</small></div>
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
            onChange={(e) => { setHops(Number(e.target.value)); setSelected(null); }}
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
        <label>Graph size
          <select aria-label="Graph node limit" value={nodeLimit} onChange={e => { setNodeLimit(Number(e.target.value)); setSelected(null); }}>
            <option value={60}>60 nodes</option><option value={240}>240 nodes</option><option value={400}>400 nodes</option>
          </select>
        </label>
      </div>
      <p className={styles.note}>
        {anchor !== null
          ? `Focused on entity #${anchor}.`
          : value
            ? `Focused on ${kind}: ${value}.`
            : "Showing the highest-scoring starting entity and its connections."}{" "}
        Class filters choose starting entities; connected benign and malicious
        nodes remain visible. Connected entities gather into clusters; distance is a layout aid, not a model score. Click a node to inspect it without moving the layout.
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
              aria-label="Interactive 3D detection graph"
            >
              <Graph3D nodes={shown} edges={events} selected={selected} onSelect={setSelected} threshold={data?.threshold ?? NaN} thresholdBased={!clustered} />
            </div>
            {focus && (
              <aside
                className={styles.details}
                aria-label="Detection node details"
              >
                <div>
                  <span style={{ color: focusRisk?.color }}>
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
                <p style={{ color: focusRisk?.color }}>
                  {focusRisk?.label.toUpperCase()}
                </p>
                <h3>{label(focus)}</h3>
                <RecordedEntityDetails key={`${dataset}:${focus.id}`} dataset={dataset} nodeId={focus.id} />
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
                    <dt>Node anomaly score</dt>
                    <dd>{focus.score.toFixed(6)}</dd>
                  </div>
                  <div>
                    <dt>{clustered ? "Validation reference" : "Alert threshold"}</dt>
                    <dd>{data!.threshold.toFixed(6)}</dd>
                  </div>
                  {!clustered && <div><dt>Medium band starts at</dt><dd>{(data!.threshold * WARNING_RATIO).toFixed(6)} (80%)</dd></div>}
                </dl>
                {focusRisk?.level === "medium" && <p className={styles.mediumNote}>Near-threshold warning. The saved model prediction remains benign because the score has not exceeded the alert threshold.</p>}
                <p className={styles.decision}>
                  {clustered ? (
                    <>{focus.alert ? "Selected" : "Not selected"} by anomaly clustering → {focus.alert ? "malicious prediction" : "benign prediction"}</>
                  ) : (
                    <>{focus.score.toFixed(6)} {focus.alert ? ">" : "≤"}{" "}{data!.threshold.toFixed(6)} → {focus.alert ? "malicious prediction" : "benign prediction"}</>
                  )}
                </p>
                <button className={styles.investigate} onClick={() => setAnchor(focus.id)}>
                  Load this neighborhood <Network size={15} />
                </button>
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
                    Connected events use the saved predictions for this dataset.
                    Use Load this neighborhood to explore more connections.
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
                    . This label did not determine the node risk color.
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
            <span><i style={{background:RISK_COLORS.high}}/>Red: predicted malicious</span>
            {!clustered && <span><i style={{background:RISK_COLORS.medium}}/>Orange: 80–100% of threshold (warning)</span>}
            <span><i style={{background:RISK_COLORS.low}}/>Green: predicted benign / low score</span>
            <span>READ / WRITE / other event types label the connections</span>
          </div>
          <p className={styles.note}>
            {data?.scope} Only recorded IP addresses are displayed. Orange is a visual warning band, not an additional trained class.{" "}
            {data?.truncated
              ? "Additional entities/events were omitted by the display bounds."
              : ""}
          </p>
        </>
      )}
    </section>
  );
}
