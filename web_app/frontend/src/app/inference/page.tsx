"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import styles from "./page.module.css";

type Metrics = { tn: number | null; fp: number | null; fn: number | null; tp: number | null; precision: number | null; recall: number | null; f1: number | null; mcc: number | null; ap: number | null; roc_auc: number | null; scored_nodes: number; positive_support: number; unscored_truth: number[]; confusion_matrix: number[][] | null };
type Evaluation = { metrics: Metrics; threshold: number | null; unit?: string; calibration?: { supported: boolean; unit?: string } };
type Model = { name: string; epochs: number; checkpoint_epoch: number; checkpoint_sha256: string; parameters: number; batch_size: number; seed: number; events: Record<string, number>; model_config: Record<string, unknown>; evaluation: Record<string, Evaluation>; relations: string[]; relation_confusion: { labels: string[]; matrix: number[][] }; history: { epoch: number; training_loss: number; selection_ce: number }[] };
type Replay = { events: number; inference_ms: number; mean_relation_ce: number; scope: string; items: { index: number; src: number; dst: number; relation: number; predicted_relation: number; relation_ce: number; pair_anomaly: number }[] };
const format = (value: number | null | undefined, percent = false) => value == null ? "Unavailable" : percent ? `${(value * 100).toFixed(2)}%` : value.toLocaleString(undefined, { maximumFractionDigits: 6 });
async function request<T>(path: string, method = "GET"): Promise<T> {
  const response = await fetch(`/api/inference/${path}`, { method });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || "Unable to load model");
  return body;
}

export default function InferencePage() {
  const [model, setModel] = useState<Model | null>(null);
  const [method, setMethod] = useState("paper_threshold");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [replay, setReplay] = useState<Replay | null>(null);
  useEffect(() => {
    let active = true;
    request<Model>("model").then(value => { if (active) setModel(value); }).catch(reason => { if (active) setError(String(reason.message)); });
    return () => { active = false; };
  }, []);
  async function run() {
    setBusy(true); setError(""); setReplay(null);
    try { setReplay(await request<Replay>("replay", "POST")); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Replay failed"); }
    finally { setBusy(false); }
  }
  const evaluation = model?.evaluation[method];
  const metrics = evaluation?.metrics;
  const config = model?.model_config;
  return <main className={styles.page}>
    <nav><Link href="/">← CADETS workspace</Link><span>Model / Inference</span></nav>
    <header><p className={styles.eyebrow}>TRAINED CHECKPOINT</p><h1>{model?.name || "TGNN Final V3 · 30 epochs"}</h1><p>Saved model architecture, held-out CADETS evaluation, and checkpoint inference.</p></header>
    {error && <p role="alert" className={styles.notice}>{error}</p>}
    {!model && !error && <p role="status">Loading saved model…</p>}
    {model && <>
      <section className={styles.cards} aria-label="Training details">{[["Training epochs", model.epochs], ["Selected epoch", model.checkpoint_epoch], ["Parameters", model.parameters], ["Batch size", model.batch_size], ["Seed", model.seed], ["Test events", model.events.test]].map(([label, value]) => <article key={label}><span>{label}</span><strong>{Number(value).toLocaleString()}</strong></article>)}</section>
      <section className={styles.panel}><h2>Evaluation metrics</h2>
        <label>Scoring method <select value={method} onChange={event => setMethod(event.target.value)}>{Object.keys(model.evaluation).map(key => <option key={key} value={key}>{key.replaceAll("_", " ")}</option>)}</select></label>
        <p>Held-out capture-node evaluation. Each method has its own predictions and threshold.</p>
        <p>Threshold: <strong>{format(evaluation?.threshold)}</strong> · Unit: {evaluation?.unit || evaluation?.calibration?.unit || "capture node"}</p>
        {evaluation?.calibration?.supported === false && <p className={styles.notice}>Calibration is unsupported: no complete validation horizons were available. Threshold, alert metrics, and confusion matrix are unavailable.</p>}
        <div className={styles.cards}>{(["precision", "recall", "f1", "mcc", "ap", "roc_auc"] as const).map(key => <article key={key}><span>{key === "ap" ? "Average precision" : key.replaceAll("_", " ").toUpperCase()}</span><strong>{format(metrics?.[key], key !== "mcc")}</strong></article>)}</div>
        <div className={styles.columns}><div><h3>Confusion matrix</h3>{metrics?.confusion_matrix ? <table><caption>Rows: actual · Columns: predicted</caption><thead><tr><th>Node class</th><th>Benign</th><th>Malicious</th></tr></thead><tbody><tr><th scope="row">Benign</th><td>TN {format(metrics.tn)}</td><td>FP {format(metrics.fp)}</td></tr><tr><th scope="row">Malicious</th><td>FN {format(metrics.fn)}</td><td>TP {format(metrics.tp)}</td></tr></tbody></table> : <p>Confusion matrix unavailable for this method.</p>}</div>
          <div><h3>Evaluation coverage</h3><p>{format(metrics?.scored_nodes)} scored nodes · {format(metrics?.positive_support)} scored malicious nodes</p><p>{metrics?.unscored_truth.length ?? 0} ground-truth nodes outside the scoring universe.</p><p>Scores are anomaly evidence. Capture-level metrics do not describe individual event labels or live node-horizon false-alert rates.</p></div></div>
      </section>
      <section className={styles.panel}><h2>Test relation confusion matrix</h2><p>Recorded across all {model.events.test.toLocaleString()} test events. Rows are actual relations; columns are predicted relations.</p><div className={styles.scroll}><table><thead><tr><th>Actual / Predicted</th>{model.relation_confusion.labels.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{model.relation_confusion.matrix.map((row, i) => <tr key={i}><th scope="row">{model.relation_confusion.labels[i]}</th>{row.map((value, j) => <td key={j} style={i === j ? { fontWeight: 700, background: "#9573c922" } : undefined}>{value.toLocaleString()}</td>)}</tr>)}</tbody></table></div></section>
      <section className={styles.panel}><h2>Saved model architecture</h2><p>Read from this checkpoint’s configuration and source snapshot.</p>
        <ol className={styles.architecture}>{[
          ["Typed token encoder", `${config?.encoder_dim} dimensions · ordered two-hash embeddings`],
          ["Hybrid event window", `${config?.window} events · ${config?.predecessor_slots} predecessor slots`],
          ["Event-line graph attention", `${config?.gat_layers} GAT layers · ${config?.window_output_dim} output dimensions`],
          ["Causal CNN and short memory", `${config?.short_slots} slots · ${config?.short_recent} recent events · ${config?.short_policy}`],
          ["Long memory", `${config?.memory_policy} · ${config?.read_gate} read gate`],
          ["Relation and pair heads", `${config?.num_relations} relations · ${config?.head_dim} head dimensions · ${config?.pair_context}`],
        ].map(([title, detail]) => <li key={title}><strong>{title}</strong><span>{detail}</span></li>)}</ol>
        <details><summary>Full saved configuration</summary><pre>{JSON.stringify(config, null, 2)}</pre></details><p>Checkpoint SHA-256</p><code className={styles.hash}>{model.checkpoint_sha256}</code>
      </section>
      <section className={styles.panel}><h2>Run checkpoint inference</h2><p>Replay the first 1,024 prepared CADETS test events on CPU with fresh model state. The preview reports relation loss and pair anomaly; full-capture node metrics are shown above.</p>
        <button onClick={run} disabled={busy}>{busy ? "Running checkpoint…" : "Run 1,024 test events"}</button>
        {replay && <><p role="status">{replay.events.toLocaleString()} events · {format(replay.inference_ms)} ms · Mean relation CE {format(replay.mean_relation_ce)}</p><p>{replay.scope}</p><div className={styles.scroll}><table><caption>First 32 replayed events</caption><thead><tr>{["Event", "Source", "Destination", "Observed relation", "Predicted relation", "Relation CE", "Pair anomaly"].map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{replay.items.map(item => <tr key={item.index}><td>{item.index}</td><td>{item.src}</td><td>{item.dst}</td><td>{model.relations[item.relation]}</td><td>{model.relations[item.predicted_relation]}</td><td>{format(item.relation_ce)}</td><td>{format(item.pair_anomaly)}</td></tr>)}</tbody></table></div></>}
      </section>
      <section className={styles.panel}><h2>Training history</h2><div className={styles.scroll}><table><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation selection CE</th></tr></thead><tbody>{model.history.map(row => <tr key={row.epoch}><td>{row.epoch}</td><td>{format(row.training_loss)}</td><td>{format(row.selection_ce)}</td></tr>)}</tbody></table></div></section>
    </>}
  </main>;
}
