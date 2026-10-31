"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Crosshair, Maximize2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import styles from "./graph3d.module.css";
import { nodeRisk, RISK_COLORS, WARNING_RATIO } from "./graph-risk";
export type SpatialNode = {
    id: number;
    alert: boolean;
    score: number;
    ip: string | null;
    attributes: Record<string, unknown>;
};
type Edge = {
    src: number;
    dst: number;
    relation_name?: string;
};
type Point = {
    x: number;
    y: number;
    z: number;
};
type Camera = {
    yaw: number;
    pitch: number;
    zoom: number;
    panX: number;
    panY: number;
    target: Point;
};
type Projected = {
    id: number;
    x: number;
    y: number;
    depth: number;
    radius: number;
};
const initialCamera = (): Camera => ({ yaw: .24, pitch: -.16, zoom: 1, panX: 0, panY: 0, target: { x: 0, y: 0, z: 0 } });
export const spatialLabel = (n: SpatialNode) => n.ip || String(n.attributes.cmd_line || n.attributes.path || n.attributes.host || n.attributes.hostname || n.attributes.name || `${n.attributes.type || "Entity"} #${n.id}`);
function tint(color: string, target: string, amount: number) {
    const channels = [1, 3, 5].map(offset => Math.round(
        parseInt(color.slice(offset, offset + 2), 16) * (1 - amount) +
        parseInt(target.slice(offset, offset + 2), 16) * amount));
    return `rgb(${channels.join(",")})`;
}
function category(n: SpatialNode) {
    if (n.ip || n.attributes.type === "netflow")
        return { name: "Network", color: "#24bf9b" };
    if (n.attributes.type === "subject" || n.attributes.type === "process")
        return { name: "Process", color: "#e2b438" };
    if (n.attributes.type === "file")
        return { name: "File", color: "#588bcb" };
    if (n.attributes.host || n.attributes.hostname)
        return { name: "Host", color: "#a381d3" };
    return { name: "Other", color: "#8a99ac" };
}
// Solve once in three dimensions, then freeze all positions. Only the camera moves.
function spatialLayout(nodes: SpatialNode[], edges: Edge[]) {
    const ordered = [...nodes].sort((a, b) => a.id - b.id);
    const index = new Map(ordered.map((n, i) => [n.id, i]));
    const unique = [...new Map(edges.filter(e => e.src !== e.dst).map(e => [`${Math.min(e.src, e.dst)}:${Math.max(e.src, e.dst)}`, e])).values()];
    const degree = new Map(nodes.map(n => [n.id, 0]));
    unique.forEach(e => { degree.set(e.src, (degree.get(e.src) || 0) + 1); degree.set(e.dst, (degree.get(e.dst) || 0) + 1); });
    // Deterministic modularity communities separate densely related neighborhoods.
    // These are visual communities of observed edges, not attack classifications.
    const adjacency = ordered.map(() => new Set<number>());
    for (const e of unique) {
        const a = index.get(e.src), b = index.get(e.dst);
        if (a !== undefined && b !== undefined) {
            adjacency[a].add(b);
            adjacency[b].add(a);
        }
    }
    const membership = ordered.map((_, i) => i), totals = adjacency.map(a => a.size), mass = Math.max(1, totals.reduce((a, b) => a + b, 0));
    for (let pass = 0; pass < 25; pass++) {
        let changed = false;
        for (let i = 0; i < ordered.length; i++) {
            const k = adjacency[i].size, previous = membership[i];
            totals[previous] -= k;
            const weights = new Map<number, number>();
            for (const j of adjacency[i])
                weights.set(membership[j], (weights.get(membership[j]) || 0) + 1);
            let best = previous, gain = (weights.get(previous) || 0) - 1.35 * k * totals[previous] / mass;
            for (const [community, weight] of weights) {
                const score = weight - 1.35 * k * totals[community] / mass;
                if (score > gain + 1e-8) {
                    gain = score;
                    best = community;
                }
            }
            membership[i] = best;
            totals[best] += k;
            changed ||= best !== previous;
        }
        if (!changed)
            break;
    }
    const communities = [...new Set(membership)], clusterIndex = new Map(communities.map((c, i) => [c, i]));
    const centers = communities.map((_, i) => { const t = i * 2.399963, z = 1 - 2 * (i + .5) / Math.max(1, communities.length), r = Math.sqrt(1 - z * z); return { x: Math.cos(t) * r * 240, y: Math.sin(t) * r * 240, z: z * 240 }; });
    const points = ordered.map((n, i) => {
        const t = i * 2.399963, z = 1 - 2 * (i + .5) / Math.max(1, ordered.length), r = Math.sqrt(1 - z * z);
        const c = centers[clusterIndex.get(membership[i])!];
        return { x: c.x + Math.cos(t) * r * 45, y: c.y + Math.sin(t) * r * 45, z: c.z + z * 45 };
    });
    const links = unique.map(e => [index.get(e.src), index.get(e.dst)]).filter((p): p is [
        number,
        number
    ] => p[0] !== undefined && p[1] !== undefined);
    for (let tick = 0; tick < 180; tick++) {
        const forces = points.map((p, i) => { const c = centers[clusterIndex.get(membership[i])!]; return { x: (c.x - p.x) * .025, y: (c.y - p.y) * .025, z: (c.z - p.z) * .025 }; });
        for (let i = 0; i < points.length; i++)
            for (let j = i + 1; j < points.length; j++) {
                const a = points[i], b = points[j], dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
                const d = Math.max(4, Math.hypot(dx, dy, dz)), repulsion = 1900 / (d * d);
                for (const [axis, delta] of [["x", dx], ["y", dy], ["z", dz]] as const) {
                    const f = delta / d * repulsion;
                    forces[i][axis] += f;
                    forces[j][axis] -= f;
                }
            }
        for (const [i, j] of links) {
            const a = points[i], b = points[j], dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
            const d = Math.hypot(dx, dy, dz) || 1, same = membership[i] === membership[j], f = (d - (same ? 48 : 190)) * (same ? .035 : .004);
            for (const [axis, delta] of [["x", dx], ["y", dy], ["z", dz]] as const) {
                const v = delta / d * f;
                forces[i][axis] += v;
                forces[j][axis] -= v;
            }
        }
        const cooling = 1 - tick / 220;
        points.forEach((p, i) => { for (const axis of ["x", "y", "z"] as const)
            p[axis] += Math.max(-7, Math.min(7, forces[i][axis])) * cooling; });
    }
    const center = points.reduce((a, p) => ({ x: a.x + p.x / Math.max(1, points.length), y: a.y + p.y / Math.max(1, points.length), z: a.z + p.z / Math.max(1, points.length) }), { x: 0, y: 0, z: 0 });
    const radius = Math.max(60, ...points.map(p => Math.hypot(p.x - center.x, p.y - center.y, p.z - center.z)));
    return { positions: new Map(ordered.map((n, i) => [n.id, { x: (points[i].x - center.x) / radius * 260, y: (points[i].y - center.y) / radius * 260, z: (points[i].z - center.z) / radius * 260 }])), links: unique, degree, communities: communities.length };
}
export function Graph3D({ nodes, edges, selected, onSelect, threshold, thresholdBased = true }: {
    nodes: SpatialNode[];
    edges: Edge[];
    selected: number | null;
    onSelect: (id: number | null) => void;
    threshold: number;
    thresholdBased?: boolean;
}) {
    const wrapper = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
    const camera = useRef(initialCamera()), projected = useRef<Projected[]>([]);
    const invalidate = useRef<() => void>(() => { });
    const [colorMode, setColorMode] = useState("risk");
    const current = useRef({ selected, onSelect, threshold, thresholdBased, colorMode });
    current.current = { selected, onSelect, threshold, thresholdBased, colorMode };
    const [hovered, setHovered] = useState<number | null>(null), [find, setFind] = useState("");
    const scene = useMemo(() => spatialLayout(nodes, edges), [nodes, edges]);
    const hoveredRef = useRef<number | null>(null);
    const directedLinks = useMemo(() => {
        const grouped = new Map<string, { src: number; dst: number; relations: Set<string> }>();
        for (const e of edges) {
            const key = `${e.src}:${e.dst}`;
            const group = grouped.get(key) || { src: e.src, dst: e.dst, relations: new Set<string>() };
            if (e.relation_name) group.relations.add(e.relation_name);
            grouped.set(key, group);
        }
        return [...grouped.values()];
    }, [edges]);
    useEffect(() => { invalidate.current(); }, [selected, threshold, thresholdBased, colorMode]);
    useEffect(() => {
        if (!canvas.current || !wrapper.current)
            return;
        const element: HTMLCanvasElement = canvas.current, host = wrapper.current;
        const context = element.getContext("2d");
        if (!context)
            return;
        const ctx = context;
        let width = 1, height = 1, frame = 0, destroyed = false;
        camera.current = initialCamera();
        hoveredRef.current = null;
        function draw() {
            frame = 0;
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = "#070b0e";
            ctx.fillRect(0, 0, width, height);
            const c = camera.current, cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
            const scale = Math.min(width * .85, height * 1.25) / 650 * c.zoom;
            const coords = new Map<number, Projected>();
            for (const n of nodes) {
                const p = scene.positions.get(n.id)!;
                const px = p.x - c.target.x, py = p.y - c.target.y, pz = p.z - c.target.z;
                const x = px * cy + pz * sy, z = -px * sy + pz * cy, y = py * cp - z * sp, depth = py * sp + z * cp;
                const perspective = 850 / Math.max(140, 850 + depth);
                const radius = (4.8 + Math.min(7, Math.sqrt(scene.degree.get(n.id) || 0) * 1.6)) * scale * perspective * .86;
                coords.set(n.id, { id: n.id, x: width / 2 + c.panX + x * scale * perspective, y: height / 2 + c.panY + y * scale * perspective, depth, radius: Math.max(2.4, radius) });
            }
            projected.current = [...coords.values()].sort((a, b) => b.depth - a.depth);
            const active = current.current.selected ?? hoveredRef.current;
            const neighbors = new Set<number>();
            if (active !== null) {
                neighbors.add(active);
                for (const e of scene.links)
                    if (e.src === active || e.dst === active) {
                        neighbors.add(e.src);
                        neighbors.add(e.dst);
                    }
            }
            const relationLabels: { x: number; y: number; text: string }[] = [];
            for (const e of directedLinks) {
                const a = coords.get(e.src), b = coords.get(e.dst);
                if (!a || !b) continue;
                const connected = e.src === active || e.dst === active;
                ctx.strokeStyle = connected ? "rgba(119,193,195,.46)" : active !== null ? "rgba(66,106,101,.025)" : "rgba(66,113,105,.16)";
                ctx.lineWidth = connected ? 1 : .55;
                ctx.beginPath();
                if (e.src === e.dst) ctx.arc(a.x, a.y - a.radius * 1.3, a.radius * 1.4, .4, Math.PI * 2.8);
                else { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); }
                ctx.stroke();
                if (connected && e.src !== e.dst) {
                    const dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy),angle=Math.atan2(dy,dx);
                    if (d > a.radius+b.radius+24) {
                        const endX=b.x-dx/d*(b.radius+5),endY=b.y-dy/d*(b.radius+5);
                        ctx.fillStyle="#77C1C3";ctx.beginPath();ctx.moveTo(endX,endY);
                        ctx.lineTo(endX-7*Math.cos(angle-.4),endY-7*Math.sin(angle-.4));
                        ctx.lineTo(endX-7*Math.cos(angle+.4),endY-7*Math.sin(angle+.4));ctx.closePath();ctx.fill();
                    }
                    if (d > 90 && relationLabels.length < 12 && e.relations.size) {
                        const names=[...e.relations].sort();
                        const x=a.x+dx*.57,y=a.y+dy*.57;
                        if(!relationLabels.some(p=>Math.abs(p.x-x)<95&&Math.abs(p.y-y)<22))
                            relationLabels.push({x,y,text:names.slice(0,2).join(" / ")+(names.length>2?` +${names.length-2}`:"")});
                    }
                }
            }
            const byId = new Map(nodes.map(n => [n.id, n]));
            const labelBoxes: {
                x: number;
                y: number;
                w: number;
                h: number;
            }[] = [];
            for (const p of projected.current) {
                const n = byId.get(p.id)!, isActive = p.id === active, near = active === null || neighbors.has(p.id), risk = nodeRisk(n, current.current.threshold, current.current.thresholdBased), color = current.current.colorMode === "risk" ? risk.color : category(n).color;
                ctx.globalAlpha = current.current.selected !== null
                    ? (isActive ? 1 : near ? .55 : .045)
                    : (near ? 1 : .18);
                if ((current.current.colorMode === "type" && risk.level === "high") || isActive) {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.radius + 3, 0, Math.PI * 2);
                    ctx.strokeStyle = isActive ? "#D3DFE6" : RISK_COLORS.high;
                    ctx.lineWidth = isActive ? 1.4 : 1;
                    ctx.stroke();
                }
                // A reflected light band and a soft specular patch give the
                // accepted palette a polished surface without drawn highlight arcs.
                const body = ctx.createLinearGradient(p.x, p.y - p.radius, p.x, p.y + p.radius);
                body.addColorStop(0, tint(color, "#DFEDF2", .3));
                body.addColorStop(.22, color);
                body.addColorStop(.48, tint(color, "#08151B", .52));
                body.addColorStop(.56, tint(color, "#08151B", .58));
                body.addColorStop(.66, tint(color, "#D1E6E8", .12));
                body.addColorStop(.82, color);
                body.addColorStop(1, tint(color, "#08151B", .48));
                ctx.fillStyle = body;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();
                const curvature = ctx.createRadialGradient(
                    p.x - p.radius * .15, p.y - p.radius * .2, 0,
                    p.x, p.y, p.radius);
                curvature.addColorStop(0, "rgba(5,12,17,0)");
                curvature.addColorStop(.65, "rgba(5,12,17,0)");
                curvature.addColorStop(1, "rgba(5,12,17,.25)");
                ctx.fillStyle = curvature;
                ctx.fill();
                ctx.strokeStyle = tint(color, "#C8D9D6", .16);
                ctx.lineWidth = .55;
                ctx.stroke();
                ctx.save();
                ctx.translate(p.x - p.radius * .22, p.y - p.radius * .43);
                ctx.rotate(-.35);
                ctx.scale(p.radius * .48, p.radius * .18);
                const reflection = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
                reflection.addColorStop(0, "rgba(235,248,255,.68)");
                reflection.addColorStop(.4, "rgba(235,248,255,.32)");
                reflection.addColorStop(1, "rgba(235,248,255,0)");
                ctx.fillStyle = reflection;
                ctx.beginPath();
                ctx.arc(0, 0, 1, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }
            ctx.globalAlpha=1;
            ctx.font="600 9px Arial";
            for(const label of relationLabels) {
                const w=ctx.measureText(label.text).width;
                ctx.fillStyle="#101B24";ctx.fillRect(label.x-w/2-5,label.y-10,w+10,16);
                ctx.fillStyle="#B7EDFF";ctx.fillText(label.text,label.x-w/2,label.y+1);
            }
            // Label important nodes first, avoiding an unreadable wall of overlapping names.
            const labelOrder = [...projected.current].sort((a, b) => (b.id === active ? 1 : 0) - (a.id === active ? 1 : 0) || (scene.degree.get(b.id) || 0) - (scene.degree.get(a.id) || 0));
            for (const p of labelOrder) {
                const n = byId.get(p.id)!, isActive = p.id === active;
                if (!isActive && (active !== null && !neighbors.has(p.id) || p.radius < 6 || labelBoxes.length >= (active === null ? Math.max(10, Math.round(nodes.length / 25)) : 18)))
                    continue;
                const raw = spatialLabel(n), text = raw.length > 32 ? raw.slice(0, 15) + "…" + raw.slice(-14) : raw;
                ctx.font = `${isActive ? 600 : 400} ${isActive ? 12 : 10}px Arial`;
                const w = ctx.measureText(text).width, x = p.x - w / 2, y = p.y + p.radius + 15;
                if (!isActive && labelBoxes.some(b => x < b.x + b.w && x + w > b.x && y - 11 < b.y + b.h && y > b.y))
                    continue;
                labelBoxes.push({ x: x - 3, y: y - 11, w: w + 6, h: 15 });
                ctx.globalAlpha = isActive ? 1 : current.current.selected !== null ? .5 : .62;
                ctx.fillStyle = isActive ? "#DFE9E5" : "#A3B4AE";
                ctx.fillText(text, x, y);
            }
            ctx.globalAlpha = 1;
        }
        function schedule() { if (!frame && !destroyed)
            frame = requestAnimationFrame(draw); }
        invalidate.current = schedule;
        const observer = new ResizeObserver(entries => {
            width = entries[0].contentRect.width;
            height = entries[0].contentRect.height;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            element.width = width * dpr;
            element.height = height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            schedule();
        });
        observer.observe(host);
        const pointers = new Map<number, {
            x: number;
            y: number;
        }>();
        let drag: {
            x: number;
            y: number;
            distance: number;
            pan: boolean;
        } | null = null;
        function position(e: PointerEvent) { const r = element.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
        function hit(x: number, y: number) { return [...projected.current].reverse().find(p => Math.hypot(x - p.x, y - p.y) < Math.max(10, p.radius + 4))?.id ?? null; }
        function down(e: PointerEvent) { if (e.button !== 0 && e.button !== 2)
            return; element.focus(); element.setPointerCapture(e.pointerId); pointers.set(e.pointerId, position(e)); drag = { ...position(e), distance: 0, pan: e.button === 2 || e.shiftKey }; element.style.cursor = "grabbing"; }
        function move(e: PointerEvent) {
            const pos = position(e), prev = pointers.get(e.pointerId);
            if (!prev || !drag) {
                const id = hit(pos.x, pos.y);
                if (hoveredRef.current !== id) {
                    hoveredRef.current = id;
                    setHovered(id);
                    element.style.cursor = id !== null ? "pointer" : "grab";
                    schedule();
                }
                return;
            }
            const dx = pos.x - prev.x, dy = pos.y - prev.y;
            drag.distance += Math.hypot(dx, dy);
            if (pointers.size === 2) {
                const other = [...pointers.entries()].find(([id]) => id !== e.pointerId)![1];
                const before = Math.hypot(prev.x - other.x, prev.y - other.y), after = Math.hypot(pos.x - other.x, pos.y - other.y);
                camera.current.zoom = Math.max(.25, Math.min(8, camera.current.zoom * after / Math.max(1, before)));
                camera.current.panX += dx / 2;
                camera.current.panY += dy / 2;
            }
            else if (drag.pan) {
                camera.current.panX += dx;
                camera.current.panY += dy;
            }
            else {
                camera.current.yaw += dx * .007;
                camera.current.pitch += dy * .007;
            }
            pointers.set(e.pointerId, pos);
            schedule();
        }
        function up(e: PointerEvent) { const p = position(e); if (drag && drag.distance < 5 && e.button === 0 && pointers.size === 1)
            current.current.onSelect(hit(p.x, p.y)); pointers.delete(e.pointerId); if (!pointers.size)
            drag = null; if (element.hasPointerCapture(e.pointerId))
            element.releasePointerCapture(e.pointerId); element.style.cursor = "grab"; }
        function cancel(e: PointerEvent) { pointers.delete(e.pointerId); drag = null; }
        function leave() { if (!pointers.size) {
            hoveredRef.current = null;
            setHovered(null);
            element.style.cursor = "grab";
            schedule();
        } }
        function wheel(e: WheelEvent) { e.preventDefault(); camera.current.zoom = Math.max(.25, Math.min(8, camera.current.zoom * Math.exp(-e.deltaY * .001))); schedule(); }
        function menu(e: MouseEvent) { e.preventDefault(); }
        function key(e: KeyboardEvent) {
            const c = camera.current;
            if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "0", "Escape"].includes(e.key))
                e.preventDefault();
            if (e.key === "ArrowLeft")
                c.yaw -= .12;
            if (e.key === "ArrowRight")
                c.yaw += .12;
            if (e.key === "ArrowUp")
                c.pitch -= .12;
            if (e.key === "ArrowDown")
                c.pitch += .12;
            if (e.key === "+" || e.key === "=")
                c.zoom = Math.min(8, c.zoom * 1.15);
            if (e.key === "-")
                c.zoom = Math.max(.25, c.zoom / 1.15);
            if (e.key === "0")
                camera.current = initialCamera();
            if (e.key === "Escape")
                current.current.onSelect(null);
            schedule();
        }
        element.addEventListener("pointerleave", leave);
        element.addEventListener("pointerdown", down);
        element.addEventListener("pointermove", move);
        element.addEventListener("pointerup", up);
        element.addEventListener("pointercancel", cancel);
        element.addEventListener("wheel", wheel, { passive: false });
        element.addEventListener("contextmenu", menu);
        element.addEventListener("keydown", key);
        return () => { destroyed = true; cancelAnimationFrame(frame); observer.disconnect(); element.removeEventListener("pointerleave", leave); element.removeEventListener("pointerdown", down); element.removeEventListener("pointermove", move); element.removeEventListener("pointerup", up); element.removeEventListener("pointercancel", cancel); element.removeEventListener("wheel", wheel); element.removeEventListener("contextmenu", menu); element.removeEventListener("keydown", key); invalidate.current = () => { }; };
    }, [nodes, scene, directedLinks]);
    function control(action: string) { const c = camera.current; if (action === "in")
        c.zoom = Math.min(8, c.zoom * 1.25); if (action === "out")
        c.zoom = Math.max(.25, c.zoom / 1.25); if (action === "reset")
        camera.current = initialCamera(); if (action === "focus" && selected !== null) {
        c.target = { ...scene.positions.get(selected)! };
        c.panX = 0;
        c.panY = 0;
        c.zoom = 2.1;
    } invalidate.current(); }
    const hoverNode = nodes.find(n => n.id === (hovered ?? selected));
    return <div className={styles.scene} ref={wrapper}>
    <canvas ref={canvas} className={styles.canvas} tabIndex={0} role="application" aria-label="Interactive 3D entity graph. Drag to orbit, Shift drag to pan, scroll to zoom. Arrow keys rotate, plus and minus zoom, zero resets. Use Select node for keyboard node selection."/>
    <div className={styles.title}><span className={styles.live}/><strong>ENTITY CONSTELLATION</strong><span>3D / FIXED LAYOUT</span></div>
    <div className={styles.tools}>
      <button aria-label="Zoom 3D graph in" title="Zoom in" onClick={() => control("in")}><ZoomIn size={17}/></button>
      <button aria-label="Zoom 3D graph out" title="Zoom out" onClick={() => control("out")}><ZoomOut size={17}/></button>
      <button aria-label="Focus selected node" title="Focus selected node" disabled={selected === null} onClick={() => control("focus")}><Crosshair size={17}/></button>
      <button aria-label="Reset 3D camera" title="Fit graph / reset camera" onClick={() => control("reset")}><RotateCcw size={17}/></button>
      <button aria-label="Fullscreen 3D graph" title="Toggle fullscreen" onClick={() => { if (document.fullscreenElement)
        void document.exitFullscreen();
    else
        void wrapper.current?.requestFullscreen(); }}><Maximize2 size={17}/></button>
    </div>
    <div className={styles.picker}><label>Node colors<select aria-label="Node color mode" value={colorMode} onChange={e=>setColorMode(e.target.value)}><option value="risk">Risk / threshold</option><option value="type">Entity type</option></select></label><label htmlFor="spatial-search">Locate a node</label><input id="spatial-search" value={find} onChange={e => setFind(e.target.value)} placeholder="Search name, IP or ID"/><select aria-label="Select graph node" value={selected ?? ""} onChange={e => onSelect(e.target.value === "" ? null : Number(e.target.value))}><option value="">Select node…</option>{nodes.filter(n => `${spatialLabel(n)} ${n.id}`.toLowerCase().includes(find.toLowerCase())).map(n => <option key={n.id} value={n.id}>{n.alert ? "⚠ " : ""}{spatialLabel(n).slice(0, 65)} · #{n.id} · {nodeRisk(n,threshold,thresholdBased).label}</option>)}</select></div>
    {hoverNode && <div className={styles.hover}>{spatialLabel(hoverNode)}<small>#{hoverNode.id} · {category(hoverNode).name} · {nodeRisk(hoverNode,threshold,thresholdBased).label} · Score {hoverNode.score.toFixed(6)} / threshold {threshold.toFixed(6)}</small></div>}
    {!nodes.length && <div className={styles.empty}>No entities in this neighborhood.</div>}
    <div className={styles.footer}><span>{nodes.length} NODES <i /> {scene.links.length} LINKS <i /> {scene.communities} CLUSTERS</span><span>Drag to orbit · Shift / right drag to pan · Scroll / pinch to zoom</span></div>
    <div className={styles.legend} aria-label="Graph color legend">
      {colorMode === "risk" ? <>
        <span><i style={{background:RISK_COLORS.high}}/>Red · predicted malicious {thresholdBased ? `> ${threshold.toFixed(6)}` : ""}</span>
        {thresholdBased && <span><i style={{background:RISK_COLORS.medium}}/>Orange · medium {(threshold*WARNING_RATIO).toFixed(6)}–{threshold.toFixed(6)}</span>}
        <span><i style={{background:RISK_COLORS.low}}/>Green · predicted benign {thresholdBased ? `< ${(threshold*WARNING_RATIO).toFixed(6)}` : ""}</span>
      </> : <>{[["#24bf9b", "Network"], ["#e2b438", "Process"], ["#588bcb", "File"], ["#a381d3", "Host"], ["#8a99ac", "Other"]].map(([color,name])=><span key={name}><i style={{background:color}}/>{name}</span>)}<span><i style={{background:RISK_COLORS.high}}/>Red ring: predicted malicious</span></>}
      <span>Arrows: event direction · Select a node for READ / WRITE / other relations</span>
    </div>
  </div>;
}
