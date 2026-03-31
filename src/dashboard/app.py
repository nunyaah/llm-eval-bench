from fastapi import FastAPI
from fastapi.responses import HTMLResponse


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>llm-eval-bench Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; padding: 2rem; }
        h1 { margin-bottom: 1.5rem; color: #1a1a2e; font-size: 1.6rem; }
        h2 { margin: 0 0 0.75rem; color: #16213e; font-size: 1.1rem; }
        h3 { margin: 1.25rem 0 0.5rem; color: #16213e; font-size: 1rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
        table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.9rem; }
        th, td { padding: 0.55rem 0.85rem; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #555; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
        tr:last-child td { border-bottom: none; }
        .kpi-row { display: flex; flex-wrap: wrap; gap: 1.25rem; margin-bottom: 0.5rem; }
        .kpi { min-width: 120px; }
        .kpi-value { font-size: 1.4rem; font-weight: 700; color: #0a3d62; }
        .kpi-label { font-size: 0.78rem; color: #888; margin-top: 2px; }
        .badge { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
        .badge-green { background: #d4edda; color: #155724; }
        .badge-orange { background: #fff3cd; color: #856404; }
        .badge-blue { background: #cce5ff; color: #004085; }
        .badge-red { background: #f8d7da; color: #721c24; }
        .tag-warn { color: #856404; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 0.4rem 0.75rem; font-size: 0.82rem; margin-bottom: 0.75rem; display: inline-block; }
        .btn { padding: 0.45rem 1rem; border: none; border-radius: 5px; cursor: pointer; font-size: 0.88rem; font-weight: 500; transition: background 0.15s; }
        .btn-primary { background: #0a3d62; color: white; }
        .btn-primary:hover { background: #0c4b7a; }
        .btn-sm { padding: 0.3rem 0.7rem; font-size: 0.8rem; }
        #run-detail { display: none; }
        .loading { color: #999; font-style: italic; }
        .error { color: #e74c3c; }
        .correct { color: #27ae60; }
        .wrong { color: #e74c3c; }
        .mono { font-family: 'Courier New', monospace; font-size: 0.82rem; }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
        .run-title { font-size: 0.85rem; color: #777; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <h1>llm-eval-bench Dashboard</h1>

    <div class="card" id="runs-card">
        <h2>Evaluation Runs</h2>
        <button class="btn btn-primary" onclick="loadRuns()" style="margin-bottom:0.75rem">Load Runs</button>
        <div id="runs-list" style="display:none">
            <table>
                <thead>
                    <tr>
                        <th>ID</th><th>Name</th><th>Dataset</th><th>Models</th>
                        <th>Metric</th><th>Samples</th><th>Status</th><th>Created</th><th></th>
                    </tr>
                </thead>
                <tbody id="runs-tbody"></tbody>
            </table>
        </div>
    </div>

    <div id="run-detail">
        <!-- Section 1: Run summary -->
        <div class="card" id="section-summary">
            <div class="section-header">
                <h2>Run Summary</h2>
                <button class="btn btn-sm" onclick="document.getElementById('run-detail').style.display='none'">✕ Close</button>
            </div>
            <div class="run-title" id="run-title"></div>
            <div class="kpi-row" id="run-kpis"></div>
        </div>

        <!-- Section 2: Model comparison table -->
        <div class="card">
            <h2>Model Scores</h2>
            <div id="model-table"></div>
        </div>

        <!-- Section 3: Statistical result -->
        <div class="card" id="section-stats">
            <h2>Statistical Result</h2>
            <div id="stats-content"></div>
        </div>

        <!-- Section 4: Per-sample failures -->
        <div class="card">
            <h2>Per-Sample Failures</h2>
            <div id="failures-content"><p class="loading">Loading failures...</p></div>
        </div>
    </div>

    <script>
        async function loadRuns() {
            const tbody = document.getElementById('runs-tbody');
            tbody.innerHTML = '<tr><td colspan="9" class="loading">Loading...</td></tr>';
            document.getElementById('runs-list').style.display = 'block';
            try {
                const resp = await fetch('/api/runs');
                const runs = await resp.json();
                if (runs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9">No runs yet.</td></tr>';
                    return;
                }
                tbody.innerHTML = runs.map(r => `
                    <tr>
                        <td>${r.id}</td>
                        <td>${r.name || '-'}</td>
                        <td class="mono">${r.dataset_path}</td>
                        <td>${r.models.join(', ')}</td>
                        <td>${r.primary_metric || r.evaluators[0] || '-'}</td>
                        <td>${r.sample_count ?? '-'}</td>
                        <td><span class="badge ${r.status === 'completed' ? 'badge-green' : r.status === 'failed' ? 'badge-red' : 'badge-orange'}">${r.status}</span></td>
                        <td>${new Date(r.created_at).toLocaleString()}</td>
                        <td><button class="btn btn-primary btn-sm" onclick="loadRunDetail(${r.id})">Inspect</button></td>
                    </tr>
                `).join('');
            } catch (e) {
                tbody.innerHTML = `<tr><td colspan="9" class="error">Error: ${e.message}</td></tr>`;
            }
        }

        async function loadRunDetail(runId) {
            document.getElementById('run-detail').style.display = 'block';
            document.getElementById('run-kpis').innerHTML = '<p class="loading">Loading...</p>';
            document.getElementById('model-table').innerHTML = '';
            document.getElementById('stats-content').innerHTML = '';
            document.getElementById('failures-content').innerHTML = '<p class="loading">Loading failures...</p>';

            // Fetch compare (has stats) and results (has per-sample) in parallel
            const [cmpResp, resResp] = await Promise.all([
                fetch('/api/compare/' + runId),
                fetch('/api/results/' + runId),
            ]);

            if (!cmpResp.ok) {
                const err = await cmpResp.json().catch(() => ({detail: cmpResp.statusText}));
                document.getElementById('run-kpis').innerHTML = `<p class="error">${err.detail}</p>`;
                return;
            }

            const cmp = await cmpResp.json();
            const resData = await resResp.json();

            renderSummary(cmp);
            renderModelTable(cmp);
            renderStats(cmp);
            renderFailures(resData, cmp);
        }

        function renderSummary(cmp) {
            const run = cmp.run || {};
            document.getElementById('run-title').textContent =
                `Run #${cmp.run_id}  ·  ${run.name || ''}  ·  ${new Date(run.created_at).toLocaleString()}`;
            const sampleCount = run.sample_count ?? '-';
            document.getElementById('run-kpis').innerHTML = `
                <div class="kpi"><div class="kpi-value">${cmp.run_id}</div><div class="kpi-label">Run ID</div></div>
                <div class="kpi"><div class="kpi-value">${sampleCount}</div><div class="kpi-label">Samples</div></div>
                <div class="kpi"><div class="kpi-value">${cmp.primary_metric || '-'}</div><div class="kpi-label">Primary Metric</div></div>
                <div class="kpi"><div class="kpi-value">${cmp.models.length}</div><div class="kpi-label">Models</div></div>
                <div class="kpi"><div class="kpi-value">${run.dataset_path || '-'}</div><div class="kpi-label">Dataset</div></div>
            `;
        }

        function renderModelTable(cmp) {
            let html = '<table><thead><tr><th>Model</th>';
            cmp.evaluators.forEach(ev => { html += `<th>${ev} (mean)</th><th>95% CI</th>`; });
            html += '<th>Avg Latency</th><th>Total Cost</th></tr></thead><tbody>';

            cmp.models.forEach(model => {
                const stats = cmp.model_stats[model] || {};
                const isWinner = cmp.comparison && cmp.comparison.winner === model;
                html += `<tr><td><strong>${model}</strong>${isWinner ? ' <span class="badge badge-green">winner</span>' : ''}</td>`;
                cmp.evaluators.forEach(ev => {
                    if (stats[ev]) {
                        const s = stats[ev];
                        html += `<td>${(s.mean * 100).toFixed(1)}%</td>`;
                        html += `<td class="mono">${(s.lower * 100).toFixed(1)}–${(s.upper * 100).toFixed(1)}%</td>`;
                    } else {
                        html += '<td>-</td><td>-</td>';
                    }
                });
                const t = (stats.tracking || {});
                html += `<td>${(t.avg_latency_ms || 0).toFixed(0)} ms</td>`;
                html += `<td>$${(t.total_cost || 0).toFixed(5)}</td>`;
                html += '</tr>';
            });
            html += '</tbody></table>';
            document.getElementById('model-table').innerHTML = html;
        }

        function renderStats(cmp) {
            const c = cmp.comparison;
            if (!c) {
                document.getElementById('stats-content').innerHTML = '<p>Not enough models to compare.</p>';
                return;
            }
            const sigClass = c.is_significant ? 'badge-green' : 'badge-orange';
            const sigText = c.is_significant ? 'Significant' : 'Not significant';
            let html = '';
            if (c.warning === 'small_sample_size') {
                html += `<div class="tag-warn">⚠ Small sample size — confidence intervals may be wide.</div>`;
            }
            html += `
                <div class="kpi-row" style="margin-bottom:1rem">
                    <div class="kpi"><div class="kpi-value">${(c.mean_diff * 100).toFixed(2)}%</div><div class="kpi-label">Difference (A−B)</div></div>
                    <div class="kpi"><div class="kpi-value">${c.p_value.toFixed(4)}</div><div class="kpi-label">p-value</div></div>
                    <div class="kpi"><div class="kpi-value"><span class="badge ${sigClass}">${sigText}</span></div><div class="kpi-label">Result</div></div>
                    <div class="kpi"><div class="kpi-value">${c.winner || '-'}</div><div class="kpi-label">Winner</div></div>
                </div>
                <table style="max-width:600px">
                    <tr><th>Model A</th><td>${c.model_a}</td><th>Score</th><td>${(c.mean_a*100).toFixed(1)}%</td></tr>
                    <tr><th>Model B</th><td>${c.model_b}</td><th>Score</th><td>${(c.mean_b*100).toFixed(1)}%</td></tr>
                    <tr><th>Metric</th><td>${c.evaluator}</td><th>n</th><td>${c.n_samples ?? '-'}</td></tr>
                    <tr><th>Method</th><td>${c.comparison_method || 'paired_bootstrap'}</td>
                        <th>Resamples</th><td>${c.n_bootstrap ?? '-'}</td></tr>
                </table>
                <p style="margin-top:0.75rem;font-size:0.9rem;color:#555">→ ${c.interpretation}</p>
            `;
            document.getElementById('stats-content').innerHTML = html;
        }

        function renderFailures(resData, cmp) {
            const results = resData.results || [];
            const primaryEv = cmp.primary_metric || (cmp.evaluators || [])[0] || 'exact_match';

            // Collect failures: rows where any model scored 0 on the primary metric
            const byInput = {};
            results.forEach(r => {
                if (!byInput[r.input]) byInput[r.input] = {};
                byInput[r.input][r.model] = r;
            });

            const failures = Object.values(byInput).filter(group =>
                Object.values(group).some(r => (r.scores || {})[primaryEv] === 0)
            );

            if (failures.length === 0) {
                document.getElementById('failures-content').innerHTML =
                    '<p class="correct">✓ No failures on primary metric.</p>';
                return;
            }

            const models = cmp.models;
            let html = `<p style="margin-bottom:0.75rem;font-size:0.85rem;color:#555">${failures.length} sample(s) with at least one model failing <strong>${primaryEv}</strong>.</p>`;
            html += `<table><thead><tr><th>Question</th><th>Expected</th>`;
            models.forEach(m => { html += `<th>${m}<br><small style="font-weight:normal">(actual / normalized)</small></th>`; });
            html += `</tr></thead><tbody>`;

            failures.slice(0, 50).forEach(group => {
                const first = Object.values(group)[0];
                html += `<tr><td class="mono" style="max-width:250px">${esc(first.input)}</td>`;
                html += `<td class="mono">${esc(first.expected_output)}</td>`;
                models.forEach(m => {
                    const r = group[m] || {};
                    const score = (r.scores || {})[primaryEv];
                    const cls = score === 1 ? 'correct' : score === 0 ? 'wrong' : '';
                    html += `<td class="${cls} mono">${esc(r.actual_output || '-')}<br>${r.normalized_actual ? '<small>' + esc(r.normalized_actual) + '</small>' : ''}</td>`;
                });
                html += '</tr>';
            });

            html += '</tbody></table>';
            document.getElementById('failures-content').innerHTML = html;
        }

        function esc(str) {
            return String(str ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }
    </script>
</body>
</html>"""


def mount_dashboard(app: FastAPI) -> None:
    """Mount the dashboard route on a FastAPI app."""

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return DASHBOARD_HTML

