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
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; padding: 2rem; }
        h1 { margin-bottom: 1.5rem; color: #1a1a2e; }
        h2 { margin: 1.5rem 0 0.75rem; color: #16213e; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
        th, td { padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        .stat { display: inline-block; margin-right: 2rem; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: #0a3d62; }
        .stat-label { font-size: 0.85rem; color: #666; }
        .significant { color: #27ae60; font-weight: 600; }
        .not-significant { color: #e67e22; font-weight: 600; }
        .btn { padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem; }
        .btn-primary { background: #0a3d62; color: white; }
        .btn-primary:hover { background: #0c4b7a; }
        #runs-list, #comparison-view { display: none; }
        .loading { color: #999; font-style: italic; }
        .error { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>llm-eval-bench Dashboard</h1>

    <div class="card">
        <h2>Evaluation Runs</h2>
        <button class="btn btn-primary" onclick="loadRuns()">Load Runs</button>
        <div id="runs-list">
            <table>
                <thead>
                    <tr><th>ID</th><th>Name</th><th>Models</th><th>Status</th><th>Created</th><th>Actions</th></tr>
                </thead>
                <tbody id="runs-tbody"></tbody>
            </table>
        </div>
    </div>

    <div id="comparison-view" class="card">
        <h2>Model Comparison</h2>
        <div id="comparison-content"></div>
    </div>

    <script>
        async function loadRuns() {
            const tbody = document.getElementById('runs-tbody');
            tbody.innerHTML = '<tr><td colspan="6" class="loading">Loading...</td></tr>';
            document.getElementById('runs-list').style.display = 'block';
            try {
                const resp = await fetch('/api/runs');
                const runs = await resp.json();
                if (runs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6">No runs yet</td></tr>';
                    return;
                }
                tbody.innerHTML = runs.map(r => `
                    <tr>
                        <td>${r.id}</td>
                        <td>${r.name || '-'}</td>
                        <td>${r.models.join(', ')}</td>
                        <td>${r.status}</td>
                        <td>${new Date(r.created_at).toLocaleString()}</td>
                        <td><button class="btn btn-primary" onclick="loadComparison(${r.id})">Compare</button></td>
                    </tr>
                `).join('');
            } catch (e) {
                tbody.innerHTML = `<tr><td colspan="6" class="error">Error: ${e.message}</td></tr>`;
            }
        }

        async function loadComparison(runId) {
            const view = document.getElementById('comparison-view');
            const content = document.getElementById('comparison-content');
            view.style.display = 'block';
            content.innerHTML = '<p class="loading">Loading comparison...</p>';
            try {
                const resp = await fetch(`/api/compare/${runId}`);
                if (!resp.ok) {
                    const err = await resp.json();
                    content.innerHTML = `<p class="error">${err.detail}</p>`;
                    return;
                }
                const data = await resp.json();
                let html = '<h3>Per-Model Results</h3><table><thead><tr><th>Model</th>';
                data.evaluators.forEach(ev => { html += `<th>${ev} (mean ± CI)</th>`; });
                html += '<th>Avg Latency</th><th>Total Cost</th></tr></thead><tbody>';
                data.models.forEach(model => {
                    const stats = data.model_stats[model];
                    html += `<tr><td><strong>${model}</strong></td>`;
                    data.evaluators.forEach(ev => {
                        if (stats[ev]) {
                            const s = stats[ev];
                            html += `<td>${(s.mean * 100).toFixed(1)}% (${(s.lower * 100).toFixed(1)}–${(s.upper * 100).toFixed(1)}%)</td>`;
                        } else {
                            html += '<td>-</td>';
                        }
                    });
                    const t = stats.tracking || {};
                    html += `<td>${(t.avg_latency_ms || 0).toFixed(0)}ms</td>`;
                    html += `<td>$${(t.total_cost || 0).toFixed(4)}</td>`;
                    html += '</tr>';
                });
                html += '</tbody></table>';
                if (data.comparison) {
                    const c = data.comparison;
                    const sigClass = c.is_significant ? 'significant' : 'not-significant';
                    html += `
                        <h3 style="margin-top:1.5rem">Statistical Comparison</h3>
                        <div style="margin-top:0.75rem">
                            <div class="stat"><span class="stat-label">p-value</span><br><span class="stat-value">${c.p_value.toFixed(4)}</span></div>
                            <div class="stat"><span class="stat-label">Mean Difference</span><br><span class="stat-value">${(c.mean_diff * 100).toFixed(2)}%</span></div>
                            <div class="stat"><span class="stat-label">Result</span><br><span class="${sigClass}">${c.interpretation}</span></div>
                        </div>`;
                }
                content.innerHTML = html;
            } catch (e) {
                content.innerHTML = `<p class="error">Error: ${e.message}</p>`;
            }
        }
    </script>
</body>
</html>"""


def mount_dashboard(app: FastAPI) -> None:
    """Mount the dashboard route on a FastAPI app."""

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return DASHBOARD_HTML
