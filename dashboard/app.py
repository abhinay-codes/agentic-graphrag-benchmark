import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statistics
import http.server
import socketserver
import threading
import uuid
import time
import urllib.parse

RESULTS_FILE = "reports/phase11_public_benchmark/results_public_100.jsonl"

def process_results():
    stats = {
        "raw_records": 0, "historical_errors": 0, "duplicates": 0, "unique_successes": 0,
        "pipelines": {
            "RAG": {"records": [], "latencies": [], "total_tokens": [], "prompt_tokens": [], "eval_tokens": []},
            "GraphRAG": {"records": [], "latencies": [], "total_tokens": [], "prompt_tokens": [], "eval_tokens": []},
            "AgenticGraphRAG": {"records": [], "latencies": [], "total_tokens": [], "prompt_tokens": [], "eval_tokens": [], "steps": [], "actions": {}, "stopping_reasons": {}}
        },
        "questions": {}
    }

    if not os.path.exists(RESULTS_FILE): return stats
    successful_pairs = {}
    import json
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                stats["raw_records"] += 1
                q_id = rec.get("question_id")
                pipeline = rec.get("pipeline")
                status = rec.get("status")

                if status != "success":
                    stats["historical_errors"] += 1
                    continue
                if (q_id, pipeline) in successful_pairs:
                    stats["duplicates"] += 1
                    continue
                successful_pairs[(q_id, pipeline)] = rec

                if q_id not in stats["questions"]: stats["questions"][q_id] = {"question": rec.get("question", ""), "RAG": {}, "GraphRAG": {}, "AgenticGraphRAG": {}}
                stats["questions"][q_id][pipeline] = rec

                p_stats = stats["pipelines"].get(pipeline)
                if not p_stats: continue
                p_stats["records"].append(rec)

                duration = rec.get("latency_s")
                if duration is not None:
                    p_stats["latencies"].append(duration)
                    stats["questions"][q_id][pipeline]["latency"] = duration

                total = rec.get("total_tokens")
                if total is not None:
                    p_stats["total_tokens"].append(total)
                    stats["questions"][q_id][pipeline]["tokens"] = total

                pt = rec.get("prompt_tokens")
                et = rec.get("output_tokens")
                if pt is not None: p_stats["prompt_tokens"].append(pt)
                if et is not None: p_stats["eval_tokens"].append(et)

                if pipeline == "AgenticGraphRAG":
                    steps = rec.get("steps")
                    if steps is not None and isinstance(steps, list):
                        p_stats["steps"].append(len(steps))

                    actions = rec.get("action_sequence")
                    if actions is not None and isinstance(actions, list):
                        for act in actions:
                            p_stats["actions"][act] = p_stats["actions"].get(act, 0) + 1

                    stop_reason = rec.get("stopping_reason")
                    if stop_reason is not None:
                        p_stats["stopping_reasons"][stop_reason] = p_stats["stopping_reasons"].get(stop_reason, 0) + 1
            except: pass
    stats["unique_successes"] = len(successful_pairs)
    return stats

def safe_mean(lst): return sum(lst)/len(lst) if lst else 0
def safe_median(lst): return statistics.median(lst) if lst else 0

def generate_benchmark_html(stats):
    html = f"""
    <div class="row">
        <div class="col card">
            <h2>Benchmark Coverage</h2>
            <p><strong>100</strong> public questions</p>
            <p><strong>300</strong> pipeline/question evaluations</p>
            <p><strong>{stats['unique_successes']}</strong> successful evaluations</p>
            <p><strong>{300 - stats['unique_successes']}</strong> missing successful pairs</p>
            <div class="bar-container"><div class="bar" style="width: {(stats['unique_successes']/300)*100}%;"></div></div>
        </div>
        <div class="col card">
            <h2>Historical Failures</h2>
            <p><strong>Raw records:</strong> {stats['raw_records']}</p>
            <p><strong>Historical error records:</strong> {stats['historical_errors']}</p>
            <p><strong>Duplicate pairs:</strong> {stats['duplicates']}</p>
            <p><strong>Unique successful pairs:</strong> {stats['unique_successes']}</p>
        </div>
    </div>

    <div class="card">
        <h2>Pipeline Comparison</h2>
        <table>
            <tr><th>Metric</th><th>RAG</th><th>GraphRAG</th><th>Agentic GraphRAG</th></tr>
            <tr><td>Successful Evaluations</td><td>{len(stats['pipelines']['RAG']['records'])}/100</td><td>{len(stats['pipelines']['GraphRAG']['records'])}/100</td><td>{len(stats['pipelines']['AgenticGraphRAG']['records'])}/100</td></tr>
            <tr><td>Mean Latency (s)</td><td>{safe_mean(stats['pipelines']['RAG']['latencies']):.2f}</td><td>{safe_mean(stats['pipelines']['GraphRAG']['latencies']):.2f}</td><td>{safe_mean(stats['pipelines']['AgenticGraphRAG']['latencies']):.2f}</td></tr>
            <tr><td>Median Latency (s)</td><td>{safe_median(stats['pipelines']['RAG']['latencies']):.2f}</td><td>{safe_median(stats['pipelines']['GraphRAG']['latencies']):.2f}</td><td>{safe_median(stats['pipelines']['AgenticGraphRAG']['latencies']):.2f}</td></tr>
            <tr><td>Mean Prompt Tokens</td><td>{safe_mean(stats['pipelines']['RAG']['prompt_tokens']):.0f}</td><td>{safe_mean(stats['pipelines']['GraphRAG']['prompt_tokens']):.0f}</td><td>{safe_mean(stats['pipelines']['AgenticGraphRAG']['prompt_tokens']):.0f}</td></tr>
            <tr><td>Mean Output Tokens</td><td>{safe_mean(stats['pipelines']['RAG']['eval_tokens']):.0f}</td><td>{safe_mean(stats['pipelines']['GraphRAG']['eval_tokens']):.0f}</td><td>{safe_mean(stats['pipelines']['AgenticGraphRAG']['eval_tokens']):.0f}</td></tr>
            <tr><td>Mean Total Tokens</td><td>{safe_mean(stats['pipelines']['RAG']['total_tokens']):.0f}</td><td>{safe_mean(stats['pipelines']['GraphRAG']['total_tokens']):.0f}</td><td>{safe_mean(stats['pipelines']['AgenticGraphRAG']['total_tokens']):.0f}</td></tr>
        </table>
    </div>

    <div class="row">
        <div class="col card">
            <h2>Agentic-Specific Metrics</h2>
            <p><strong>Mean Steps:</strong> {safe_mean(stats['pipelines']['AgenticGraphRAG']['steps']):.2f}</p>
            <p><strong>Median Steps:</strong> {safe_median(stats['pipelines']['AgenticGraphRAG']['steps'])}</p>
            <h3>Action Frequencies</h3><ul>{"".join([f"<li>{k}: {v}</li>" for k,v in stats['pipelines']['AgenticGraphRAG']['actions'].items()])}</ul>
            <h3>Stopping Reasons</h3><ul>{"".join([f"<li>{k}: {v}</li>" for k,v in stats['pipelines']['AgenticGraphRAG']['stopping_reasons'].items()])}</ul>
        </div>
        <div class="col card">
            <h2>Retrieval & Evidence</h2>
            <p>Provenance citations are strictly verified for Agentic Pipeline.</p>
            <p>Number of Agentic runs with valid traces: {sum(1 for x in stats['pipelines']['AgenticGraphRAG']['records'] if x.get("action_sequence") is not None)}</p>
            <canvas id="latencyChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>Question-Level View</h2>
    """
    for q_id, data in sorted(stats['questions'].items()):
        html += f"""
        <details>
            <summary>{q_id}: {data['question'].replace('<', '&lt;').replace('>', '&gt;')}</summary>
            <table>
                <tr><th>Pipeline</th><th>Latency (s)</th><th>Tokens</th><th>Answer Snapshot</th></tr>
                <tr><td>RAG</td><td>{round(data['RAG'].get('latency', 0), 2) if isinstance(data['RAG'].get('latency'), (int, float)) else 'N/A'}</td><td>{data['RAG'].get('tokens', 'N/A')}</td><td><pre>{str(data['RAG'].get('answer', ''))[:200].replace('<', '&lt;').replace('>', '&gt;')}...</pre></td></tr>
                <tr><td>GraphRAG</td><td>{round(data['GraphRAG'].get('latency', 0), 2) if isinstance(data['GraphRAG'].get('latency'), (int, float)) else 'N/A'}</td><td>{data['GraphRAG'].get('tokens', 'N/A')}</td><td><pre>{str(data['GraphRAG'].get('answer', ''))[:200].replace('<', '&lt;').replace('>', '&gt;')}...</pre></td></tr>
                <tr><td>AgenticGraphRAG</td><td>{round(data['AgenticGraphRAG'].get('latency', 0), 2) if isinstance(data['AgenticGraphRAG'].get('latency'), (int, float)) else 'N/A'}</td><td>{data['AgenticGraphRAG'].get('tokens', 'N/A')}</td><td><pre>{str(data['AgenticGraphRAG'].get('answer', ''))[:200].replace('<', '&lt;').replace('>', '&gt;')}...</pre></td></tr>
            </table>
        </details>"""
    html += "</div>"

    # Chart Script
    html += f"""
    <script>
        var ctx = document.getElementById('latencyChart');
        if(ctx) {{
            new Chart(ctx.getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: ['RAG', 'GraphRAG', 'Agentic GraphRAG'],
                    datasets: [{{
                        label: 'Mean Latency (s)',
                        data: [{safe_mean(stats['pipelines']['RAG']['latencies']):.2f}, {safe_mean(stats['pipelines']['GraphRAG']['latencies']):.2f}, {safe_mean(stats['pipelines']['AgenticGraphRAG']['latencies']):.2f}],
                        backgroundColor: ['#36a2eb', '#ff6384', '#4bc0c0']
                    }}]
                }}
            }});
        }}
    </script>
    """
    return html

def generate_index_html():
    stats = process_results()
    bench_html = generate_benchmark_html(stats)

    head_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gate 6 & Demo Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
        h1, h2, h3 { color: #333; }
        .card { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; vertical-align: top; }
        th { background-color: #f8f9fa; }
        .row { display: flex; gap: 20px; flex-wrap: wrap; }
        .col { flex: 1; min-width: 300px; }
        .bar-container { width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden; margin-top: 5px; }
        .bar { height: 20px; background-color: #4CAF50; }
        pre { white-space: pre-wrap; word-wrap: break-word; background: #eee; padding: 10px; border-radius: 4px; font-size: 12px; }
        details { margin-bottom: 10px; padding: 10px; background: white; border-radius: 5px; border: 1px solid #ccc; }
        summary { font-weight: bold; cursor: pointer; }
        .tabs { margin-bottom: 20px; }
        .tab-btn { padding: 10px 20px; cursor: pointer; font-size: 16px; border: none; background: #ddd; margin-right: 5px; border-radius: 5px 5px 0 0; }
        .tab-btn.active { background: #4CAF50; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .btn { padding: 10px 15px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; }
        .btn:hover { background: #0056b3; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        input, select, textarea { width: 100%; padding: 8px; margin: 5px 0 15px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        .status-msg { font-weight: bold; color: #d9534f; }
    </style>
</head>
<body>
    <h1>Benchmark & Live Comparison</h1>
    <div class="tabs">
        <button class="tab-btn active" onclick="showTab('benchmark')">Phase 11 Benchmark</button>
        <button class="tab-btn" onclick="showTab('live')">Live Interactive Demo</button>
    </div>
"""

    live_html = """
    <div id="benchmark" class="tab-content active">
        <!-- BENCH_HTML -->
    </div>

    <div id="live" class="tab-content">
        <div class="card">
            <h2>LLM Configuration (Applies to all 3 pipelines)</h2>
            <div class="row">
                <div class="col">
                    <label>Provider:</label>
                    <select id="provider" onchange="updateModels()">
                        <option value="ollama">Local Ollama</option>
                        <option value="openai">OpenAI</option>
                        <option value="gemini">Google Gemini</option>
                        <option value="anthropic">Anthropic Claude</option>
                    </select>
                </div>
                <div class="col">
                    <label>Model:</label>
                    <input type="text" id="model" value="qwen3:8b">
                </div>
                <div class="col">
                    <label>API Key (Kept local/server-side):</label>
                    <input type="password" id="api_key">
                </div>
                <div class="col">
                    <label>Base URL (Ollama only):</label>
                    <input type="text" id="base_url" value="http://localhost:11434">
                </div>
            </div>
            <p><em>Note: The same provider and model will be used across RAG, GraphRAG, and Agentic GraphRAG to ensure a fair architectural comparison.</em></p>
        </div>

        <div class="card">
            <h2>Live Query</h2>
            <label>Question:</label>
            <textarea id="question" rows="3" placeholder="Enter your question here..."></textarea>
            <label>Optional Reference Answer (for LLM Judge Evaluation):</label>
            <textarea id="reference" rows="2" placeholder="Enter a reference answer if you want automated evaluation..."></textarea>
            <button class="btn" id="runBtn" onclick="runLiveQuery()">RUN ALL THREE</button>
            <div id="status" class="status-msg" style="margin-top: 10px;"></div>
        </div>

        <div id="resultsArea" style="display: none;">
            <h2 style="text-align:center;">Results (<span id="used_provider"></span> / <span id="used_model"></span>)</h2>
            <div class="row">
                <div class="col card" id="res_rag"><h3>RAG</h3><div class="content">Waiting...</div></div>
                <div class="col card" id="res_graphrag"><h3>GraphRAG</h3><div class="content">Waiting...</div></div>
                <div class="col card" id="res_agentic"><h3>Agentic GraphRAG</h3><div class="content">Waiting...</div></div>
            </div>

            <div class="card">
                <h2>Comparison Summary</h2>
                <table id="summaryTable">
                    <tr><th>Metric</th><th>RAG</th><th>GraphRAG</th><th>Agentic GraphRAG</th></tr>
                    <tr><td>Status</td><td id="sum_stat_rag"></td><td id="sum_stat_gr"></td><td id="sum_stat_ag"></td></tr>
                    <tr><td>Latency (s)</td><td id="sum_lat_rag"></td><td id="sum_lat_gr"></td><td id="sum_lat_ag"></td></tr>
                    <tr><td>Total Tokens</td><td id="sum_tok_rag"></td><td id="sum_tok_gr"></td><td id="sum_tok_ag"></td></tr>
                    <tr><td>Judge Score</td><td id="sum_eval_rag"></td><td id="sum_eval_gr"></td><td id="sum_eval_ag"></td></tr>
                    <tr><td>Cost</td><td colspan="3">Local/Provided API ??? no extra platform cost</td></tr>
                </table>
            </div>
        </div>
    </div>

    <script>
        function showTab(id) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.currentTarget.classList.add('active');
        }

        function updateModels() {
            const p = document.getElementById('provider').value;
            const m = document.getElementById('model');
            if(p === 'ollama') m.value = 'qwen3:8b';
            else if(p === 'openai') m.value = 'gpt-4o-mini';
            else if(p === 'gemini') m.value = 'gemini-1.5-flash';
            else if(p === 'anthropic') m.value = 'claude-3-haiku-20240307';
        }

        async function runLiveQuery() {
            const q = document.getElementById('question').value;
            if(!q) return alert("Please enter a question.");

            document.getElementById('runBtn').disabled = true;
            document.getElementById('resultsArea').style.display = 'block';
            document.getElementById('status').innerText = 'Initializing...';

            document.getElementById('used_provider').innerText = document.getElementById('provider').value;
            document.getElementById('used_model').innerText = document.getElementById('model').value;

            ['rag', 'graphrag', 'agentic'].forEach(p => {
                document.querySelector('#res_' + p + ' .content').innerHTML = 'Waiting...';
                document.getElementById('sum_stat_' + (p==='graphrag'?'gr':(p==='agentic'?'ag':p))).innerText = '';
                document.getElementById('sum_lat_' + (p==='graphrag'?'gr':(p==='agentic'?'ag':p))).innerText = '';
                document.getElementById('sum_tok_' + (p==='graphrag'?'gr':(p==='agentic'?'ag':p))).innerText = '';
                document.getElementById('sum_eval_' + (p==='graphrag'?'gr':(p==='agentic'?'ag':p))).innerText = '';
            });

            const payload = {
                question: q,
                reference_answer: document.getElementById('reference').value,
                provider: document.getElementById('provider').value,
                model: document.getElementById('model').value,
                api_key: document.getElementById('api_key').value,
                base_url: document.getElementById('base_url').value
            };

            try {
                document.getElementById('status').innerText = 'Running sequentially: RAG -> GraphRAG -> Agentic GraphRAG... (Please wait)';

                const res = await fetch('/api/live-query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                if(!res.ok) {
                    document.getElementById('status').innerText = 'Error: ' + (data.error || 'Unknown server error');
                    document.getElementById('runBtn').disabled = false;
                    return;
                }

                document.getElementById('status').innerText = 'Complete!';

                // Render RAG
                renderCard('rag', data.results.RAG, data.evaluations.RAG, 'rag');
                // Render GraphRAG
                renderCard('graphrag', data.results.GraphRAG, data.evaluations.GraphRAG, 'gr');
                // Render Agentic
                renderCard('agentic', data.results.AgenticGraphRAG, data.evaluations.AgenticGraphRAG, 'ag');

            } catch(e) {
                document.getElementById('status').innerText = 'Error: ' + e.message;
            }

            document.getElementById('runBtn').disabled = false;
        }

        function escapeHtml(unsafe) {
            return String(unsafe)
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
         }

        function renderCard(id, res, evalData, sumId) {
            const div = document.querySelector('#res_' + id + ' .content');
            if(!res) {
                div.innerHTML = `<p style="color:red">No result returned</p>`;
                document.getElementById('sum_stat_' + sumId).innerText = 'ERROR';
                return;
            }
            if(res.error) {
                div.innerHTML = `<p style="color:red">Error: ${escapeHtml(res.error)}</p>`;
                document.getElementById('sum_stat_' + sumId).innerText = 'ERROR';
                return;
            }

            let html = `<h4>Answer</h4><p>${escapeHtml(res.answer || 'No answer')}</p>`;

            // Use normalized latency_s field (set by backend normalization)
            let lat = Number(res.latency_s) || 0;
            let tok = Number(res.total_tokens) || 0;

            document.getElementById('sum_stat_' + sumId).innerText = 'SUCCESS';
            document.getElementById('sum_lat_' + sumId).innerText = lat.toFixed(2);
            document.getElementById('sum_tok_' + sumId).innerText = tok;

            html += `<h4>Metrics</h4><ul>
                <li>Latency: ${lat.toFixed(2)}s</li>
                <li>Total Tokens: ${tok}</li>
            </ul>`;

            if(evalData && Object.keys(evalData).length > 0) {
                if(evalData.error) {
                    html += `<h4>Evaluation</h4><p>Error: ${escapeHtml(evalData.error)}</p>`;
                    document.getElementById('sum_eval_' + sumId).innerText = 'Error';
                } else {
                    let evalStr = `Score: ${evalData.score}/100, Verdict: ${evalData.verdict}, Grounded: ${evalData.grounded}`;
                    html += `<h4>Evaluation</h4><p>${evalStr}</p>`;
                    if(evalData.reason) html += `<p><em>Reason: ${escapeHtml(evalData.reason)}</em></p>`;
                    document.getElementById('sum_eval_' + sumId).innerText = evalData.score !== undefined ? evalData.score : 'N/A';
                }
            } else {
                document.getElementById('sum_eval_' + sumId).innerText = 'Not evaluated';
            }

            if(id === 'agentic' && res.trace) {
                let steps = res.trace.steps || [];
                let stopReason = res.trace.stopping_reason || 'unknown';
                html += `<h4>Agentic Telemetry</h4><ul>
                    <li>Steps: ${steps.length}</li>`;
                if(steps.length > 0) {
                    html += `<li>Actions: ${steps.map(s => escapeHtml(s.action)).join(' &rarr; ')}</li>`;
                }
                html += `<li>Stopping Reason: ${escapeHtml(stopReason)}</li>
                </ul>`;
            }

            div.innerHTML = html;
        }
    </script>
</body>
</html>
"""
    return head_html + live_html.replace("<!-- BENCH_HTML -->", bench_html)

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        html = generate_index_html()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/live-query":
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)

                question = data.get("question")
                reference_answer = data.get("reference_answer")
                provider = data.get("provider", "ollama")
                model = data.get("model", "qwen3:8b")
                api_key = data.get("api_key")
                base_url = data.get("base_url")

                from src.llm.factory import get_llm_client
                try:
                    llm = get_llm_client(provider=provider, base_url=base_url, api_key=api_key, model=model)
                except Exception as e:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    import re as _re
                    err_msg = _re.sub(r'api_key=[^\s,]+', 'api_key="***"', str(e))
                    self.wfile.write(json.dumps({"error": f"LLM init error: {err_msg}"}).encode("utf-8"))
                    return

                from src.pipelines.rag import VectorRAGPipeline
                from src.pipelines.graphrag import GraphRAGPipeline
                from src.pipelines.agentic_graphrag import AgenticGraphRAGPipeline

                rag = VectorRAGPipeline(llm_client=llm)
                graphrag = GraphRAGPipeline(llm_client=llm)
                agentic = AgenticGraphRAGPipeline(llm_client=llm)

                q_id = f"live-{uuid.uuid4().hex[:8]}"
                raw_results = {}

                try: raw_results["RAG"] = rag.answer(q_id, question)
                except Exception as e: raw_results["RAG"] = {"error": str(e), "status": "error"}

                try: raw_results["GraphRAG"] = graphrag.answer(q_id, question)
                except Exception as e: raw_results["GraphRAG"] = {"error": str(e), "status": "error"}

                try: raw_results["AgenticGraphRAG"] = agentic.answer(q_id, question)
                except Exception as e: raw_results["AgenticGraphRAG"] = {"error": str(e), "status": "error"}

                # ---- Normalize results into common schema ----
                import re as _re
                results = {}
                for p_name, p_res in raw_results.items():
                    if "error" in p_res:
                        results[p_name] = p_res
                        continue

                    # Strip <think> tags from answer
                    answer = p_res.get("answer", "")
                    if isinstance(answer, str):
                        answer = _re.sub(r'<think>.*?</think>', '', answer, flags=_re.DOTALL).strip()

                    trace = p_res.get("trace", {})

                    # Extract latency from the correct schema location
                    latency = 0.0
                    if "total_pipeline_duration_s" in trace:
                        latency = trace["total_pipeline_duration_s"]
                    elif "total_pipeline_duration" in trace:
                        latency = trace["total_pipeline_duration"]
                    elif "timing" in trace and "total_s" in trace.get("timing", {}):
                        latency = trace["timing"]["total_s"]

                    # Extract tokens from the correct schema location
                    total_tokens = 0
                    if "token_usage" in trace and "total_tokens" in trace.get("token_usage", {}):
                        total_tokens = trace["token_usage"]["total_tokens"]
                    elif "total_tokens" in trace:
                        total_tokens = trace["total_tokens"] or 0

                    normalized = {
                        "question_id": p_res.get("question_id", q_id),
                        "answer": answer,
                        "latency_s": latency,
                        "total_tokens": total_tokens,
                        "citations": p_res.get("citations", []),
                        "status": "success"
                    }

                    # Agentic-specific telemetry
                    if p_name == "AgenticGraphRAG" and trace:
                        steps = trace.get("steps", [])
                        action_sequence = [s.get("action", "unknown") for s in steps]

                        # Determine stopping reason from controller, not from evaluator parse errors
                        raw_stopping_reason = trace.get("stopping_reason", "unknown")

                        # If the action sequence contains an explicit "stop" action
                        # and the stopping reason contains evaluator parse warnings,
                        # use a clean reason instead
                        has_stop_action = "stop" in action_sequence
                        if has_stop_action:
                            # Check if the reason is just an evaluator parse error echoed by the controller
                            parse_error_phrases = [
                                "failed to parse evaluator",
                                "unable to confirm evidence sufficiency",
                                "parse_error"
                            ]
                            is_parse_error_reason = any(
                                phrase in raw_stopping_reason.lower()
                                for phrase in parse_error_phrases
                            )
                            if is_parse_error_reason:
                                raw_stopping_reason = "Stopped after evidence evaluation (controller decided to stop)"

                        normalized["trace"] = {
                            "steps": steps,
                            "stopping_reason": raw_stopping_reason,
                            "tools_used": trace.get("tools_used", []),
                            "evidence_history": trace.get("evidence_history", {}),
                            "token_usage": trace.get("token_usage", {}),
                            "timing": trace.get("timing", {}),
                        }

                    results[p_name] = normalized

                # ---- Judge Evaluation ----
                evaluations = {}
                if reference_answer and reference_answer.strip():
                    for p_name, p_res in results.items():
                        if "error" in p_res:
                            continue
                        generated_answer = p_res.get("answer", "")
                        prompt = (
                            "You are a strict factual correctness judge.\n\n"
                            "TASK: Determine if the Generated Answer correctly answers the Question "
                            "when compared to the Reference Answer.\n\n"
                            f"Question: {question}\n"
                            f"Reference Answer: {reference_answer}\n"
                            f"Generated Answer: {generated_answer}\n\n"
                            "RULES:\n"
                            "- If the Generated Answer contains or derives the same factual content as the Reference Answer, score it highly.\n"
                            "- If the Generated Answer says 'insufficient evidence', 'not available', or refuses to answer, "
                            "it MUST receive a FAIL verdict and a low score because it did NOT provide the correct answer.\n"
                            "- A correct refusal is still a wrong answer relative to the reference.\n"
                            "- 'grounded' means the answer only uses information from the provided evidence (not whether it matches the reference).\n"
                            "- 'evidence_supported' means the evidence actually contained the answer.\n\n"
                            "Return ONLY a JSON object with these exact keys:\n"
                            '{"verdict": "PASS or FAIL", "score": 0-100, "grounded": true/false, '
                            '"evidence_supported": true/false, "reason": "brief explanation"}\n\n'
                            "JSON Output:"
                        )
                        try:
                            eval_res = llm.generate(prompt, temperature=0.0, options={"num_predict": 2048})
                            text = eval_res.get("response", "")
                            # Strip <think> from judge output
                            text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
                            s = text.find('{')
                            e = text.rfind('}') + 1
                            if s >= 0 and e > s:
                                evaluations[p_name] = json.loads(text[s:e])
                            else:
                                evaluations[p_name] = {"error": "Failed to parse judge output"}
                        except Exception as ex:
                            evaluations[p_name] = {"error": str(ex)}

                response = {"question": question, "results": results, "evaluations": evaluations}

                def json_default(obj):
                    return str(obj)

                json_str = json.dumps(response, default=json_default)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json_str.encode("utf-8"))

            except Exception as e:
                import traceback
                import re as _re
                tb = traceback.format_exc()

                try:
                    print("Internal Server Error:", str(e).encode('ascii', 'replace').decode('ascii'))
                except:
                    pass

                tb = _re.sub(r'api_key=[^\s,]+', 'api_key="***"', tb)
                tb = _re.sub(r'password=[^\s,]+', 'password="***"', tb)
                tb = _re.sub(r'secret=[^\s,]+', 'secret="***"', tb)

                try:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    err_msg = _re.sub(r'api_key=[^\s,]+', 'api_key="***"', str(e))
                    err_msg = _re.sub(r'password=[^\s,]+', 'password="***"', err_msg)
                    err_msg = _re.sub(r'secret=[^\s,]+', 'secret="***"', err_msg)
                    self.wfile.write(json.dumps({
                        "error": err_msg,
                        "traceback": "Check server logs for details."
                    }).encode("utf-8"))
                except:
                    pass

def run_server():
    server = socketserver.TCPServer(("", 8080), DashboardHandler)
    server.allow_reuse_address = True
    print("Dashboard running on http://localhost:8080")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
