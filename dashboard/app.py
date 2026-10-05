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

RESULTS_FILE = "reports/phase10_public_benchmark/results_official_public.jsonl"
PUBLIC_EVAL_FILE = "data/public/eval_public.jsonl"

def process_results():
    stats = {
        "questions": {}
    }

    # Load 100 public questions
    if os.path.exists(PUBLIC_EVAL_FILE):
        with open(PUBLIC_EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    qid = rec["qid"]
                    stats["questions"][qid] = {
                        "qid": qid,
                        "question": rec["question"],
                        "RAG": None,
                        "GraphRAG": None,
                        "AgenticGraphRAG": None
                    }
                except json.JSONDecodeError:
                    pass

    # Load benchmark results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("question_id")
                    pipeline = rec.get("pipeline")

                    if qid in stats["questions"] and pipeline in ["RAG", "GraphRAG", "AgenticGraphRAG"]:
                        stats["questions"][qid][pipeline] = rec
                except json.JSONDecodeError:
                    pass

    return stats

def generate_index_html():
    stats = process_results()

    # Serialize stats for JS
    def json_default(obj):
        return str(obj)
    stats_json = json.dumps(stats, default=json_default)

    head_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agentic GraphRAG Benchmark</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; color: #333; }
        h1, h2, h3, h4 { color: #222; margin-top: 0; }
        .header { margin-bottom: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header h1 { margin-bottom: 5px; }
        .header h3 { color: #666; font-weight: normal; margin-bottom: 20px; }
        .overview-stats { display: flex; gap: 20px; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #ddd; flex: 1; text-align: center; }
        .stat-box .number { font-size: 24px; font-weight: bold; color: #007bff; }
        .stat-box .label { font-size: 14px; color: #555; text-transform: uppercase; margin-top: 5px; }

        .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .selector-area { margin-bottom: 20px; }
        select { width: 100%; padding: 10px; font-size: 16px; border-radius: 4px; border: 1px solid #ccc; }

        .row { display: flex; gap: 20px; flex-wrap: wrap; }
        .col { flex: 1; min-width: 300px; }
        .pipeline-card { background: white; padding: 15px; border-radius: 8px; border: 1px solid #ddd; display: flex; flex-direction: column; }
        .pipeline-card h2 { text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }

        table { width: 100%; border-collapse: collapse; margin-top: 10px; background: white; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f8f9fa; }
        tr:hover { background-color: #f1f1f1; }

        .status-badge { padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
        .status-complete { background: #d4edda; color: #155724; }
        .status-pending { background: #fff3cd; color: #856404; }
        .status-failed { background: #f8d7da; color: #721c24; }

        details { margin-top: 15px; background: #f8f9fa; border-radius: 5px; border: 1px solid #ddd; }
        summary { font-weight: bold; padding: 10px; cursor: pointer; user-select: none; }
        details > div { padding: 15px; border-top: 1px solid #ddd; font-size: 14px; }

        pre { white-space: pre-wrap; word-wrap: break-word; background: #eee; padding: 10px; border-radius: 4px; font-size: 13px; max-height: 400px; overflow-y: auto; }
        ul { margin-top: 5px; padding-left: 20px; }
        li { margin-bottom: 5px; }

        .tabs { margin-bottom: 20px; }
        .tab-btn { padding: 10px 20px; cursor: pointer; font-size: 16px; border: none; background: #ddd; margin-right: 5px; border-radius: 5px 5px 0 0; }
        .tab-btn.active { background: #007bff; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .btn { padding: 8px 12px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; font-size: 14px; }
        .btn:hover { background: #0056b3; }

        /* Live Demo Specific */
        #live input, #live select, #live textarea { width: 100%; padding: 8px; margin: 5px 0 15px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    </style>
</head>
<body>
    <div class="tabs">
        <button class="tab-btn active" onclick="showTab('benchmark')">Phase 10 Benchmark</button>
        <button class="tab-btn" onclick="showTab('live')">Live Interactive Demo</button>
    </div>

    <div id="benchmark" class="tab-content active">
        <div class="header">
            <h1>Agentic GraphRAG Benchmark</h1>
            <h3>RAG vs GraphRAG vs Agentic GraphRAG</h3>
            <div class="overview-stats">
                <div class="stat-box">
                    <div class="number" id="stat-public">100</div>
                    <div class="label">Public Questions</div>
                </div>
                <div class="stat-box">
                    <div class="number" id="stat-completed">0</div>
                    <div class="label">Completed Pipelines</div>
                </div>
                <div class="stat-box">
                    <div class="number" id="stat-status">In Progress</div>
                    <div class="label">Benchmark Status</div>
                </div>
            </div>
        </div>

        <div class="card selector-area">
            <h2>Select a Question</h2>
            <select id="question-selector" onchange="renderSelectedQuestion()"></select>
            <div id="selected-question-text" style="margin-top: 15px; font-size: 18px; font-weight: bold;"></div>
        </div>

        <div class="row" id="pipeline-cards">
            <!-- RAG -->
            <div class="col pipeline-card" id="card-rag">
                <h2>RAG</h2>
                <div class="content">Select a question...</div>
            </div>
            <!-- GraphRAG -->
            <div class="col pipeline-card" id="card-graphrag">
                <h2>GraphRAG</h2>
                <div class="content">Select a question...</div>
            </div>
            <!-- Agentic -->
            <div class="col pipeline-card" id="card-agentic">
                <h2>Agentic GraphRAG</h2>
                <div class="content">Select a question...</div>
            </div>
        </div>

        <div class="card" id="metrics-card" style="display:none;">
            <h2>Metrics Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pipeline</th>
                        <th>Input Tokens</th>
                        <th>Output Tokens</th>
                        <th>Total Tokens</th>
                        <th>Latency (s)</th>
                    </tr>
                </thead>
                <tbody id="metrics-body"></tbody>
            </table>
        </div>

        <div class="card">
            <h2>All Public Questions</h2>
            <table id="questions-table">
                <thead>
                    <tr>
                        <th>QID</th>
                        <th style="width: 40%;">Question</th>
                        <th>RAG</th>
                        <th>GraphRAG</th>
                        <th>Agentic</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <div id="live" class="tab-content">
        <!-- Preserve existing Live Demo UI -->
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
            <button class="btn" id="runBtn" onclick="runLiveQuery()" style="padding:10px 15px; font-size:16px;">RUN ALL THREE</button>
            <div id="status" class="status-msg" style="margin-top: 10px; font-weight:bold; color:#d9534f;"></div>
        </div>

        <div id="resultsArea" style="display: none;">
            <h2 style="text-align:center;">Results (<span id="used_provider"></span> / <span id="used_model"></span>)</h2>
            <div class="row">
                <div class="col pipeline-card" id="res_rag"><h2>RAG</h2><div class="content" style="padding:15px;">Waiting...</div></div>
                <div class="col pipeline-card" id="res_graphrag"><h2>GraphRAG</h2><div class="content" style="padding:15px;">Waiting...</div></div>
                <div class="col pipeline-card" id="res_agentic"><h2>Agentic GraphRAG</h2><div class="content" style="padding:15px;">Waiting...</div></div>
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
        const stats = """ + stats_json + """;
        const questionsList = Object.values(stats.questions);

        function escapeHtml(unsafe) {
            if (unsafe == null) return "N/A";
            return String(unsafe)
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }

        function initDashboard() {
            // Populate overview
            document.getElementById('stat-public').innerText = questionsList.length;
            let completedPipelines = 0;
            questionsList.forEach(q => {
                if (q.RAG && q.RAG.status === 'success') completedPipelines++;
                if (q.GraphRAG && q.GraphRAG.status === 'success') completedPipelines++;
                if (q.AgenticGraphRAG && q.AgenticGraphRAG.status === 'success') completedPipelines++;
            });
            document.getElementById('stat-completed').innerText = completedPipelines;

            if (completedPipelines >= questionsList.length * 3 && questionsList.length > 0) {
                document.getElementById('stat-status').innerText = 'Complete';
                document.getElementById('stat-status').style.color = '#155724';
            }

            // Populate selector
            const sel = document.getElementById('question-selector');
            sel.innerHTML = '<option value="">-- Select a Question --</option>';
            questionsList.forEach(q => {
                const opt = document.createElement('option');
                opt.value = q.qid;
                opt.innerText = `[${q.qid}] ${q.question.substring(0, 100)}...`;
                sel.appendChild(opt);
            });

            // Populate table
            const tbody = document.querySelector('#questions-table tbody');
            tbody.innerHTML = '';
            questionsList.forEach(q => {
                const tr = document.createElement('tr');

                const sRag = q.RAG ? (q.RAG.status === 'success' ? 'complete' : 'failed') : 'pending';
                const sGr = q.GraphRAG ? (q.GraphRAG.status === 'success' ? 'complete' : 'failed') : 'pending';
                const sAg = q.AgenticGraphRAG ? (q.AgenticGraphRAG.status === 'success' ? 'complete' : 'failed') : 'pending';

                tr.innerHTML = `
                    <td>${escapeHtml(q.qid)}</td>
                    <td>${escapeHtml(q.question)}</td>
                    <td><span class="status-badge status-${sRag}">${sRag.toUpperCase()}</span></td>
                    <td><span class="status-badge status-${sGr}">${sGr.toUpperCase()}</span></td>
                    <td><span class="status-badge status-${sAg}">${sAg.toUpperCase()}</span></td>
                    <td><button class="btn" onclick="selectQuestion('${q.qid}')">View</button></td>
                `;
                tbody.appendChild(tr);
            });
        }

        function selectQuestion(qid) {
            document.getElementById('question-selector').value = qid;
            renderSelectedQuestion();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function renderSelectedQuestion() {
            const qid = document.getElementById('question-selector').value;
            if (!qid) {
                document.getElementById('selected-question-text').innerText = '';
                document.getElementById('card-rag').querySelector('.content').innerHTML = 'Select a question...';
                document.getElementById('card-graphrag').querySelector('.content').innerHTML = 'Select a question...';
                document.getElementById('card-agentic').querySelector('.content').innerHTML = 'Select a question...';
                document.getElementById('metrics-card').style.display = 'none';
                return;
            }

            const q = stats.questions[qid];
            document.getElementById('selected-question-text').innerText = q.question;

            renderPipeline('RAG', q.RAG, 'card-rag');
            renderPipeline('GraphRAG', q.GraphRAG, 'card-graphrag');
            renderPipeline('Agentic GraphRAG', q.AgenticGraphRAG, 'card-agentic');

            // Render metrics table
            document.getElementById('metrics-card').style.display = 'block';
            const mBody = document.getElementById('metrics-body');
            mBody.innerHTML = '';

            ['RAG', 'GraphRAG', 'Agentic GraphRAG'].forEach(pName => {
                const key = pName === 'Agentic GraphRAG' ? 'AgenticGraphRAG' : pName;
                const res = q[key];

                let iTok = 'N/A', oTok = 'N/A', tTok = 'N/A', lat = 'N/A';
                if (res && res.status === 'success') {
                    iTok = res.prompt_tokens ?? 'N/A';
                    oTok = res.output_tokens ?? 'N/A';
                    tTok = res.total_tokens ?? 'N/A';
                    lat = (res.latency_s != null) ? Number(res.latency_s).toFixed(2) : 'N/A';
                }

                mBody.innerHTML += `
                    <tr>
                        <td><strong>${pName}</strong></td>
                        <td>${iTok}</td>
                        <td>${oTok}</td>
                        <td>${tTok}</td>
                        <td>${lat}</td>
                    </tr>
                `;
            });
        }

        function renderPipeline(name, data, cardId) {
            const container = document.getElementById(cardId).querySelector('.content');
            if (!data) {
                container.innerHTML = `<div class="status-badge status-pending" style="display:inline-block;">Pending benchmark result</div>`;
                return;
            }
            if (data.status !== 'success') {
                container.innerHTML = `
                    <div class="status-badge status-failed" style="display:inline-block; margin-bottom:10px;">FAILED</div>
                    <pre style="color:red;">${escapeHtml(data.error)}</pre>
                `;
                return;
            }

            let html = `<h4>Answer</h4><pre>${escapeHtml(data.answer)}</pre>`;

            if (name === 'RAG') {
                if (data.citations && data.citations.length > 0) {
                    html += `<details><summary>Evidence / Citations</summary><div><ul>`;
                    data.citations.forEach(c => {
                         html += `<li>[${escapeHtml(c.doc_id)}] ${escapeHtml(c.title)}</li>`;
                    });
                    html += `</ul></div></details>`;
                }
            }

            if (name === 'GraphRAG') {
                html += `<details><summary>GraphRAG Evidence / Provenance</summary><div>`;

                html += `<ul>
                    <li>Selected Chunks: ${escapeHtml(data.selected_chunks ?? (data.trace?.selected_chunks) ?? 'Not available')}</li>
                    <li>Graph Candidate Chunks: ${escapeHtml(data.graph_candidate_chunks ?? (data.trace?.graph_candidate_chunks) ?? 'Not available')}</li>
                    <li>Related Document Count: ${escapeHtml(data.related_document_count ?? (data.trace?.related_document_count) ?? 'Not available')}</li>
                    <li>Entity Count: ${escapeHtml(data.entity_count ?? (data.trace?.entity_count) ?? 'Not available')}</li>
                </ul>`;

                const prov = data.graph_provenance || (data.trace?.graph_provenance);
                if (prov && prov.length > 0) {
                    html += `<h5>Graph Paths (Provenance)</h5><ul>`;
                    prov.forEach(p => {
                        html += `<li><code>${escapeHtml(p.graph_path)}</code><br/>
                        <small>Doc: ${escapeHtml(p.doc_id)} | Chunk: ${escapeHtml(p.chunk_id)} | Entities: ${escapeHtml((p.entities_used || []).join(', '))}</small>
                        <br/><i>${escapeHtml(p.text)}</i>
                        </li>`;
                    });
                    html += `</ul>`;
                }
                html += `</div></details>`;
            }

            if (name === 'Agentic GraphRAG') {
                html += `<details><summary>Agentic Trace</summary><div>`;

                html += `<ul>
                    <li>Stopping Reason: ${escapeHtml(data.stopping_reason ?? (data.trace?.stopping_reason) ?? 'Not available')}</li>
                    <li>Tools/Methods Used: ${escapeHtml((data.tools_used ?? (data.trace?.tools_used) ?? []).join(', ') || 'Not available')}</li>
                </ul>`;

                const steps = data.steps || (data.trace?.steps);
                if (steps && steps.length > 0) {
                    html += `<h5>Reasoning / Controller Steps</h5><ol>`;
                    steps.forEach(s => {
                        html += `<li><strong>Action:</strong> ${escapeHtml(s.action)}<br/>`;
                        if (s.result && s.result.strategy_changes && s.result.strategy_changes.length > 0) {
                            html += `<em>Strategy Changes:</em> <ul>`;
                            s.result.strategy_changes.forEach(sc => {
                                html += `<li>${escapeHtml(sc)}</li>`;
                            });
                            html += `</ul>`;
                        }
                        if (s.result && s.result.evidence_found !== undefined) {
                            html += `<em>Evidence found:</em> ${s.result.evidence_found}<br/>`;
                        }
                        html += `</li>`;
                    });
                    html += `</ol>`;
                }

                const ev = data.evidence_history?.collected || data.trace?.evidence_history?.collected || [];
                if (ev.length > 0) {
                    html += `<h5>Evidence / Chunks</h5><ul>`;
                    ev.forEach(c => {
                        html += `<li>[${escapeHtml(c.doc_id)}] ${escapeHtml(c.title)}</li>`;
                    });
                    html += `</ul>`;
                }

                html += `</div></details>`;
            }

            container.innerHTML = html;
        }

        /* Tabs Logic */
        function showTab(id) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.currentTarget.classList.add('active');
        }

        /* Live Demo Logic */
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

                renderLiveCard('rag', data.results.RAG, data.evaluations.RAG, 'rag');
                renderLiveCard('graphrag', data.results.GraphRAG, data.evaluations.GraphRAG, 'gr');
                renderLiveCard('agentic', data.results.AgenticGraphRAG, data.evaluations.AgenticGraphRAG, 'ag');

            } catch(e) {
                document.getElementById('status').innerText = 'Error: ' + e.message;
            }
            document.getElementById('runBtn').disabled = false;
        }

        function renderLiveCard(id, res, evalData, sumId) {
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

            let html = `<h4>Status</h4><p>SUCCESS</p>`;
            html += `<h4>Answer</h4><p>${escapeHtml(res.answer || 'No answer')}</p>`;

            let lat = Number(res.latency_s) || 0;
            let tok = Number(res.total_tokens) || 0;

            document.getElementById('sum_stat_' + sumId).innerText = 'SUCCESS';
            document.getElementById('sum_lat_' + sumId).innerText = lat.toFixed(2);
            document.getElementById('sum_tok_' + sumId).innerText = tok;

            if(evalData && Object.keys(evalData).length > 0) {
                if(evalData.error) {
                    html += `<h4>Evaluation</h4><p>Error: ${escapeHtml(evalData.error)}</p>`;
                    document.getElementById('sum_eval_' + sumId).innerText = 'Error';
                } else {
                    let evalStr = `Score: ${evalData.score}/100, Verdict: ${evalData.verdict}, Grounded: ${evalData.grounded}`;
                    html += `<h4>Evaluation</h4><p>${evalStr}</p>`;
                    document.getElementById('sum_eval_' + sumId).innerText = evalData.score !== undefined ? evalData.score : 'N/A';
                }
            } else {
                document.getElementById('sum_eval_' + sumId).innerText = 'Not evaluated';
            }
            div.innerHTML = html;
        }

        window.onload = initDashboard;
    </script>
</body>
</html>
"""
    return head_html

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
