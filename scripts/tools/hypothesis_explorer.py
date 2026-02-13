#!/usr/bin/env python3
"""
Minimal hypothesis explorer web server.

Usage:
    python hypothesis_explorer.py [--data PATH] [--port PORT]

Default data path: output/hypothesis_evaluation_results.json
Default port: 8080
"""

import argparse
import http.server
import json
import socketserver
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_DATA_PATH = "output/hypothesis_evaluation_results.json"
DEFAULT_PORT = 8080

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hypothesis Explorer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; padding: 20px; background: #f5f5f5; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        .control-group { display: flex; align-items: center; gap: 8px; }
        label { font-weight: 600; font-size: 14px; }
        select { padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }
        .container { display: flex; gap: 20px; }
        .column { flex: 1; background: white; border-radius: 8px; padding: 15px; max-height: calc(100vh - 180px); overflow-y: auto; }
        .column h2 { margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #eee; font-size: 18px; }
        .hypothesis { padding: 12px; margin-bottom: 10px; border: 1px solid #e0e0e0; border-radius: 6px; background: #fafafa; }
        .hypothesis:hover { background: #f0f0f0; }
        .hypothesis-text { font-size: 14px; margin-bottom: 8px; line-height: 1.4; }
        .scores { display: flex; gap: 12px; flex-wrap: wrap; font-size: 12px; color: #666; }
        .score { background: #e8e8e8; padding: 3px 8px; border-radius: 4px; }
        .score.highlight { background: #c8e6c9; font-weight: 600; }
        .count { color: #888; font-size: 14px; }
        .no-data { color: #999; font-style: italic; padding: 20px; text-align: center; }
    </style>
</head>
<body>
    <div class="controls">
        <div class="control-group">
            <label>Dataset:</label>
            <select id="dataset">
                <option value="qwen_em">Qwen EM</option>
                <option value="gemma_gender">Gemma Gender</option>
                <option value="gemini">Gemini</option>
            </select>
        </div>
        <div class="control-group">
            <label>Sort by:</label>
            <select id="metric">
                <option value="test_accuracy">Test Accuracy</option>
                <option value="test_frequency">Test Frequency</option>
                <option value="interestingness_score">Interestingness</option>
                <option value="abstraction_score">Abstraction</option>
            </select>
        </div>
        <div class="control-group">
            <label>Order:</label>
            <select id="order">
                <option value="desc">Decreasing</option>
                <option value="asc">Increasing</option>
            </select>
        </div>
    </div>
    <div class="container">
        <div class="column">
            <h2>SAE Hypotheses <span id="sae-count" class="count"></span></h2>
            <div id="sae-list"></div>
        </div>
        <div class="column">
            <h2>LLM Hypotheses <span id="llm-count" class="count"></span></h2>
            <div id="llm-list"></div>
        </div>
    </div>

    <script>
        let data = null;

        async function loadData() {
            const resp = await fetch('/data');
            data = await resp.json();
            render();
        }

        function formatScore(value, metric) {
            if (value === null || value === undefined) return 'N/A';
            if (metric.includes('accuracy') || metric.includes('frequency')) {
                return (value * 100).toFixed(1) + '%';
            }
            return value.toFixed(2);
        }

        function renderHypothesis(h, sortMetric, modelA, modelB) {
            const metrics = ['test_accuracy', 'test_frequency', 'interestingness_score', 'abstraction_score'];
            const labels = { test_accuracy: 'Acc', test_frequency: 'Freq', interestingness_score: 'Int', abstraction_score: 'Abs' };

            const scoresHtml = metrics.map(m => {
                const cls = m === sortMetric ? 'score highlight' : 'score';
                return `<span class="${cls}">${labels[m]}: ${formatScore(h[m], m)}</span>`;
            }).join('');

            // Substitute <MODEL> and <OTHER MODEL> based on predicted_direction
            let text = h.hypothesis_text;
            const predictedModel = h.predicted_direction === 'A' ? modelA : modelB;
            const otherModel = h.predicted_direction === 'A' ? modelB : modelA;
            text = text.replace(/<MODEL>/g, predictedModel || '<MODEL>');
            text = text.replace(/<OTHER MODEL>/g, otherModel || '<OTHER MODEL>');

            return `
                <div class="hypothesis">
                    <div class="hypothesis-text">${escapeHtml(text)}</div>
                    <div class="scores">${scoresHtml}</div>
                </div>
            `;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function render() {
            if (!data) return;

            const dataset = document.getElementById('dataset').value;
            const metric = document.getElementById('metric').value;
            const order = document.getElementById('order').value;

            const saeKey = dataset + '_sae';
            const llmKey = dataset + '_llm';

            function getHypotheses(key) {
                if (!data[key]) return [];
                return data[key].hypotheses.filter(h => h.accepted);
            }

            function sortHypotheses(hypotheses) {
                return [...hypotheses].sort((a, b) => {
                    let va = a[metric], vb = b[metric];
                    // Handle nulls - put them at the end
                    if (va === null || va === undefined) va = order === 'desc' ? -Infinity : Infinity;
                    if (vb === null || vb === undefined) vb = order === 'desc' ? -Infinity : Infinity;
                    return order === 'desc' ? vb - va : va - vb;
                });
            }

            const saeHypotheses = sortHypotheses(getHypotheses(saeKey));
            const llmHypotheses = sortHypotheses(getHypotheses(llmKey));

            document.getElementById('sae-count').textContent = `(${saeHypotheses.length} accepted)`;
            document.getElementById('llm-count').textContent = `(${llmHypotheses.length} accepted)`;

            const saeList = document.getElementById('sae-list');
            const llmList = document.getElementById('llm-list');

            const saeExp = data[saeKey];
            const llmExp = data[llmKey];

            if (saeHypotheses.length === 0) {
                saeList.innerHTML = '<div class="no-data">No accepted hypotheses for this dataset</div>';
            } else {
                const saeModelA = saeExp?.model_a || '';
                const saeModelB = saeExp?.model_b || '';
                saeList.innerHTML = saeHypotheses.map(h => renderHypothesis(h, metric, saeModelA, saeModelB)).join('');
            }

            if (llmHypotheses.length === 0) {
                llmList.innerHTML = '<div class="no-data">No accepted hypotheses for this dataset</div>';
            } else {
                const llmModelA = llmExp?.model_a || '';
                const llmModelB = llmExp?.model_b || '';
                llmList.innerHTML = llmHypotheses.map(h => renderHypothesis(h, metric, llmModelA, llmModelB)).join('');
            }
        }

        document.getElementById('dataset').addEventListener('change', render);
        document.getElementById('metric').addEventListener('change', render);
        document.getElementById('order').addEventListener('change', render);

        loadData();
    </script>
</body>
</html>
"""


class HypothesisHandler(http.server.SimpleHTTPRequestHandler):
    data = None

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.data).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress request logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Hypothesis Explorer Web Server")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to hypothesis evaluation JSON file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to serve on (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run evaluate_hypotheses.py first to generate the data file.")
        return 1

    with open(data_path, encoding="utf-8") as f:
        HypothesisHandler.data = json.load(f)

    print(f"Loaded data from: {data_path}")
    print(f"Available experiments: {list(HypothesisHandler.data.keys())}")

    # Start server
    with socketserver.TCPServer(("", args.port), HypothesisHandler) as httpd:
        print(f"\nServing at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            return 0


if __name__ == "__main__":
    exit(main() or 0)
