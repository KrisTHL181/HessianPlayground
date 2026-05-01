// ============================================================
// Constants
// ============================================================
const DATASET_DEFAULTS = {
    mnist: { inputSize: 784, outputSize: 10 },
    cifar10: { inputSize: 3072, outputSize: 10 },
    cifar100: { inputSize: 3072, outputSize: 100 },
    xor: { inputSize: 2, outputSize: 2 },
    polynomial: { inputSize: 1, outputSize: 1 },
};

// ============================================================
// WebSocket Manager
// ============================================================
class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnect = 20;
        this.pending = new Map();   // msg_id -> {resolve, reject, timer}
        this.listeners = new Map(); // type -> Set<callback>
        this.msgCounter = 0;
        this.onStatusChange = null;
        this._connect();
    }

    _connect() {
        this._setStatus('connecting');
        this.ws = new WebSocket(this.url);
        this.ws.onopen = () => {
            this._setStatus('connected');
            this.reconnectAttempts = 0;
        };
        this.ws.onmessage = (e) => this._onMessage(e.data);
        this.ws.onclose = (e) => {
            this._setStatus('disconnected');
            this.ws = null;
            // Reject all pending
            for (const [id, p] of this.pending) {
                clearTimeout(p.timer);
                p.reject(new Error('WebSocket disconnected'));
            }
            this.pending.clear();
            this._scheduleReconnect();
        };
        this.ws.onerror = () => { };
    }

    _setStatus(s) {
        this.status = s;
        document.getElementById('status-text').textContent = t('status.' + s);
        if (this.onStatusChange) this.onStatusChange(s);
    }

    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnect) return;
        const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 30000);
        this.reconnectAttempts++;
        setTimeout(() => this._connect(), delay);
    }

    _onMessage(data) {
        try {
            const msg = JSON.parse(data);
            const msgId = msg.msg_id;
            if (msgId && this.pending.has(msgId)) {
                const p = this.pending.get(msgId);
                clearTimeout(p.timer);
                this.pending.delete(msgId);
                if (msg.type === 'error') {
                    p.reject(new Error(msg.payload.message));
                } else {
                    p.resolve(msg.payload);
                }
                return;
            }
            // Push message — dispatch to type listeners
            const ls = this.listeners.get(msg.type);
            if (ls) ls.forEach(cb => { try { cb(msg.payload); } catch (e) { console.error(e); } });
            // Also dispatch to '*' catchall
            const all = this.listeners.get('*');
            if (all) all.forEach(cb => { try { cb(msg); } catch (e) { console.error(e); } });
        } catch (e) {
            console.error('WebSocket parse error:', e);
        }
    }

    send(type, payload = {}, timeoutMs = 120000) {
        return new Promise((resolve, reject) => {
            const msgId = 'm' + (++this.msgCounter);
            const timer = setTimeout(() => {
                this.pending.delete(msgId);
                reject(new Error(`Request '${type}' timed out`));
            }, timeoutMs);
            this.pending.set(msgId, { resolve, reject, timer });
            this.ws.send(JSON.stringify({ type, msg_id: msgId, payload }));
        });
    }

    on(type, callback) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type).add(callback);
    }

    off(type, callback) {
        const ls = this.listeners.get(type);
        if (ls) ls.delete(callback);
    }
}

// ============================================================
// Log Panel
// ============================================================
class LogPanel {
    constructor(el) {
        this.el = el;
        this.maxEntries = 500;
        this.autoScroll = true;
        this.el.addEventListener('scroll', () => {
            this.autoScroll = (this.el.scrollTop + this.el.clientHeight + 20 >= this.el.scrollHeight);
        });
    }

    log(level, message) {
        const div = document.createElement('div');
        div.className = 'log-entry ' + level;
        const ts = new Date().toLocaleTimeString();
        div.textContent = `[${ts}] ${message}`;
        this.el.appendChild(div);
        // Trim old entries
        while (this.el.children.length > this.maxEntries) {
            this.el.firstChild.remove();
        }
        if (this.autoScroll) this.el.scrollTop = this.el.scrollHeight;
    }

    info(msg) { this.log('info', msg); }
    warn(msg) { this.log('warning', msg); }
    error(msg) { this.log('error', msg); }
    debug(msg) { this.log('debug', msg); }
}

// ============================================================
// Code Editor (CodeMirror wrapper)
// ============================================================
class CodeEditor {
    constructor(parentEl, options = {}) {
        this.editor = CodeMirror(parentEl, {
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            indentUnit: 4,
            tabSize: 4,
            viewportMargin: Infinity,
            value: options.defaultValue || '',
            readOnly: options.readOnly || false,
        });
    }

    getCode() { return this.editor.getValue(); }
    setCode(code) { this.editor.setValue(code); }
    setReadOnly(ro) { this.editor.setOption('readOnly', ro); }
    onChange(cb) { this.editor.on('change', cb); }
}

// ============================================================
// Visualization Panel
// ============================================================
class VisualizationPanel {
    constructor() {
        this.tabs = ['loss', 'hessian', 'landscape', 'eigenvalues', 'equation'];
        this.currentTab = 'loss';
        this._plotIds = {
            loss: 'plot-loss',
            hessian: 'plot-hessian',
            landscape: 'plot-landscape',
            eigenvalues: 'plot-eigenvalues',
            equation: 'plot-equation',
        };
    }

    switchTab(tabId) {
        this.currentTab = tabId;
        document.querySelectorAll('.vis-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabId);
        });
        for (const [tab, id] of Object.entries(this._plotIds)) {
            document.getElementById(id).style.display = (tab === tabId) ? 'block' : 'none';
        }
        // Show/hide diagonal toggle button
        const btn = document.getElementById('btn-diag-toggle');
        if (btn) {
            btn.style.display = (tabId === 'hessian' && this._hessianData && !this._hessianData.isDiag) ? 'block' : 'none';
        }
        // Trigger resize on the newly visible plot
        setTimeout(() => {
            const el = document.getElementById(this._plotIds[tabId]);
            if (el && el._plotly) Plotly.Plots.resize(el);
        }, 50);
    }

    _getDiv(tab) { return document.getElementById(this._plotIds[tab]); }

    showEmpty(tab, message) {
        const div = this._getDiv(tab);
        Plotly.purge(div);
        div.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-dim)">${message || t('plot.no_data')}</div>`;
        if (tab === 'hessian') {
            this._hessianData = null;
            this._hessianDiagToggled = false;
            const btn = document.getElementById('btn-diag-toggle');
            if (btn) btn.style.display = 'none';
        }
    }

    showLossPlot(lossHistory, accuracyHistory = null) {
        const div = this._getDiv('loss');
        const traces = [{
            y: lossHistory,
            mode: 'lines',
            name: t('plot.loss'),
            line: { color: '#89b4fa', width: 2 },
        }];
        if (accuracyHistory && accuracyHistory.length > 0) {
            traces.push({
                y: accuracyHistory,
                mode: 'lines',
                name: t('plot.accuracy'),
                line: { color: '#a6e3a1', width: 2 },
                yaxis: 'y2',
            });
        }
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#cdd6f4', size: 11 },
            margin: { l: 40, r: 40, t: 10, b: 30 },
            xaxis: { title: t('plot.step'), color: '#8888aa', gridcolor: '#404070' },
            yaxis: { title: t('plot.loss'), color: '#8888aa', gridcolor: '#404070' },
            yaxis2: { title: t('plot.accuracy'), color: '#a6e3a1', overlaying: 'y', side: 'right' },
            showlegend: true,
            legend: { font: { size: 10 } },
        };
        Plotly.react(div, traces, layout, { responsive: true });
    }

    updateLossTrace(batchIdx, loss, accuracy = null) {
        const div = this._getDiv('loss');
        if (!div._plotly) return;
        Plotly.extendTraces(div, { y: [[loss]] }, [0]);
        if (accuracy !== null && div._plotly.data.length > 1) {
            Plotly.extendTraces(div, { y: [[accuracy]] }, [1]);
        }
    }

    showHessianHeatmap(matrix, labels, isDiag) {
        const div = this._getDiv('hessian');
        // Handle labels: backend sends list of strings
        const labelNames = Array.isArray(labels) ? labels : (labels.names || []);
        const tickIndices = Array.isArray(labels) ? labels.map((_, i) => i) : (labels.indices || []);
        // Subsample tick labels if too many
        const step = Math.max(1, Math.floor(labelNames.length / 30));
        const showTicks = tickIndices.filter((_, i) => i % step === 0);
        const showText = labelNames.filter((_, i) => i % step === 0);

        // Compute symmetric colorscale range
        const flat = Array.isArray(matrix) ? matrix.flat() : (matrix.data || []);
        const absMax = Math.max(...flat.map(v => Math.abs(v || 0)), 0.001);

        // Store original data for diagonal toggle
        this._hessianData = { matrix, labelNames, tickIndices, showTicks, showText, absMax, isDiag };
        this._hessianDiagToggled = false;

        this._renderHessian();
    }

    _renderHessian() {
        const div = this._getDiv('hessian');
        const d = this._hessianData;
        if (!d) return;

        let matrix = d.matrix;
        if (this._hessianDiagToggled && !d.isDiag) {
            matrix = d.matrix.map((row, i) =>
                row.map((v, j) => i === j ? v : 0)
            );
        }

        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#cdd6f4', size: 9 },
            margin: { l: 120, r: 20, t: 30, b: 60 },
            xaxis: { tickvals: d.showTicks, ticktext: d.showText, tickangle: 45, tickfont: { size: 8 } },
            yaxis: { tickvals: d.showTicks, ticktext: d.showText, autorange: 'reversed', tickfont: { size: 8 } },
            title: this._hessianDiagToggled ? t('plot.diagonal_hessian') : (d.isDiag ? t('plot.diagonal_hessian') : t('plot.hessian_matrix')),
            titlefont: { size: 12, color: '#cdd6f4' },
        };
        Plotly.react(div, [{
            z: matrix,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmin: -d.absMax,
            zmax: d.absMax,
        }], layout, { responsive: true });

        const btn = document.getElementById('btn-diag-toggle');
        if (btn) {
            btn.style.display = d.isDiag ? 'none' : 'block';
            btn.textContent = this._hessianDiagToggled ? t('btn.diagonal_only') : t('btn.diagonal_only');
        }
    }

    showLandscape(gridX, gridY, lossGrid, trajectory, mode) {
        const div = this._getDiv('landscape');
        const traces = [{
            z: lossGrid,
            x: gridX,
            y: gridY,
            type: 'contour',
            colorscale: 'Viridis',
            contours: { coloring: 'heatmap' },
            colorbar: { title: t('plot.loss') },
        }];
        if (trajectory && trajectory.x && trajectory.y) {
            traces.push({
                x: trajectory.x,
                y: trajectory.y,
                mode: 'lines+markers',
                type: 'scatter',
                line: { color: '#f38ba8', width: 3 },
                marker: { size: 4, color: '#f38ba8' },
                name: 'Trajectory',
            });
        }
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#cdd6f4', size: 11 },
            margin: { l: 40, r: 40, t: 30, b: 40 },
            title: mode === 'pca' ? t('plot.loss_landscape_pca') : t('plot.loss_landscape_random'),
            titlefont: { size: 12, color: '#cdd6f4' },
            xaxis: { title: t('plot.direction1'), color: '#8888aa' },
            yaxis: { title: t('plot.direction2'), color: '#8888aa' },
        };
        Plotly.react(div, traces, layout, { responsive: true });
    }

    showEigenvalues(evals, histBins, histCounts, stats) {
        const div = this._getDiv('eigenvalues');
        // Ensure container is visible so Plotly gets correct dimensions
        const prevDisplay = div.style.display;
        if (prevDisplay === 'none') div.style.display = 'block';
        const traces = [{
            x: histBins,
            y: histCounts,
            type: 'bar',
            marker: { color: '#89b4fa' },
        }];
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#cdd6f4', size: 11 },
            margin: { l: 40, r: 40, t: 30, b: 40 },
            title: `${t('tab.eigenvalues')} (min=${stats.min?.toFixed(3)}, max=${stats.max?.toFixed(3)}, cond=${stats.condition})`,
            titlefont: { size: 12, color: '#cdd6f4' },
            xaxis: { title: t('plot.eigenvalue'), color: '#8888aa' },
            yaxis: { title: t('plot.count'), color: '#8888aa' },
        };
        Plotly.react(div, traces, layout, { responsive: true });
        if (prevDisplay === 'none') div.style.display = 'none';
    }

    showEquationResult(data) {
        const div = this._getDiv('equation');
        const traces = [{
            x: [t('plot.before'), t('plot.after')],
            y: [data.loss_before, data.loss_after],
            type: 'bar',
            marker: { color: ['#f38ba8', '#a6e3a1'] },
            text: [data.loss_before?.toFixed(4), data.loss_after?.toFixed(4)],
            textposition: 'auto',
        }];
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#cdd6f4', size: 11 },
            margin: { l: 40, r: 40, t: 40, b: 40 },
            title: `${t('plot.newton_step')} (loss improved by ${data.loss_improvement?.toFixed(4)})`,
            titlefont: { size: 12, color: '#cdd6f4' },
        };
        Plotly.react(div, traces, layout, { responsive: true });
    }
}

// ============================================================
// Settings Panel
// ============================================================
class SettingsPanel {
    constructor(ws, log) {
        this.ws = ws;
        this.log = log;
        this._defaults = null;
        this._setup();
    }

    _setup() {
        const modal = document.getElementById('settings-modal');

        document.getElementById('btn-settings').onclick = () => this._open();
        document.getElementById('btn-settings-close').onclick = () => this._close();
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this._close();
        });

        // Tab switching
        document.querySelectorAll('.settings-tab').forEach(tab => {
            tab.onclick = () => {
                const stab = tab.dataset.stab;
                document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('settings-page-language').style.display = stab === 'language' ? 'block' : 'none';
                document.getElementById('settings-page-params').style.display = stab === 'params' ? 'block' : 'none';
            };
        });

        // Language selector
        document.getElementById('settings-lang').onchange = (e) => {
            setLanguage(e.target.value);
        };

        // Save
        document.getElementById('btn-settings-save').onclick = () => this._save();
        // Reset
        document.getElementById('btn-settings-reset').onclick = () => this._reset();
    }

    async _open() {
        document.getElementById('settings-lang').value = getLanguage();
        document.getElementById('settings-modal').classList.add('show');
        try {
            const config = await this.ws.send('get_config', {}, 5000);
            this._defaults = config;
            this._populate(config);
            this.log.debug(t('settings.loaded'));
        } catch (e) {
            this.log.warn(t('settings.error_load') + ': ' + e.message);
        }
    }

    _close() {
        document.getElementById('settings-modal').classList.remove('show');
    }

    _populate(config) {
        for (const [key, value] of Object.entries(config)) {
            const el = document.getElementById('cfg-' + key);
            if (el) el.value = value;
        }
    }

    _readConfig() {
        const updates = {};
        const inputs = document.querySelectorAll('[id^="cfg-"]');
        for (const inp of inputs) {
            const key = inp.id.replace('cfg-', '');
            const raw = inp.value.trim();
            if (raw === '') continue;
            const num = Number(raw);
            updates[key] = Number.isNaN(num) ? raw : num;
        }
        return updates;
    }

    async _save() {
        const updates = this._readConfig();
        try {
            const result = await this.ws.send('update_config', { updates }, 5000);
            this._populate(result);
            this.log.info(t('settings.saved'));
            this._close();
        } catch (e) {
            this.log.error(t('settings.error_save') + ': ' + e.message);
        }
    }

    async _reset() {
        if (this._defaults) {
            this._populate(this._defaults);
        }
        try {
            const result = await this.ws.send('update_config', { updates: this._readConfig() }, 5000);
            this._populate(result);
            this.log.info(t('settings.reset'));
        } catch (e) {
            this.log.error(t('settings.error_save') + ': ' + e.message);
        }
    }
}

// ============================================================
// Presets Loader
// ============================================================
class Presets {
    static async load() {
        const files = ['mlp.py', 'dataset.py', 'optimizer.py'];
        const results = {};
        for (const f of files) {
            const resp = await fetch(`/static/presets/${f}`);
            if (!resp.ok) throw new Error(`Failed to load preset: ${f}`);
            results[f.replace('.py', '')] = await resp.text();
        }
        return results;
    }

    static async loadModel(name) {
        const resp = await fetch(`/static/presets/${name}.py`);
        if (!resp.ok) throw new Error(`Failed to load preset: ${name}`);
        return await resp.text();
    }
}

// ============================================================
// App Orchestrator
// ============================================================
class App {
    static async init() {
        const presets = await Presets.load();
        return new App(presets);
    }

    constructor(presets) {
        this.presets = presets;

        // Populate textarea defaults from presets
        const dsTextarea = document.getElementById('custom-dataset-code');
        if (dsTextarea) dsTextarea.textContent = presets.dataset;
        const optTextarea = document.getElementById('custom-opt-code');
        if (optTextarea) optTextarea.textContent = presets.optimizer;

        // Init components
        this.ws = new WebSocketManager(`ws://${location.host}/ws`);
        this.log = new LogPanel(document.getElementById('log-output'));
        this.modelEditor = new CodeEditor(document.getElementById('model-editor'), {
            defaultValue: presets.mlp,
        });
        this.vis = new VisualizationPanel();
        this.settings = new SettingsPanel(this.ws, this.log);

        this.state = {
            hasModel: false,
            hasDataset: false,
            hasOptimizer: false,
            training: false,
            paramCount: 0,
            lossHistory: [],
            accHistory: [],
            datasetReady: false,
        };
        this._pendingAdaptation = null;  // 'reset_full' | 'reset_first_last' | 'expand_matrices'

        this._setupListeners();
        this._initEmptyPlots();
    }

    _initEmptyPlots() {
        for (const tab of this.vis.tabs) {
            if (tab === 'loss') continue; // handled by training
            this.vis.showEmpty(tab);
        }
    }

    _setupListeners() {
        // Connection status
        this.ws.onStatusChange = (status) => {
            document.getElementById('status-dot').className = 'status-dot ' + status;
        };

        // Push messages
        this.ws.on('training_progress', (p) => this._onTrainingProgress(p));
        this.ws.on('training_complete', (p) => this._onTrainingComplete(p));
        this.ws.on('dataset_ready', (p) => {
            this.state.datasetReady = true;
            // Auto-populate model dimensions from non-custom datasets
            if (p.dataset_name !== 'custom') {
                if (p.input_size) document.getElementById('input-size').value = p.input_size;
                if (p.num_classes && p.task === 'classification') document.getElementById('output-size').value = p.num_classes;
                else if (p.task === 'regression') document.getElementById('output-size').value = p.num_classes || 1;
            }
            // Invalidate model if already created — unless the user already handled adaptation
            if (this.state.hasModel && !this._pendingAdaptation) {
                this.state.hasModel = false;
                this.state.paramCount = 0;
                this._setButtons(false);
                this.log.warn('Model invalidated — recreate model to match new dataset dimensions');
            }
            // Clear one-shot adaptation flag after the first dataset_ready that consumed it
            if (this._pendingAdaptation === 'reset_full') {
                this._pendingAdaptation = null;
            }
            this.log.info(`Dataset ready: ${p.dataset_name || 'custom'} (input=${p.input_size}, output=${p.num_classes}, ${p.train_samples} train)`);
        });
        this.ws.on('model_created', (p) => {
            this.state.hasModel = true;
            this.state.paramCount = p.num_parameters;
            this._pendingAdaptation = null;
        });
        this.ws.on('status', (p) => this.log.log(p.level, p.message));
        this.ws.on('*', (msg) => {
            if (msg.type === 'error') {
                this.log.error(`[${msg.payload.code}] ${msg.payload.message}`);
            }
        });

        // Load preset button
        document.getElementById('btn-load-preset').onclick = () => this._loadPreset();

        // Buttons
        document.getElementById('btn-create').onclick = () => this._createModel();
        document.getElementById('btn-train').onclick = () => this._startTraining();
        document.getElementById('btn-stop').onclick = () => this._stopTraining();
        document.getElementById('btn-hessian').onclick = () => this._computeHessian();
        document.getElementById('btn-pca-landscape').onclick = () => this._computeLandscape('pca');
        document.getElementById('btn-random-landscape').onclick = () => this._computeLandscape('random');
        document.getElementById('btn-newton').onclick = () => this._solveNewton();
        document.getElementById('btn-reset').onclick = () => this._reset();

        // Tab switching
        document.querySelectorAll('.vis-tab').forEach(t => {
            t.onclick = () => this.vis.switchTab(t.dataset.tab);
        });

        // Diagonal toggle for Hessian heatmap
        document.getElementById('btn-diag-toggle').onclick = () => {
            this.vis._hessianDiagToggled = !this.vis._hessianDiagToggled;
            this.vis._renderHessian();
        };

        // Dataset toggle
        document.getElementById('dataset-select').onchange = (e) => {
            const isCustom = e.target.value === 'custom';
            document.getElementById('custom-dataset-group').style.display =
                isCustom ? 'block' : 'none';
            document.getElementById('size-group').style.display =
                isCustom ? 'flex' : 'none';
            this.state.datasetReady = false;

            if (!isCustom) {
                const defs = DATASET_DEFAULTS[e.target.value];
                if (defs) {
                    document.getElementById('input-size').value = defs.inputSize;
                    document.getElementById('output-size').value = defs.outputSize;
                }
            }

            // Show adaptation popup if a model already exists
            if (this.state.hasModel && !isCustom) {
                this._pendingAdaptation = null;
                document.getElementById('adapt-popup').classList.add('show');
            }
        };

        // Optimizer toggle
        document.getElementById('optimizer-select').onchange = (e) => {
            document.getElementById('custom-opt-group').style.display =
                e.target.value === 'custom' ? 'block' : 'none';
        };

        // Hessian modal
        document.getElementById('hessian-cancel').onclick = () =>
            document.getElementById('hessian-modal').classList.remove('show');
        document.getElementById('hessian-diag').onclick = () => {
            document.getElementById('hessian-modal').classList.remove('show');
            this._computeHessian(true);
        };
        document.getElementById('hessian-full').onclick = () => {
            document.getElementById('hessian-modal').classList.remove('show');
            this._computeHessian(false);
        };

        // Dataset adaptation popup
        document.getElementById('adapt-reset-full').onclick = () => this._handleAdaptChoice('reset_full');
        document.getElementById('adapt-reset-ends').onclick = () => this._handleAdaptChoice('reset_first_last');
        document.getElementById('adapt-expand').onclick = () => this._handleAdaptChoice('expand_matrices');

        // Log resizer
        const resizer = document.getElementById('log-resizer');
        const logPanel = document.getElementById('log-panel');
        let resizeStart = 0, startHeight = 0;
        resizer.onmousedown = (e) => {
            resizeStart = e.clientY;
            startHeight = logPanel.offsetHeight;
            document.onmousemove = (e) => {
                const newH = Math.max(80, Math.min(400, startHeight - (e.clientY - resizeStart)));
                logPanel.style.height = newH + 'px';
            };
            document.onmouseup = () => { document.onmousemove = null; document.onmouseup = null; };
        };
    }

    _setButtons(enabled) {
        const btns = ['btn-hessian', 'btn-pca-landscape', 'btn-random-landscape', 'btn-newton'];
        btns.forEach(id => {
            document.getElementById(id).disabled = !enabled;
        });
        document.getElementById('btn-train').disabled = !this.state.hasModel || this.state.training;
        document.getElementById('btn-stop').disabled = !this.state.training;
    }

    async _createModel() {
        const code = this.modelEditor.getCode();
        try {
            // Load dataset first to get correct input/output dimensions and ensure data_loader exists
            if (!this.state.datasetReady) {
                await this._setDataset();
            }
            const inputSize = parseInt(document.getElementById('input-size').value) || 784;
            const outputSize = parseInt(document.getElementById('output-size').value) || 10;
            const hiddenRaw = document.getElementById('hidden-sizes').value || '128,64';
            const hiddenSizes = hiddenRaw.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            this.log.info(`Creating model: input=${inputSize}, hidden=${hiddenSizes}, output=${outputSize}...`);
            const result = await this.ws.send('create_model', {
                code,
                input_size: inputSize,
                hidden_sizes: hiddenSizes,
                output_size: outputSize,
            });
            this.state.hasModel = true;
            this.state.paramCount = result.num_parameters;
            this.log.info(`Model created: ${result.model_name} (${result.num_parameters} params)`);
            this._setButtons(true);
        } catch (e) {
            this.log.error(`Failed to create model: ${e.message}`);
        }
    }

    async _startTraining() {
        // Ensure dataset is set
        if (!this.state.datasetReady) {
            await this._setDataset();
        }
        // Ensure optimizer is set
        await this._setOptimizer();

        try {
            const epochs = parseInt(document.getElementById('epochs-input').value) || 5;
            this.state.lossHistory = [];
            this.state.accHistory = [];
            this.log.info(`Starting training for ${epochs} epochs...`);
            const result = await this.ws.send('start_training', {
                epochs,
                record_params_every: 50,
                record_loss_every: 1,
            });
            if (result.status === 'started') {
                this.state.training = true;
                this._setButtons(true);
            }
        } catch (e) {
            this.log.error(`Training failed: ${e.message}`);
            this.state.training = false;
            this._setButtons(false);
        }
    }

    async _stopTraining() {
        try {
            await this.ws.send('stop_training');
            this.log.info('Stopping training...');
        } catch (e) {
            this.log.error(`Stop failed: ${e.message}`);
        }
    }

    async _setDataset() {
        const dsName = document.getElementById('dataset-select').value;
        const batchSize = parseInt(document.getElementById('batch-size').value) || 64;
        try {
            this.log.info(`Loading dataset: ${dsName}...`);
            if (dsName === 'custom') {
                const code = document.getElementById('custom-dataset-code').value;
                if (!code.trim()) throw new Error('Enter custom dataset code');
                const task = document.getElementById('custom-task-type').value;
                await this.ws.send('set_custom_dataset', { code, batch_size: batchSize, task });
            } else {
                await this.ws.send('set_dataset', {
                    dataset: dsName,
                    params: { batch_size: batchSize, normalize: true },
                });
            }
            this.state.datasetReady = true;
            this.log.info(`Dataset ${dsName} loaded`);
        } catch (e) {
            this.log.error(`Dataset error: ${e.message}`);
            throw e;
        }
    }

    async _setOptimizer() {
        const optName = document.getElementById('optimizer-select').value;
        const lr = parseFloat(document.getElementById('lr-input').value) || 0.001;
        try {
            if (optName === 'custom') {
                const code = document.getElementById('custom-opt-code').value;
                await this.ws.send('set_custom_optimizer', { code });
            } else {
                const defaultParams = { lr };
                if (optName === 'SGD') defaultParams.momentum = 0.9;
                if (optName === 'Adam' || optName === 'AdamW') {
                    defaultParams.betas = [0.9, 0.999];
                    defaultParams.eps = 1e-8;
                }
                await this.ws.send('set_optimizer', { optimizer: optName, params: defaultParams });
            }
            this.state.hasOptimizer = true;
        } catch (e) {
            this.log.error(`Optimizer error: ${e.message}`);
            throw e;
        }
    }

    _onTrainingProgress(p) {
        this.state.lossHistory.push(p.loss);
        if (p.train_accuracy !== undefined) this.state.accHistory.push(p.train_accuracy);

        if (this.state.lossHistory.length === 1) {
            this.vis.showLossPlot(this.state.lossHistory, this.state.accHistory.length > 0 ? this.state.accHistory : null);
            this.vis.switchTab('loss');
        } else {
            this.vis.updateLossTrace(p.batch || this.state.lossHistory.length, p.loss,
                this.state.accHistory.length > 0 ? p.train_accuracy : null);
        }

        this.log.debug(`Epoch ${p.epoch}/${p.total_epochs} Batch ${p.batch} Loss ${p.loss.toFixed(4)}`);
    }

    _onTrainingComplete(p) {
        this.state.training = false;
        this.state.lossHistory = p.loss_history || [];
        this.state.accHistory = p.accuracy_history || [];
        this._setButtons(true);
        this.vis.showLossPlot(this.state.lossHistory, this.state.accHistory);
        this.log.info(`Training complete: loss=${p.final_loss?.toFixed(4)}, elapsed=${p.elapsed_seconds?.toFixed(1)}s`);
    }

    async _computeHessian(forceDiag = null) {
        if (!this.state.hasModel) {
            this.log.error('Create a model first');
            return;
        }
        const useDiag = forceDiag !== null ? forceDiag : (this.state.paramCount > 10000);
        if (this.state.paramCount > 10000 && forceDiag === null) {
            document.getElementById('hessian-modal-text').textContent =
                `Model has ${this.state.paramCount} parameters. Full Hessian needs ~${(this.state.paramCount * this.state.paramCount * 8 / 1024 / 1024).toFixed(0)} MB. Use diagonal instead?`;
            // Keep data-i18n as fallback; dynamic text overrides it
            document.getElementById('hessian-modal').classList.add('show');
            return;
        }
        try {
            this.log.info('Computing Hessian...');
            const result = await this.ws.send('compute_hessian', {
                use_diagonal_approx: useDiag,
                sample_batches: 1,
            });
            this.vis.switchTab('hessian');
            this.vis.showHessianHeatmap(result.hessian_matrix, result.dim_labels || [], result.is_diagonal);

            // Auto-compute eigenvalues
            const evResult = await this.ws.send('compute_hessian_eigenvalues', { method: result.is_diagonal ? 'diagonal' : 'exact' });
            this.vis.showEigenvalues(evResult.eigenvalues, evResult.histogram_bins, evResult.histogram_counts, {
                min: evResult.min_eigenvalue,
                max: evResult.max_eigenvalue,
                condition: evResult.condition_number,
                num_positive: evResult.num_positive,
                num_negative: evResult.num_negative,
            });
            this.log.info(`Hessian computed: shape=${result.hessian_shape}, diagonal=${result.is_diagonal}`);
        } catch (e) {
            this.log.error(`Hessian error: ${e.message}`);
        }
    }

    async _computeLandscape(mode) {
        if (!this.state.hasModel) {
            this.log.error('Create a model first');
            return;
        }
        try {
            const res = parseInt(document.getElementById('batch-size').value) > 16 ? 20 : 30;
            this.log.info(`Computing ${mode} landscape...`);
            const type = mode === 'pca' ? 'compute_pca_landscape' : 'compute_random_landscape';
            const result = await this.ws.send(type, { grid_resolution: res, range_factor: 2.0 });
            const traj = result.mode === 'pca' ? {
                x: result.trajectory_x,
                y: result.trajectory_y,
            } : null;
            this.vis.switchTab('landscape');
            this.vis.showLandscape(result.grid_x, result.grid_y, result.loss_grid, traj, result.mode);
            this.log.info(`Landscape computed (${result.mode})`);
        } catch (e) {
            this.log.error(`Landscape error: ${e.message}`);
        }
    }

    async _solveNewton() {
        if (!this.state.hasModel) {
            this.log.error('Create a model first');
            return;
        }
        try {
            this.log.info('Solving Newton step...');
            const result = await this.ws.send('solve_newton_step', {
                regularization: 1e-4,
                apply_step: true,
                step_scale: 1.0,
            });
            this.vis.switchTab('equation');
            this.vis.showEquationResult(result);
            this.log.info(`Newton step: loss ${result.loss_before?.toFixed(4)} → ${result.loss_after?.toFixed(4)} (Δ=${result.loss_improvement?.toFixed(4)})`);
        } catch (e) {
            this.log.error(`Newton step error: ${e.message}`);
        }
    }

    async _reset() {
        try {
            await this.ws.send('reset_model');
        } catch (e) { }
        this.state = {
            hasModel: false,
            hasDataset: false,
            hasOptimizer: false,
            training: false,
            paramCount: 0,
            lossHistory: [],
            accHistory: [],
            datasetReady: false,
        };
        this._pendingAdaptation = null;
        this._setButtons(false);
        document.getElementById('btn-train').disabled = true;
        document.getElementById('btn-stop').disabled = true;
        this._initEmptyPlots();
        this.log.info('Reset complete');
    }

    async _loadPreset() {
        const name = document.getElementById('preset-select').value;
        try {
            const code = await Presets.loadModel(name);
            this.modelEditor.setCode(code);
            this.log.info(`Loaded preset: ${name}`);
        } catch (e) {
            this.log.error(`Failed to load preset '${name}': ${e.message}`);
        }
    }

    async _handleAdaptChoice(mode) {
        document.getElementById('adapt-popup').classList.remove('show');

        if (mode === 'reset_full') {
            this._pendingAdaptation = 'reset_full';
            try {
                await this.ws.send('reset_model');
            } catch (e) { }
            this.state.hasModel = false;
            this.state.paramCount = 0;
            this._setButtons(false);
            document.getElementById('btn-train').disabled = true;
            document.getElementById('btn-stop').disabled = true;
            this.log.warn('Model reset — recreate model to match new dataset dimensions');
            return;
        }

        // For reset_first_last and expand_matrices, send adapt_model immediately
        const inputSize = parseInt(document.getElementById('input-size').value) || 784;
        const outputSize = parseInt(document.getElementById('output-size').value) || 10;

        try {
            this.log.info(`Adapting model (${mode})...`);
            const result = await this.ws.send('adapt_model', {
                mode,
                input_size: inputSize,
                output_size: outputSize,
            });
            this.state.paramCount = result.num_parameters;
            this._pendingAdaptation = mode;
            this.log.info(`Model adapted: ${result.old_input_size}→${result.new_input_size} in, ${result.old_output_size}→${result.new_output_size} out (${result.num_parameters} params)`);
        } catch (e) {
            this.log.error(`Model adaptation failed: ${e.message}`);
            this._pendingAdaptation = null;
        }
    }
}

// ============================================================
// Boot
// ============================================================
(async () => {
    setLanguage(getLanguage());
    const app = await App.init();
    app._setButtons(false);
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-stop').disabled = true;
})();
