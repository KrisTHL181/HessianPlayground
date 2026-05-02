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

// Optimizer-specific hyperparameter definitions
const OPTIMIZER_PARAMS = {
    SGD: [
        { name: 'momentum', label: 'Momentum', type: 'number', defaultValue: 0.9, step: 0.1, min: 0 },
        { name: 'weight_decay', label: 'Weight Decay', type: 'number', defaultValue: 0, step: 0.0001, min: 0 },
        { name: 'nesterov', label: 'Nesterov', type: 'checkbox', defaultValue: false },
    ],
    Adam: [
        { name: 'betas_0', label: 'β₁', type: 'number', defaultValue: 0.9, step: 0.01, min: 0 },
        { name: 'betas_1', label: 'β₂', type: 'number', defaultValue: 0.999, step: 0.001, min: 0 },
        { name: 'eps', label: 'ε', type: 'number', defaultValue: 1e-8, step: 1e-8, min: 0 },
        { name: 'weight_decay', label: 'Weight Decay', type: 'number', defaultValue: 0, step: 0.0001, min: 0 },
    ],
    AdamW: [
        { name: 'betas_0', label: 'β₁', type: 'number', defaultValue: 0.9, step: 0.01, min: 0 },
        { name: 'betas_1', label: 'β₂', type: 'number', defaultValue: 0.999, step: 0.001, min: 0 },
        { name: 'eps', label: 'ε', type: 'number', defaultValue: 1e-8, step: 1e-8, min: 0 },
        { name: 'weight_decay', label: 'Weight Decay', type: 'number', defaultValue: 0.01, step: 0.0001, min: 0 },
    ],
    RMSprop: [
        { name: 'alpha', label: 'Smoothing', type: 'number', defaultValue: 0.99, step: 0.01, min: 0 },
        { name: 'momentum', label: 'Momentum', type: 'number', defaultValue: 0, step: 0.1, min: 0 },
        { name: 'weight_decay', label: 'Weight Decay', type: 'number', defaultValue: 0, step: 0.0001, min: 0 },
    ],
    Adagrad: [
        { name: 'lr_decay', label: 'LR Decay', type: 'number', defaultValue: 0, step: 0.001, min: 0 },
        { name: 'weight_decay', label: 'Weight Decay', type: 'number', defaultValue: 0, step: 0.0001, min: 0 },
        { name: 'eps', label: 'ε', type: 'number', defaultValue: 1e-8, step: 1e-8, min: 0 },
    ],
    Adadelta: [
        { name: 'rho', label: 'Decay', type: 'number', defaultValue: 0.9, step: 0.01, min: 0 },
        { name: 'weight_decay', label: 'Weight Decay', type: 'number', defaultValue: 0, step: 0.0001, min: 0 },
        { name: 'eps', label: 'ε', type: 'number', defaultValue: 1e-6, step: 1e-8, min: 0 },
    ],
};

const METHOD_NAMES = { full: 'Full', diagonal: 'Diagonal', kfac: 'K-FAC', block_diag: 'Block-Diag' };

// ============================================================
// Theme management
// ============================================================
const THEME_STORAGE_KEY = 'hessian-theme';

function getCurrentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

function getThemeColors() {
    const style = getComputedStyle(document.documentElement);
    const v = (name) => style.getPropertyValue(name).trim();
    return {
        plotFont: v('--plot-font') || '#141413',
        plotGrid: v('--plot-grid') || '#e8e6dc',
        plotLoss: v('--plot-loss') || '#c96442',
        plotAccuracy: v('--plot-accuracy') || '#3b8258',
        plotEigen: v('--plot-eigen') || '#c96442',
        plotTrajectory: v('--plot-trajectory') || '#8c2981',
        plotEqBefore: v('--plot-equation-before') || '#b53333',
        plotEqAfter: v('--plot-equation-after') || '#3b8258',
        textDim: v('--text-dim') || '#87867f',
        text: v('--text') || '#141413',
    };
}

function setTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    _updateThemeButton(theme);
    _updateThemeSettings(theme);
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

function _updateThemeButton(theme) {
    const btn = document.getElementById('btn-theme');
    if (btn) {
        btn.textContent = theme === 'dark' ? '☀' : '☾';
        btn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
}

function _updateThemeSettings(theme) {
    const sel = document.getElementById('settings-theme');
    if (sel) {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        sel.value = stored || 'auto';
    }
}

function toggleTheme() {
    const next = getCurrentTheme() === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem(THEME_STORAGE_KEY, next);
}

function initTheme() {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') {
        setTheme(saved);
    } else {
        // Auto-detect system preference, default to dark
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setTheme(prefersDark ? 'dark' : 'light');
    }

    // Listen for system preference changes (only applies when no manual override)
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem(THEME_STORAGE_KEY)) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
}

// ============================================================
// Toast notification
// ============================================================
function showToast(message, duration = 3000) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

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
        this.onReconnect = null;
        this._computeDevice = null; // "local_cpu" | "local_cuda" | "remote_cpu" | "remote_cuda"
        this._connect();
    }

    _connect() {
        this._setStatus('connecting');
        this.ws = new WebSocket(this.url);
        this.ws.onopen = () => {
            const wasReconnect = this.reconnectAttempts > 0;
            this._setStatus('connected');
            this.reconnectAttempts = 0;
            if (wasReconnect && this.onReconnect) {
                this.onReconnect();
            }
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
        this.updateStatusText();
        if (this.onStatusChange) this.onStatusChange(s);
    }

    setComputeDevice(device) {
        this._computeDevice = device;
        this.updateStatusText();
    }

    updateStatusText() {
        if (this.status === 'connected' && this._computeDevice) {
            const key = 'status.connected_' + this._computeDevice;
            document.getElementById('status-text').textContent = t(key);
        } else {
            document.getElementById('status-text').textContent = t('status.' + this.status);
        }
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
        const isDark = getCurrentTheme() === 'dark';
        this._cmTheme = isDark ? 'monokai' : 'eclipse';
        this.editor = CodeMirror(parentEl, {
            mode: 'python',
            theme: this._cmTheme,
            lineNumbers: true,
            indentUnit: 4,
            tabSize: 4,
            viewportMargin: Infinity,
            value: options.defaultValue || '',
            readOnly: options.readOnly || false,
        });
        this._onThemeChange = (e) => {
            this._cmTheme = e.detail.theme === 'dark' ? 'monokai' : 'eclipse';
            this.editor.setOption('theme', this._cmTheme);
        };
        window.addEventListener('themechange', this._onThemeChange);
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
        // Hessian sub-selector visibility
        const subSel = document.getElementById('hessian-sub-selector');
        const diagBtn = document.getElementById('btn-diag-toggle');
        const badge = document.getElementById('hessian-method-badge');
        const isHessian = tabId === 'hessian';
        const d = this._hessianData;
        const isKfac = d && d.type === 'kfac';
        const isBlkDiag = d && d.type === 'block_diag';
        const isSingle = d && d.type === 'single';
        if (subSel) subSel.style.display = (isHessian && (isKfac || isBlkDiag)) ? 'flex' : 'none';
        if (diagBtn) diagBtn.style.display = (isHessian && isSingle && !d.isDiag) ? 'block' : 'none';
        if (badge) badge.style.display = (isHessian && d) ? 'block' : 'none';
        const kfacSel = document.getElementById('kfac-layer-select');
        const blkSel = document.getElementById('block-select');
        if (kfacSel) kfacSel.style.display = (isHessian && isKfac) ? 'inline-block' : 'none';
        if (blkSel) blkSel.style.display = (isHessian && isBlkDiag) ? 'inline-block' : 'none';
        // Trigger resize
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
            const badge = document.getElementById('hessian-method-badge');
            if (badge) badge.style.display = 'none';
            const subSel = document.getElementById('hessian-sub-selector');
            if (subSel) subSel.style.display = 'none';
            const kfacSel = document.getElementById('kfac-layer-select');
            if (kfacSel) kfacSel.style.display = 'none';
            const blkSel = document.getElementById('block-select');
            if (blkSel) blkSel.style.display = 'none';
        }
    }

    showLossPlot(lossHistory, accuracyHistory = null) {
        const div = this._getDiv('loss');
        const C = getThemeColors();
        const traces = [{
            y: lossHistory,
            mode: 'lines',
            name: t('plot.loss'),
            line: { color: C.plotLoss, width: 2 },
        }];
        if (accuracyHistory && accuracyHistory.length > 0) {
            traces.push({
                y: accuracyHistory,
                mode: 'lines',
                name: t('plot.accuracy'),
                line: { color: C.plotAccuracy, width: 2 },
                yaxis: 'y2',
            });
        }
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 11 },
            margin: { l: 40, r: 40, t: 10, b: 30 },
            xaxis: { title: t('plot.step'), color: C.textDim, gridcolor: C.plotGrid },
            yaxis: { title: t('plot.loss'), color: C.textDim, gridcolor: C.plotGrid },
            yaxis2: { title: t('plot.accuracy'), color: C.plotAccuracy, overlaying: 'y', side: 'right' },
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

    showHessianHeatmap(result) {
        const div = this._getDiv('hessian');
        const method = result.method || 'full';
        const displayType = result.display_type || 'full';

        // Method badge
        const badge = document.getElementById('hessian-method-badge');
        if (badge) {
            badge.textContent = METHOD_NAMES[method] || method;
            badge.style.display = 'block';
        }

        // Hide sub-selectors initially
        ['kfac-layer-select', 'block-select', 'btn-diag-toggle'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        const subSel = document.getElementById('hessian-sub-selector');
        if (subSel) subSel.style.display = 'none';

        if (displayType === 'kfac') {
            this._showKfacView(result);
        } else if (displayType === 'block_diagonal') {
            this._showBlockDiagView(result);
        } else {
            this._showSingleHeatmap(result);
        }
    }

    _showSingleHeatmap(result) {
        const matrix = result.hessian_matrix;
        if (!matrix) { Plotly.purge(this._getDiv('hessian')); return; }

        const labels = result.dim_labels || [];
        const isDiag = result.is_diagonal;
        const N = labels.length;

        const step = Math.max(1, Math.floor(N / 40));
        const showTicks = labels.map((_, i) => i).filter((_, i) => i % step === 0);
        const showText = labels.filter((_, i) => i % step === 0);

        const flat = Array.isArray(matrix) ? matrix.flat(2) : (matrix.data || []);
        const absMax = Math.max(...flat.map(v => Math.abs(v || 0)), 0.001);

        this._hessianData = {
            type: 'single', matrix, labels, isDiag,
            tickvals: showTicks, ticktext: showText, absMax,
        };
        this._hessianDiagToggled = false;

        this._renderSingleHeatmap();
    }

    _renderHessian() {
        const d = this._hessianData;
        if (!d) return;
        if (d.type === 'kfac') this._renderKfacLayer();
        else if (d.type === 'block_diag') this._renderBlockDetail();
        else this._renderSingleHeatmap();
    }

    _renderSingleHeatmap() {
        const div = this._getDiv('hessian');
        const d = this._hessianData;
        if (!d || d.type !== 'single') return;

        let matrix = d.matrix;
        if (this._hessianDiagToggled && !d.isDiag) {
            matrix = d.matrix.map((row, i) =>
                row.map((v, j) => i === j ? v : 0)
            );
        }

        const C = getThemeColors();
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 9 },
            margin: { l: 120, r: 20, t: 30, b: 60 },
            xaxis: { tickvals: d.tickvals, ticktext: d.ticktext, tickangle: 45, tickfont: { size: 8 } },
            yaxis: { tickvals: d.tickvals, ticktext: d.ticktext, autorange: 'reversed', tickfont: { size: 8 } },
            title: this._hessianDiagToggled ? t('plot.diagonal_hessian') : (d.isDiag ? t('plot.diagonal_hessian') : t('plot.hessian_matrix')),
            titlefont: { size: 12, color: C.plotFont },
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
            btn.textContent = t('btn.diagonal_only');
        }
    }

    _showKfacView(result) {
        const factors = result.kfac_factors || [];
        if (factors.length === 0) return;

        this._hessianData = { type: 'kfac', factors, currentIdx: 0 };

        const sel = document.getElementById('kfac-layer-select');
        const subSel = document.getElementById('hessian-sub-selector');
        if (!sel) return;
        sel.innerHTML = factors.map((f, i) =>
            `<option value="${i}">${f.layer_name} (${f.param_count} params)</option>`
        ).join('');
        sel.style.display = 'inline-block';
        if (subSel) subSel.style.display = 'flex';
        sel.onchange = () => {
            this._hessianData.currentIdx = parseInt(sel.value);
            this._renderKfacLayer();
        };
        document.getElementById('btn-diag-toggle').style.display = 'none';
        document.getElementById('block-select').style.display = 'none';
        this._renderKfacLayer();
    }

    _renderKfacLayer() {
        const div = this._getDiv('hessian');
        const d = this._hessianData;
        if (!d || d.type !== 'kfac') return;
        const f = d.factors[d.currentIdx];
        if (!f) return;

        const aFlat = f.A_matrix.flat(2);
        const gFlat = f.G_matrix.flat(2);
        const absMax = Math.max(
            ...aFlat.map(v => Math.abs(v || 0)),
            ...gFlat.map(v => Math.abs(v || 0)),
            0.001,
        );

        const C = getThemeColors();
        Plotly.react(div, [
            {
                z: f.A_matrix, type: 'heatmap', colorscale: 'RdBu',
                zmin: -absMax, zmax: absMax, xaxis: 'x', yaxis: 'y',
                name: 'Input Cov A',
            },
            {
                z: f.G_matrix, type: 'heatmap', colorscale: 'RdBu',
                zmin: -absMax, zmax: absMax, xaxis: 'x2', yaxis: 'y2',
                name: 'Gradient Cov G',
            },
        ], {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 10 },
            margin: { l: 60, r: 60, t: 50, b: 50 },
            grid: { rows: 1, columns: 2, pattern: 'independent' },
            title: `K-FAC: ${f.layer_name} (${f.in_features}→${f.out_features})`,
            titlefont: { size: 13, color: C.plotFont },
            xaxis: { title: 'Input dim' },
            yaxis: { title: 'Input dim', autorange: 'reversed' },
            xaxis2: { title: 'Output dim' },
            yaxis2: { title: 'Output dim', autorange: 'reversed' },
        }, { responsive: true });
    }

    _showBlockDiagView(result) {
        const blocks = result.block_matrices || [];
        if (blocks.length === 0) { this._showSingleHeatmap(result); return; }

        this._hessianData = { type: 'block_diag', blocks, currentIdx: 0 };

        const sel = document.getElementById('block-select');
        const subSel = document.getElementById('hessian-sub-selector');
        if (!sel) return;
        sel.innerHTML = blocks.map((b, i) =>
            `<option value="${i}">${b.block_name} (${b.block_param_count} params)</option>`
        ).join('');
        sel.style.display = 'inline-block';
        if (subSel) subSel.style.display = 'flex';
        sel.onchange = () => {
            this._hessianData.currentIdx = parseInt(sel.value);
            this._renderBlockDetail();
        };
        document.getElementById('btn-diag-toggle').style.display = 'none';
        document.getElementById('kfac-layer-select').style.display = 'none';
        this._renderBlockDetail();
    }

    _renderBlockDetail() {
        const div = this._getDiv('hessian');
        const d = this._hessianData;
        if (!d || d.type !== 'block_diag') return;
        const b = d.blocks[d.currentIdx];
        if (!b) return;

        const flat = b.block_matrix.flat(2);
        const absMax = Math.max(...flat.map(v => Math.abs(v || 0)), 0.001);

        const C = getThemeColors();
        Plotly.react(div, [{
            z: b.block_matrix, type: 'heatmap', colorscale: 'RdBu',
            zmin: -absMax, zmax: absMax,
        }], {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 10 },
            margin: { l: 80, r: 20, t: 50, b: 60 },
            title: `Block: ${b.block_name} (${b.block_param_count} params)`,
            titlefont: { size: 13, color: C.plotFont },
            xaxis: { automargin: true },
            yaxis: { autorange: 'reversed', automargin: true },
        }, { responsive: true });
    }

    showLandscape(gridX, gridY, lossGrid, trajectory, mode) {
        const div = this._getDiv('landscape');
        const C = getThemeColors();
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
                line: { color: C.plotTrajectory, width: 3 },
                marker: { size: 4, color: C.plotTrajectory },
                name: 'Trajectory',
            });
        }
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 11 },
            margin: { l: 40, r: 40, t: 30, b: 40 },
            title: mode === 'pca' ? t('plot.loss_landscape_pca') : t('plot.loss_landscape_random'),
            titlefont: { size: 12, color: C.plotFont },
            xaxis: { title: t('plot.direction1'), color: C.textDim },
            yaxis: { title: t('plot.direction2'), color: C.textDim },
        };
        Plotly.react(div, traces, layout, { responsive: true });
    }

    showEigenvalues(evals, histBins, histCounts, stats) {
        const div = this._getDiv('eigenvalues');
        // Ensure container is visible so Plotly gets correct dimensions
        const prevDisplay = div.style.display;
        if (prevDisplay === 'none') div.style.display = 'block';
        const C = getThemeColors();
        const traces = [{
            x: histBins,
            y: histCounts,
            type: 'bar',
            marker: { color: C.plotEigen },
        }];
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 11 },
            margin: { l: 40, r: 40, t: 30, b: 40 },
            title: `${t('tab.eigenvalues')} (min=${stats.min?.toFixed(3)}, max=${stats.max?.toFixed(3)}, cond=${stats.condition})`,
            titlefont: { size: 12, color: C.plotFont },
            xaxis: { title: t('plot.eigenvalue'), color: C.textDim },
            yaxis: { title: t('plot.count'), color: C.textDim },
        };
        Plotly.react(div, traces, layout, { responsive: true });
        if (prevDisplay === 'none') div.style.display = 'none';
    }

    showEquationResult(data) {
        const div = this._getDiv('equation');
        const C = getThemeColors();
        const traces = [{
            x: [t('plot.before'), t('plot.after')],
            y: [data.loss_before, data.loss_after],
            type: 'bar',
            marker: { color: [C.plotEqBefore, C.plotEqAfter] },
            text: [data.loss_before?.toFixed(4), data.loss_after?.toFixed(4)],
            textposition: 'auto',
        }];
        const layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: C.plotFont, size: 11 },
            margin: { l: 40, r: 40, t: 40, b: 40 },
            title: `${t('plot.newton_step')} (loss improved by ${data.loss_improvement?.toFixed(4)})`,
            titlefont: { size: 12, color: C.plotFont },
        };
        Plotly.react(div, traces, layout, { responsive: true });
    }
}

// ============================================================
// Settings Panel
// ============================================================
class SettingsPanel {
    constructor(ws, log, onRemoteChange) {
        this.ws = ws;
        this.log = log;
        this._defaults = null;
        this._remoteConnected = false;
        this._cudaAvailable = false;
        this._onRemoteChange = onRemoteChange || null;
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
                document.querySelectorAll('.settings-page').forEach(p => p.style.display = 'none');
                const page = document.getElementById('settings-page-' + stab);
                if (page) page.style.display = 'block';
            };
        });

        // Language selector
        document.getElementById('settings-lang').onchange = (e) => {
            setLanguage(e.target.value);
            this.ws.updateStatusText();
        };

        // Theme selector
        document.getElementById('settings-theme').onchange = (e) => {
            const value = e.target.value;
            if (value === 'auto') {
                localStorage.removeItem(THEME_STORAGE_KEY);
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                setTheme(prefersDark ? 'dark' : 'light');
            } else {
                localStorage.setItem(THEME_STORAGE_KEY, value);
                setTheme(value);
            }
        };

        // Remote connect/disconnect
        document.getElementById('btn-remote-connect').onclick = () => this._remoteConnect();
        document.getElementById('btn-remote-disconnect').onclick = () => this._remoteDisconnect();

        // SSH command parse
        document.getElementById('btn-parse-ssh').onclick = () => this._parseSSHCommand();
        document.getElementById('ssh-command-input').onkeydown = (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this._parseSSHCommand(); }
        };

        // Save
        document.getElementById('btn-settings-save').onclick = () => this._save();
        // Reset
        document.getElementById('btn-settings-reset').onclick = () => this._reset();
    }

    async _open() {
        document.getElementById('settings-lang').value = getLanguage();
        document.getElementById('settings-theme').value = localStorage.getItem(THEME_STORAGE_KEY) || 'auto';
        // Reset to first tab
        document.querySelectorAll('.settings-tab').forEach((t, i) => t.classList.toggle('active', i === 0));
        document.querySelectorAll('.settings-page').forEach((p, i) => p.style.display = i === 0 ? 'block' : 'none');
        document.getElementById('settings-modal').classList.add('show');
        try {
            const config = await this.ws.send('get_config', {}, 5000);
            this._defaults = config;
            this._populate(config);
            this._cudaAvailable = config.CUDA_AVAILABLE;
            this._updateCudaStatus();
            this._updateRemoteStatus();
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
            if (el) {
                if (el.tagName === 'SELECT' || el.type === 'password') {
                    el.value = String(value);
                } else if (el.type === 'number') {
                    el.value = value;
                } else {
                    el.value = value;
                }
            }
        }
    }

    _readConfig() {
        const updates = {};
        const inputs = document.querySelectorAll('[id^="cfg-"]');
        for (const inp of inputs) {
            const key = inp.id.replace('cfg-', '');
            const raw = inp.value.trim();
            if (raw === '') continue;
            if (inp.tagName === 'SELECT') {
                const val = inp.value;
                updates[key] = (val === 'true') ? true : (val === 'false') ? false : val;
            } else if (inp.type === 'number') {
                const num = Number(raw);
                updates[key] = Number.isNaN(num) ? raw : num;
            } else {
                updates[key] = raw;
            }
        }
        return updates;
    }

    _updateCudaStatus() {
        const sel = document.getElementById('cfg-DEVICE');
        const statusEl = document.getElementById('cuda-status-text');
        if (this._cudaAvailable) {
            statusEl.textContent = t('settings.cuda_available');
            statusEl.style.color = 'var(--green)';
            if (sel) {
                sel.querySelector('option[value="cuda"]').disabled = false;
            }
        } else {
            statusEl.textContent = t('settings.cuda_unavailable');
            statusEl.style.color = 'var(--text-dim)';
            if (sel) {
                sel.querySelector('option[value="cuda"]').disabled = true;
                sel.value = 'cpu';
            }
        }
    }

    _updateRemoteStatus() {
        const statusEl = document.getElementById('remote-status-text');
        const connectBtn = document.getElementById('btn-remote-connect');
        const disconnectBtn = document.getElementById('btn-remote-disconnect');
        if (this._remoteConnected) {
            statusEl.textContent = t('settings.remote_connected');
            statusEl.style.color = 'var(--green)';
            if (connectBtn) connectBtn.disabled = true;
            if (disconnectBtn) disconnectBtn.disabled = false;
        } else {
            statusEl.textContent = t('settings.remote_disconnected');
            statusEl.style.color = 'var(--text-dim)';
            if (connectBtn) connectBtn.disabled = false;
            if (disconnectBtn) disconnectBtn.disabled = true;
        }
    }

    async _remoteConnect() {
        // Save remote config first
        const configBefore = this._readConfig();
        try {
            await this.ws.send('update_config', { updates: configBefore }, 5000);
        } catch (e) {
            this.log.error(t('settings.error_save') + ': ' + e.message);
            return;
        }

        const statusEl = document.getElementById('remote-status-text');
        statusEl.textContent = t('settings.remote_connecting');
        statusEl.style.color = 'var(--orange)';

        try {
            const result = await this.ws.send('connect_remote', {}, 15000);
            this._remoteConnected = true;
            this._updateRemoteStatus();
            if (this._onRemoteChange) this._onRemoteChange();
            this.log.info(result.message || t('settings.remote_connected'));
        } catch (e) {
            this._remoteConnected = false;
            this._updateRemoteStatus();
            statusEl.textContent = t('settings.remote_error') + ': ' + e.message;
            statusEl.style.color = 'var(--red)';
            this.log.error(t('settings.remote_error') + ': ' + e.message);
        }
    }

    async _remoteDisconnect() {
        try {
            await this.ws.send('disconnect_remote', {}, 5000);
            this._remoteConnected = false;
            this._updateRemoteStatus();
            if (this._onRemoteChange) this._onRemoteChange();
            this.log.info(t('settings.remote_disconnected'));
        } catch (e) {
            this.log.error(t('settings.remote_error') + ': ' + e.message);
        }
    }

    _parseSSHCommand() {
        const input = document.getElementById('ssh-command-input').value.trim();
        if (!input) return;

        // Strip leading "ssh " if present
        let s = input.replace(/^ssh\s+/, '');

        let user = '';
        let host = '';
        let port = '';

        // Extract -p port (before stripping other flags)
        const portMatch = s.match(/(?:^|\s)-p\s+(\d+)\b/);
        if (portMatch) {
            port = portMatch[1];
            s = s.replace(portMatch[0], ' ');
        }

        // Extract -l user
        const userFlagMatch = s.match(/(?:^|\s)-l\s+(\S+)/);
        if (userFlagMatch) {
            user = userFlagMatch[1];
            s = s.replace(userFlagMatch[0], ' ');
        }

        // Remove flags that take an argument: -i, -L, -R, -D, -o, -J, -S, -O
        s = s.replace(/\s*-[iLRDoJS]\s+\S+/g, ' ');
        // Remove boolean flags (stackable): -X, -A, -v, -q, -t, -T, -N, -f, -C, -4, -6, -g, -n, -E, -G, -K, -k, -s, -x, -Y, -y
        s = s.replace(/\s*-[XAvqTtNfCgnEKGksxYy46]+\b/g, ' ');

        // Extract user@host or plain host from remaining text
        const tokens = s.trim().split(/\s+/);
        for (const tok of tokens) {
            const atIdx = tok.indexOf('@');
            if (atIdx > 0) {
                if (!user) user = tok.slice(0, atIdx);
                host = tok.slice(atIdx + 1);
            } else if (!tok.startsWith('-')) {
                host = tok;
            }
        }

        // Clean host — strip trailing colon+port (ssh user@host:22 format)
        if (host) {
            const colonPort = host.match(/^(.+):(\d+)$/);
            if (colonPort) {
                host = colonPort[1];
                if (!port) port = colonPort[2];
            }
        }

        if (host) document.getElementById('cfg-REMOTE_HOST').value = host;
        if (user) document.getElementById('cfg-REMOTE_USER').value = user;
        if (port) document.getElementById('cfg-REMOTE_PORT').value = port;

        this.log.info(`Parsed SSH: ${user ? user + '@' : ''}${host}${port ? ' :' + port : ''}`);
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
        this.ws = new WebSocketManager(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
        this.log = new LogPanel(document.getElementById('log-output'));
        this.modelEditor = new CodeEditor(document.getElementById('model-editor'), {
            defaultValue: presets.mlp,
        });
        this.vis = new VisualizationPanel();
        this.settings = new SettingsPanel(this.ws, this.log, () => this._refreshDeviceLabel());

        this.state = {
            hasModel: false,
            hasDataset: false,
            hasOptimizer: false,
            training: false,
            paramCount: 0,
            lossHistory: [],
            accHistory: [],
            datasetReady: false,
            _trainSamples: 0,
        };
        this._pendingAdaptation = null;  // 'reset_full' | 'reset_first_last' | 'expand_matrices'

        this._setupListeners();
        this._initEmptyPlots();
        this._updateOptimizerParams();
    }

    _initEmptyPlots() {
        for (const tab of this.vis.tabs) {
            if (tab === 'loss') continue; // handled by training
            this.vis.showEmpty(tab);
        }
    }

    _setupListeners() {
        // Theme toggle
        document.getElementById('btn-theme').onclick = () => toggleTheme();

        // Keyboard shortcut for theme toggle
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                toggleTheme();
            }
        });

        // Connection status
        this.ws.onStatusChange = (status) => {
            document.getElementById('status-dot').className = 'status-dot ' + status;
            if (status === 'connected') this._refreshDeviceLabel();
        };
        this.ws.onReconnect = () => {
            this.state.hasModel = false;
            this.state.paramCount = 0;
            this.state.datasetReady = false;
            this.state.hasOptimizer = false;
            this.state.training = false;
            this.state._trainSamples = 0;
            this._setButtons(false);
            document.getElementById('btn-train').disabled = true;
            document.getElementById('btn-stop').disabled = true;
            this.log.warn('Connection restored — backend session was reset. Please recreate model and dataset.');
            this._updateSnapshotEstimate();
            for (const tab of this.vis.tabs) {
                if (tab === 'loss') continue;
                this.vis.showEmpty(tab);
            }
        };

        // Push messages
        this.ws.on('training_progress', (p) => this._onTrainingProgress(p));
        this.ws.on('training_complete', (p) => this._onTrainingComplete(p));
        this.ws.on('dataset_ready', (p) => {
            this.state.datasetReady = true;
            this.state._trainSamples = p.train_samples || 0;
            this._updateSnapshotEstimate();
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

        // Snapshot estimate
        document.getElementById('snapshot-interval').oninput = () => this._updateSnapshotEstimate();
        document.getElementById('epochs-input').onchange = () => this._updateSnapshotEstimate();
        document.getElementById('batch-size').onchange = () => this._updateSnapshotEstimate();
        this._updateSnapshotEstimate();

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
            this._updateOptimizerParams();
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

    // ── Optimizer params UI ──
    _updateOptimizerParams() {
        const optName = document.getElementById('optimizer-select').value;
        const group = document.getElementById('optimizer-params-group');
        const params = OPTIMIZER_PARAMS[optName];

        if (!params || params.length === 0 || optName === 'custom') {
            group.innerHTML = '';
            group.style.display = 'none';
            return;
        }
        group.style.display = 'block';

        let html = '';
        for (const p of params) {
            if (p.type === 'checkbox') {
                html += `<div class="opt-param" style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                    <label style="font-size:11px;min-width:80px;margin-bottom:0">${p.label}</label>
                    <input type="checkbox" id="opt-${p.name}" ${p.defaultValue ? 'checked' : ''}>
                </div>`;
            } else {
                const step = p.step ? `step="${p.step}"` : 'step="any"';
                const min = p.min !== undefined ? `min="${p.min}"` : '';
                html += `<div class="opt-param" style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                    <label style="font-size:11px;min-width:80px;margin-bottom:0">${p.label}</label>
                    <input type="number" id="opt-${p.name}" value="${p.defaultValue}" ${step} ${min} style="flex:1;padding:3px 6px;font-size:12px">
                </div>`;
            }
        }
        group.innerHTML = html;
    }

    _readOptimizerParams() {
        const lr = parseFloat(document.getElementById('lr-input').value);
        const params = { lr: isNaN(lr) ? 0.001 : lr };

        const optName = document.getElementById('optimizer-select').value;
        const defs = OPTIMIZER_PARAMS[optName];
        if (!defs || optName === 'custom') return params;

        for (const p of defs) {
            if (p.name.startsWith('betas_')) continue;
            if (p.type === 'checkbox') {
                const el = document.getElementById(`opt-${p.name}`);
                if (el) params[p.name] = el.checked;
            } else {
                const el = document.getElementById(`opt-${p.name}`);
                if (el) {
                    const val = parseFloat(el.value);
                    if (!isNaN(val)) params[p.name] = val;
                }
            }
        }

        if (defs.some(p => p.name === 'betas_0')) {
            const b0 = parseFloat(document.getElementById('opt-betas_0')?.value || '0.9');
            const b1 = parseFloat(document.getElementById('opt-betas_1')?.value || '0.999');
            params.betas = [b0, b1];
        }

        return params;
    }

    // ── Snapshot estimate ──
    _updateSnapshotEstimate() {
        const el = document.getElementById('snapshot-estimate');
        const input = document.getElementById('snapshot-interval');
        if (!el || !input) return;
        const interval = parseInt(input.value) || 0;
        if (interval > 0) {
            const batches = this._estimateTotalBatches();
            const count = Math.floor(batches / interval);
            el.textContent = t('snapshot.estimated').replace('{count}', count);
        } else {
            const batches = this._estimateTotalBatches();
            const auto = Math.max(1, Math.floor(batches / 30));
            const count = Math.floor(batches / auto);
            el.textContent = t('snapshot.auto').replace('{auto}', auto).replace('{count}', count);
        }
    }

    _estimateTotalBatches() {
        const epochs = parseInt(document.getElementById('epochs-input').value) || 5;
        const trainSamples = this.state._trainSamples || 800;
        const batchSize = parseInt(document.getElementById('batch-size').value) || 64;
        const batchesPerEpoch = Math.max(1, Math.ceil(trainSamples / batchSize));
        return epochs * batchesPerEpoch;
    }

    _setButtons(enabled) {
        const btns = ['btn-hessian', 'btn-pca-landscape', 'btn-random-landscape', 'btn-newton'];
        btns.forEach(id => {
            document.getElementById(id).disabled = !enabled;
        });
        document.getElementById('btn-train').disabled = !this.state.hasModel || this.state.training;
        document.getElementById('btn-stop').disabled = !this.state.training;
    }

    async _refreshDeviceLabel() {
        try {
            const config = await this.ws.send('get_config', {}, 5000);
            const remoteStatus = await this.ws.send('get_remote_status', {}, 5000);
            let device;
            if (remoteStatus.connected && remoteStatus.remote_device) {
                device = 'remote_' + remoteStatus.remote_device;
            } else {
                device = 'local_' + (config.DEVICE || 'cpu');
            }
            this.ws.setComputeDevice(device);
        } catch (e) {
            this.ws.setComputeDevice('local_cpu');
        }
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
            let snapshotInterval = parseInt(document.getElementById('snapshot-interval').value) || 0;
            if (snapshotInterval <= 0) {
                snapshotInterval = Math.max(1, Math.floor(this._estimateTotalBatches() / 30));
            }
            this.state.lossHistory = [];
            this.state.accHistory = [];
            this.log.info(`Starting training for ${epochs} epochs, snapshot every ${snapshotInterval} batches...`);
            const result = await this.ws.send('start_training', {
                epochs,
                record_params_every: snapshotInterval,
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
        let lr = parseFloat(document.getElementById('lr-input').value) || 0.001;
        let gradientAscent = false;
        if (lr < 0) {
            gradientAscent = true;
            lr = Math.abs(lr);
            showToast('Gradient ascent mode');
        }
        try {
            if (optName === 'custom') {
                const code = document.getElementById('custom-opt-code').value;
                await this.ws.send('set_custom_optimizer', { code, gradient_ascent: gradientAscent });
            } else {
                const params = this._readOptimizerParams();
                params.lr = lr;
                await this.ws.send('set_optimizer', { optimizer: optName, params, gradient_ascent: gradientAscent });
            }
            this.state.hasOptimizer = true;
            this.state.gradientAscent = gradientAscent;
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
        const method = forceDiag !== null
            ? (forceDiag ? 'diagonal' : 'full')
            : document.getElementById('hessian-method').value;
        const dtype = document.getElementById('hessian-dtype').value;

        // Warn for full Hessian on large models (when not explicitly forced)
        if ((method === 'full' || method === 'auto') && this.state.paramCount > 2000 && forceDiag === null) {
            const sizeMB = (this.state.paramCount * this.state.paramCount * 8 / 1024 / 1024).toFixed(0);
            document.getElementById('hessian-modal-text').textContent =
                t('modal.hessian_warning_dynamic').replace('{count}', this.state.paramCount).replace('{size}', sizeMB);
            document.getElementById('hessian-modal').classList.add('show');
            return;
        }
        try {
            this.log.info(`Computing Hessian (method=${method}, dtype=${dtype})...`);
            const result = await this.ws.send('compute_hessian', {
                method, use_diagonal_approx: method === 'diagonal',
                sample_batches: 1, dtype,
            }, 600000);
            this.vis.switchTab('hessian');
            this.vis.showHessianHeatmap(result);

            // Auto-compute eigenvalues
            try {
                const evResult = await this.ws.send('compute_hessian_eigenvalues', { method: 'auto' });
                this.vis.showEigenvalues(evResult.eigenvalues, evResult.histogram_bins, evResult.histogram_counts, {
                    min: evResult.min_eigenvalue,
                    max: evResult.max_eigenvalue,
                    condition: evResult.condition_number,
                    num_positive: evResult.num_positive,
                    num_negative: evResult.num_negative,
                });
                this.log.info(`Hessian computed: method=${result.method}, display=${result.display_type}, mem≈${result.memory_mb?.toFixed(1)}MB`);
            } catch (evErr) {
                this.log.warn(`Eigenvalue computation failed: ${evErr.message}`);
            }
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
            _trainSamples: 0,
        };
        this._pendingAdaptation = null;
        this._setButtons(false);
        document.getElementById('btn-train').disabled = true;
        document.getElementById('btn-stop').disabled = true;
        this._initEmptyPlots();
        this._updateSnapshotEstimate();
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
    initTheme();
    setLanguage(getLanguage());
    const app = await App.init();
    app._setButtons(false);
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-stop').disabled = true;
})();
