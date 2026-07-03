/**
 * Reports tab component
 * Handles report generation and viewing
 * Styled to match Streamlit UI
 */

const ReportsComponent = {
    reports: [],
    evaluations: [],
    selectedReport: null,
    selectedReportId: null,  // report_id of the row selected in the table (for view/delete)
    reportScope: 'all',
    modelScope: 'all',
    selectedEvalNames: [],  // Track selected evaluation names for model filtering
    selectedModelIds: [],   // Track selected model ids (persisted across re-renders)
    selectedSections: ['latency_metrics', 'latency_distribution', 'accuracy_distribution', 'token_distribution', 'cost_metrics', 'task_analysis', 'model_task_performance', 'regional_performance'],
    reportCount: 0,
    reportLimit: 10,
    reportPollInterval: null,  // polls while any report is 'generating'

    /**
     * Initialize the reports component
     */
    async init() {
        await this.loadData();
        this.render();
        // Resume polling if a report was already generating when the tab loaded.
        if (this.reports.some(r => (r.status || 'completed') === 'generating')) {
            this._startReportPolling();
        }
    },

    /**
     * Load reports and evaluations
     */
    async loadData() {
        try {
            const [reportsRes, evalsRes] = await Promise.all([
                API.getReports(),
                API.getEvaluations()
            ]);
            this.reports = reportsRes.reports || [];
            this.reportCount = reportsRes.report_count || this.reports.length;
            this.reportLimit = reportsRes.report_limit || 10;
            this.evaluations = evalsRes.evaluations || [];
        } catch (error) {
            console.error('Failed to load data:', error);
        }
    },

    /**
     * Render the reports tab (Streamlit-like layout)
     */
    render() {
        const container = document.getElementById('reports-content');
        const completedEvals = this.evaluations.filter(e => e.status === 'completed');

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <!-- Generate New Report -->
                <h3 class="st-subheader" style="margin-top: 0;">Generate New Report</h3>

                ${this.reportCount >= this.reportLimit ? `
                    <div class="st-warning-box ds-mb-3">
                        <i class="bi bi-exclamation-triangle"></i>
                        Report limit reached (${this.reportCount}/${this.reportLimit}). Delete existing reports before generating new ones.
                    </div>
                ` : `
                    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">
                        Reports: ${this.reportCount}/${this.reportLimit}
                    </p>
                `}

                <div class="ds-flex ds-gap-3 ds-mb-4" style="align-items:flex-start; flex-wrap:wrap;">
                <div class="ds-card" style="flex:1 1 320px; min-width:0;">
                    <!-- Report Scope Radio -->
                    <label class="st-label">Report Scope</label>
                    <div class="st-radio-nav" style="flex-direction: row; gap: 1.5rem;">
                        <label>
                            <input type="radio" name="report-scope" value="all" ${this.reportScope === 'all' ? 'checked' : ''}>
                            <span>All Evaluations</span>
                        </label>
                        <label>
                            <input type="radio" name="report-scope" value="selected" ${this.reportScope === 'selected' ? 'checked' : ''}>
                            <span>Selected Evaluations</span>
                        </label>
                    </div>

                    <!-- Eval select (always rendered, toggled via display) -->
                    <div id="eval-select-panel" style="${this.reportScope === 'selected' ? '' : 'display:none;'}">
                        <div class="ds-mt-3">
                            <div class="ds-flex ds-items-center ds-gap-2" style="flex-wrap:wrap;">
                                <label class="st-label" style="margin:0;">Select evaluations for report:</label>
                                <button type="button" class="ds-btn ds-btn--sm" data-checklist-all="eval">Select all</button>
                                <button type="button" class="ds-btn ds-btn--sm" data-checklist-clear="eval">Clear</button>
                                <span class="st-caption" id="eval-select-count">${this.selectedEvalNames.length} selected</span>
                            </div>
                            <div class="st-checklist" id="report-eval-list">
                                ${this._evalChecklistHTML(completedEvals)}
                            </div>
                        </div>
                    </div>
                    <div id="eval-all-info" style="${this.reportScope === 'all' ? '' : 'display:none;'}">
                        <div class="ds-alert ds-alert--info ds-mt-2"><i class="bi bi-info-circle ds-alert__icon"></i><div class="ds-alert__content"><div class="ds-alert__message">Report will include all completed evaluations.</div></div></div>
                    </div>
                </div>

                <div class="ds-card" style="flex:1 1 320px; min-width:0;">
                    <!-- Model Scope Radio -->
                    <label class="st-label">Model Scope</label>
                    <div class="st-radio-nav" style="flex-direction: row; gap: 1.5rem;">
                        <label>
                            <input type="radio" name="model-scope" value="all" ${this.modelScope === 'all' ? 'checked' : ''}>
                            <span>All Models</span>
                        </label>
                        <label>
                            <input type="radio" name="model-scope" value="selected" ${this.modelScope === 'selected' ? 'checked' : ''}>
                            <span>Selected Models</span>
                        </label>
                    </div>

                    <!-- Model select (always rendered, toggled via display) -->
                    <div id="model-select-panel" style="${this.modelScope === 'selected' ? '' : 'display:none;'}">
                        <div class="ds-mt-3">
                            <div class="ds-flex ds-items-center ds-gap-2" style="flex-wrap:wrap;">
                                <label class="st-label" style="margin:0;">Select models for report:</label>
                                <button type="button" class="ds-btn ds-btn--sm" data-checklist-all="model">Select all</button>
                                <button type="button" class="ds-btn ds-btn--sm" data-checklist-clear="model">Clear</button>
                                <span class="st-caption" id="model-select-count">${this.selectedModelIds.length} selected</span>
                            </div>
                            <div class="st-checklist" id="report-model-list">
                                ${this._modelChecklistHTML()}
                            </div>
                            <div id="model-select-info" class="ds-mt-2"></div>
                        </div>
                    </div>
                    <div id="model-all-info" style="${this.modelScope === 'all' ? '' : 'display:none;'}">
                        <div class="ds-alert ds-alert--info ds-mt-2"><i class="bi bi-info-circle ds-alert__icon"></i><div class="ds-alert__content"><div class="ds-alert__message">Report will include all models from selected evaluations.</div></div></div>
                    </div>
                </div>
                </div>

                <!-- Report Sections -->
                <div class="ds-card ds-mb-4">
                    <label class="st-label">Report Sections</label>
                    <p style="color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 0.5rem;">
                        Select which analysis sections to include. Executive Summary, Evaluation Settings, and Model Recommendations are always included.
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;">
                        ${[
                            { key: 'latency_metrics', label: 'Latency Metrics' },
                            { key: 'cost_metrics', label: 'Cost Metrics' },
                            { key: 'latency_distribution', label: 'Latency Distribution' },
                            { key: 'task_analysis', label: 'Task Analysis' },
                            { key: 'accuracy_distribution', label: 'Accuracy Distribution' },
                            { key: 'model_task_performance', label: 'Model-Task Performance' },
                            { key: 'token_distribution', label: 'Token Distribution' },
                            { key: 'regional_performance', label: 'Regional Performance' },
                        ].map(s => `
                            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer; font-size: 0.9rem;">
                                <input type="checkbox" class="report-section-check" data-section="${s.key}" ${this.selectedSections.includes(s.key) ? 'checked' : ''}>
                                <span>${s.label}</span>
                            </label>
                        `).join('')}
                    </div>
                </div>

                <!-- Executive Summary Model -->
                <div class="ds-card ds-mb-4">
                    <label class="st-label">Executive Summary Model</label>
                    <p style="color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 0.5rem;">
                        Select the model that will generate the AI executive summary in the report.
                    </p>
                    <div class="st-columns-2" style="gap: 1rem;">
                        <div>
                            <select class="ds-select" id="summary-model-select">
                                <option value="bedrock/global.amazon.nova-2-lite-v1:0">Amazon Nova 2 Lite (default)</option>
                                <option value="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0">Claude Sonnet 4.5</option>
                                <option value="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0">Claude Sonnet 4</option>
                                <option value="bedrock/us.amazon.nova-pro-v1:0">Amazon Nova Pro</option>
                                <option value="bedrock/us.amazon.nova-premier-v1:0">Amazon Nova Premier</option>
                            </select>
                        </div>
                        <div>
                            <select class="ds-select" id="summary-region-select">
                                <option value="us-east-1">us-east-1</option>
                                <option value="us-west-2">us-west-2</option>
                                <option value="us-east-2">us-east-2</option>
                                <option value="eu-west-1">eu-west-1</option>
                                <option value="eu-central-1">eu-central-1</option>
                                <option value="ap-northeast-1">ap-northeast-1</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Generate Button -->
                <button class="ds-btn ds-btn--primary ds-btn--lg" id="generate-report-btn">
                    <i class="bi bi-file-earmark-bar-graph"></i> Generate Report
                </button>

                <hr class="ds-divider">

                <!-- Refresh Button -->
                <button class="ds-btn ds-btn--secondary" id="refresh-reports-btn">
                    <i class="bi bi-arrow-clockwise"></i> Refresh
                </button>

                <!-- Available Reports -->
                <h3 class="st-subheader">Available Reports (${this.reports.length})</h3>
                ${this.reports.length > 0 ? `
                    <p class="st-caption ds-mb-2">Click a row to view a report.</p>
                    <div class="ds-card ds-card--flush" style="max-height: 250px; overflow-y: auto;">
                        <table class="ds-table">
                            <thead>
                                <tr>
                                    <th>Created</th>
                                    <th>Status</th>
                                    <th>Evaluations</th>
                                    <th>Models</th>
                                    <th>Size</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.reports.map(r => {
                                    const st = r.status || 'completed';
                                    const id = r.report_id || r.status_file;
                                    const sel = this.selectedReportId === id ? ' is-selected' : '';
                                    const gen = st === 'generating' ? ' is-generating' : '';
                                    return `
                                    <tr class="report-row${sel}${gen}" data-report-id="${id}" data-report-path="${r.report_path || ''}" data-status="${st}">
                                        <td>${r.creation_time_formatted}</td>
                                        <td>${this._reportStatusChip(st)}</td>
                                        <td>${this._reportListCell(r.evaluations_used, 'evaluation')}</td>
                                        <td>${this._reportListCell(r.models_included, 'model')}</td>
                                        <td>${r.file_size || '—'}</td>
                                    </tr>`;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>

                    <!-- Delete Button (acts on the row selected above) -->
                    <button class="ds-btn ds-btn--danger ds-mt-3" id="delete-report-btn">
                        <i class="bi bi-trash"></i> Delete Selected Report
                    </button>
                ` : `
                    <div class="ds-empty">
                        <i class="bi bi-file-earmark-bar-graph ds-empty__icon"></i>
                        <div class="ds-empty__title">No Reports Available</div>
                        <div class="ds-empty__description">Generate a report to see it here.</div>
                    </div>
                `}

                <!-- Report Viewer -->
                ${this.selectedReport ? `
                    <hr class="ds-divider ds-divider--accent">

                    <!-- Report Details -->
                    <h4 class="st-section-header">Report Details</h4>
                    <div class="st-columns-3 ds-mb-4">
                        <div class="ds-metric ds-metric--compact">
                            <div class="ds-metric__label">Name</div>
                            <div class="ds-metric__value" style="font-size: 16px;">${this.getSelectedReportName()}</div>
                        </div>
                        <div class="ds-metric ds-metric--compact">
                            <div class="ds-metric__label">Created</div>
                            <div class="ds-metric__value" style="font-size: 16px;">${this.getSelectedReportDate()}</div>
                        </div>
                        <div class="ds-metric ds-metric--compact">
                            <div class="ds-metric__label">Size</div>
                            <div class="ds-metric__value" style="font-size: 16px;">${this.getSelectedReportSize()}</div>
                        </div>
                    </div>

                    <hr class="ds-divider">

                    <!-- Report Content -->
                    <h4 class="st-section-header">Report Content</h4>
                    <div class="st-iframe-container">
                        <iframe id="report-iframe" src="/api/reports/${this.selectedReport}"></iframe>
                    </div>

                    <!-- Download Report -->
                    <h4 class="st-section-header ds-mt-4">Download Report</h4>
                    <button class="ds-btn ds-btn--primary ds-btn--lg" id="download-report-btn">
                        <i class="bi bi-download"></i> Download HTML Report
                    </button>
                ` : ''}
            </div>
        `;

        this.bindEvents();
    },

    /**
     * Get models from source evaluations based on report scope
     * If "Selected Evaluations" and there are selected eval names, filter to those
     * Otherwise, use all completed evaluations
     */
    getModelsFromSourceEvaluations(completedEvals) {
        let sourceEvals;

        if (this.reportScope === 'selected' && this.selectedEvalNames.length > 0) {
            // Filter to only selected evaluations
            sourceEvals = completedEvals.filter(e => this.selectedEvalNames.includes(e.name));
        } else {
            // Use all completed evaluations
            sourceEvals = completedEvals;
        }

        // Extract unique models from source evaluations
        const models = new Map();
        sourceEvals.forEach(e => {
            (e.selected_models || []).forEach(m => {
                const id = m.id || m.model_id;
                if (id && !models.has(id)) {
                    models.set(id, {
                        id: id,
                        name: this.extractModelName(id)
                    });
                }
            });
        });

        // Sort by display name
        return Array.from(models.values()).sort((a, b) => a.name.localeCompare(b.name));
    },

    /** Render the evaluation checkbox list items. */
    _evalChecklistHTML(completedEvals) {
        if (!completedEvals.length) return '<span class="st-caption">No completed evaluations.</span>';
        return completedEvals.map(e => `
            <label class="st-checkitem${this.selectedEvalNames.includes(e.name) ? ' is-checked' : ''}">
                <input type="checkbox" class="report-eval-check" value="${e.name}" ${this.selectedEvalNames.includes(e.name) ? 'checked' : ''}>
                <span>${e.name}</span>
            </label>`).join('');
    },

    /** Render the model checkbox list items (union of models across the in-scope evals).
     *  Drops any previously-selected models that are no longer available. */
    _modelChecklistHTML() {
        const completed = this.evaluations.filter(ev => ev.status === 'completed');
        const models = this.getModelsFromSourceEvaluations(completed);
        const avail = new Set(models.map(m => m.id));
        this.selectedModelIds = (this.selectedModelIds || []).filter(id => avail.has(id));
        if (!models.length) return '<span class="st-caption">No models found for the selected evaluations.</span>';
        return models.map(m => `
            <label class="st-checkitem${this.selectedModelIds.includes(m.id) ? ' is-checked' : ''}">
                <input type="checkbox" class="report-model-check" value="${m.id}" ${this.selectedModelIds.includes(m.id) ? 'checked' : ''}>
                <span>${m.name}</span>
            </label>`).join('');
    },

    /** Sync eval selection from the checkboxes, refresh counts + the model list. */
    _syncEvalSelection() {
        const checks = document.querySelectorAll('#report-eval-list .report-eval-check');
        this.selectedEvalNames = Array.from(checks).filter(c => c.checked).map(c => c.value);
        checks.forEach(c => c.closest('.st-checkitem')?.classList.toggle('is-checked', c.checked));
        const cnt = document.getElementById('eval-select-count');
        if (cnt) cnt.textContent = `${this.selectedEvalNames.length} selected`;
        // The available models depend on the selected evals — re-render that list,
        // preserving still-valid model picks.
        const ml = document.getElementById('report-model-list');
        if (ml) ml.innerHTML = this._modelChecklistHTML();
        this._updateModelCount();
    },

    /** Sync model selection from the checkboxes + refresh count/info. */
    _syncModelSelection() {
        const checks = document.querySelectorAll('#report-model-list .report-model-check');
        this.selectedModelIds = Array.from(checks).filter(c => c.checked).map(c => c.value);
        checks.forEach(c => c.closest('.st-checkitem')?.classList.toggle('is-checked', c.checked));
        this._updateModelCount();
    },

    _updateModelCount() {
        const cnt = document.getElementById('model-select-count');
        if (cnt) cnt.textContent = `${(this.selectedModelIds || []).length} selected`;
        const info = document.getElementById('model-select-info');
        if (info) {
            info.innerHTML = (this.selectedModelIds || []).length
                ? ''
                : '<div class="st-warning-box"><i class="bi bi-exclamation-triangle"></i> Select at least one model for the report.</div>';
        }
    },

    /**
     * Get unique models from all evaluations (legacy, kept for compatibility)
     */
    getUniqueModels() {
        const models = new Map();

        this.evaluations.forEach(e => {
            (e.selected_models || []).forEach(m => {
                const id = m.id || m.model_id;
                if (id && !models.has(id)) {
                    models.set(id, {
                        id: id,
                        name: this.extractModelName(id)
                    });
                }
            });
        });

        return Array.from(models.values());
    },

    /** Status chip for a report row. */
    _reportStatusChip(st) {
        const map = {
            completed: ['success', 'ready'],
            generating: ['warning', 'generating…'],
            failed: ['error', 'failed'],
        };
        const [cls, label] = map[st] || ['neutral', st];
        return `<span class="ds-chip ds-chip--${cls}">${label}</span>`;
    },

    /** Render the actual item names inline as small wrapped chips. */
    _reportListCell(list, singular) {
        const items = (list || []).filter(Boolean);
        if (!items.length) return '<span class="st-caption">—</span>';
        // 'All' is the sentinel stored when the report wasn't scoped to specific items.
        if (items.length === 1 && items[0] === 'All') {
            return '<span class="ds-chip ds-chip--neutral">All</span>';
        }
        // Models are stored as full ids — show the short model name; evals are names as-is.
        const display = items.map(x => singular === 'model' ? this.extractModelName(x) : String(x));
        const chips = display.map(d => {
            const safe = String(d).replace(/"/g, '&quot;');
            return `<span class="ds-chip ds-chip--info report-name-chip" title="${safe}">${d}</span>`;
        }).join('');
        return `<div class="report-cell-chips">${chips}</div>`;
    },

    /**
     * Get selected report name
     */
    getSelectedReportName() {
        const report = this.reports.find(r => r.report_path === this.selectedReport);
        return report?.report_name || 'Unknown';
    },

    /**
     * Get selected report date
     */
    getSelectedReportDate() {
        const report = this.reports.find(r => r.report_path === this.selectedReport);
        return report?.creation_time_formatted || 'Unknown';
    },

    /**
     * Get selected report size
     */
    getSelectedReportSize() {
        const report = this.reports.find(r => r.report_path === this.selectedReport);
        return report?.file_size || 'Unknown';
    },

    /**
     * Get selected report filename
     */
    getSelectedReportFilename() {
        if (!this.selectedReport) return '';
        return this.selectedReport.split('/').pop();
    },

    /**
     * Bind events
     */
    bindEvents() {
        // Report scope radio — toggle panels without re-render
        document.querySelectorAll('input[name="report-scope"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.reportScope = e.target.value;
                const evalPanel = document.getElementById('eval-select-panel');
                const evalAllInfo = document.getElementById('eval-all-info');
                if (this.reportScope === 'selected') {
                    if (evalPanel) evalPanel.style.display = '';
                    if (evalAllInfo) evalAllInfo.style.display = 'none';
                } else {
                    if (evalPanel) evalPanel.style.display = 'none';
                    if (evalAllInfo) evalAllInfo.style.display = '';
                    this.selectedEvalNames = [];
                }
                // Available models depend on the eval scope — refresh the model list.
                const ml = document.getElementById('report-model-list');
                if (ml) ml.innerHTML = this._modelChecklistHTML();
                const ec = document.getElementById('eval-select-count');
                if (ec) ec.textContent = `${this.selectedEvalNames.length} selected`;
                this._updateModelCount();
            });
        });

        // Evaluation checkboxes (event delegation so it survives list re-renders).
        document.getElementById('report-eval-list')?.addEventListener('change', (ev) => {
            if (ev.target.classList.contains('report-eval-check')) this._syncEvalSelection();
        });

        // Model scope radio — toggle panels without re-render
        document.querySelectorAll('input[name="model-scope"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.modelScope = e.target.value;
                const modelPanel = document.getElementById('model-select-panel');
                const modelAllInfo = document.getElementById('model-all-info');
                if (this.modelScope === 'selected') {
                    if (modelPanel) modelPanel.style.display = '';
                    if (modelAllInfo) modelAllInfo.style.display = 'none';
                } else {
                    if (modelPanel) modelPanel.style.display = 'none';
                    if (modelAllInfo) modelAllInfo.style.display = '';
                }
            });
        });

        // Model checkboxes (delegation).
        document.getElementById('report-model-list')?.addEventListener('change', (ev) => {
            if (ev.target.classList.contains('report-model-check')) this._syncModelSelection();
        });

        // Select all / Clear buttons for both checklists.
        document.querySelectorAll('[data-checklist-all],[data-checklist-clear]').forEach(btn => {
            btn.addEventListener('click', () => {
                const all = 'checklistAll' in btn.dataset;
                const which = all ? btn.dataset.checklistAll : btn.dataset.checklistClear;
                const listSel = which === 'eval' ? '#report-eval-list .report-eval-check'
                                                 : '#report-model-list .report-model-check';
                document.querySelectorAll(listSel).forEach(c => { c.checked = all; });
                which === 'eval' ? this._syncEvalSelection() : this._syncModelSelection();
            });
        });

        // Report section checkboxes
        document.querySelectorAll('.report-section-check').forEach(cb => {
            cb.addEventListener('change', () => {
                const section = cb.dataset.section;
                if (cb.checked) {
                    if (!this.selectedSections.includes(section)) this.selectedSections.push(section);
                } else {
                    this.selectedSections = this.selectedSections.filter(s => s !== section);
                }
            });
        });

        // Generate report
        document.getElementById('generate-report-btn')?.addEventListener('click', async () => {
            await this.generateReport();
        });

        // Refresh button
        document.getElementById('refresh-reports-btn')?.addEventListener('click', async () => {
            await this.loadData();
            this.render();
            App.showNotification('Refreshed', 'Reports list refreshed');
        });

        // Click a report row to select it: completed -> view it; failed -> select for
        // deletion (no viewer); generating -> not yet ready.
        document.querySelectorAll('tr.report-row').forEach(row => {
            row.addEventListener('click', () => {
                const id = row.dataset.reportId;
                const path = row.dataset.reportPath;
                const st = row.dataset.status;
                if (st === 'generating') {
                    App.showNotification('Generating', 'This report is still being generated.', 'info');
                    return;
                }
                this.selectedReportId = id;
                this.selectedReport = (st === 'completed' && path) ? path : null;
                this.render();  // re-render: highlights the row + shows/hides the viewer
                const viewer = document.getElementById('report-iframe');
                if (viewer) viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        });

        // Download report button
        document.getElementById('download-report-btn')?.addEventListener('click', () => {
            if (this.selectedReport) {
                const link = document.createElement('a');
                link.href = `/api/reports/${this.selectedReport}`;
                link.download = this.getSelectedReportFilename() || 'report.html';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        });

        // Delete report button — acts on the row currently selected in the table.
        document.getElementById('delete-report-btn')?.addEventListener('click', async () => {
            if (!this.selectedReportId) {
                App.showNotification('Error', 'Click a report row to select it first', 'error');
                return;
            }
            if (confirm('Are you sure you want to delete this report?')) {
                await this.deleteReport(this.selectedReportId);
            }
        });
    },

    /**
     * Generate a new report
     */
    async generateReport() {
        let selectedEvaluations = null;
        let selectedModelIds = null;

        // Validate evaluation selection
        if (this.reportScope === 'selected') {
            selectedEvaluations = this.selectedEvalNames;

            if (selectedEvaluations.length === 0) {
                App.showNotification('Error', 'Please select at least one evaluation', 'error');
                return;
            }
        }

        // Validate model selection (now tracked in state from the checkbox list)
        if (this.modelScope === 'selected') {
            selectedModelIds = this.selectedModelIds || [];

            if (selectedModelIds.length === 0) {
                App.showNotification('Error', 'Please select at least one model', 'error');
                return;
            }
        }

        // Get summary model selection
        const summaryModel = document.getElementById('summary-model-select')?.value || 'bedrock/global.amazon.nova-2-lite-v1:0';
        const summaryRegion = document.getElementById('summary-region-select')?.value || 'us-east-1';

        try {
            App.showLoading(true);
            // Report generation now runs in the background — this returns immediately
            // with the report in a 'generating' state; we poll until it's ready.
            const result = await API.generateReport(selectedEvaluations, selectedModelIds, summaryModel, summaryRegion, this.selectedSections);
            App.showLoading(false);

            if (result && (result.status === 'generating' || result.success)) {
                App.showNotification('Report generating',
                    `"${result.report_name}" is being generated — it will appear below when ready.`);
                await this.loadData();
                this.render();
                this._startReportPolling();
            } else {
                App.showNotification('Error', (result && result.error) || 'Failed to generate report', 'error');
            }
        } catch (error) {
            App.showNotification('Error', `Failed to generate report: ${error.message}`, 'error');
        } finally {
            App.showLoading(false);
        }
    },

    /**
     * Poll the reports list every 5s while any report is still generating, and
     * re-render only when a report's status actually changes (so it doesn't reset
     * the form mid-setup). Stops automatically once nothing is generating.
     */
    _startReportPolling() {
        if (this.reportPollInterval) return;  // already polling
        this.reportPollInterval = setInterval(async () => {
            if (document.hidden) return;  // pause when the tab is hidden
            const sig = () => this.reports.map(r => `${r.report_id}:${r.status || 'completed'}`).join(',');
            const prevSig = sig();
            await this.loadData();
            if (sig() !== prevSig) this.render();
            if (!this.reports.some(r => (r.status || 'completed') === 'generating')) {
                this._stopReportPolling();
            }
        }, 5000);
    },

    _stopReportPolling() {
        if (this.reportPollInterval) {
            clearInterval(this.reportPollInterval);
            this.reportPollInterval = null;
        }
    },

    /**
     * Delete a report
     */
    async deleteReport(statusFile) {
        try {
            App.showLoading(true);
            await API.deleteReport(statusFile);

            App.showNotification('Success', 'Report deleted');
            this.selectedReport = null;
            this.selectedReportId = null;
            await this.loadData();
            this.render();
        } catch (error) {
            App.showNotification('Error', `Failed to delete report: ${error.message}`, 'error');
        } finally {
            App.showLoading(false);
        }
    },

    /**
     * Extract model name from full ID
     */
    extractModelName(modelId) {
        if (!modelId) return 'Unknown';
        let name = modelId.replace(/^bedrock\//, '');
        name = name.replace(/^[a-z]{2}\./, '');
        return name;
    }
};

// Export for use in other modules
window.ReportsComponent = ReportsComponent;
