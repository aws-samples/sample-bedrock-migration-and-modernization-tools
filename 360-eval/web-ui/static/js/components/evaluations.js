/**
 * Evaluations tab component
 * Displays completed evaluations and their details
 * Styled to match Streamlit UI
 */

const EvaluationsComponent = {
    evaluations: [],
    selectedEvaluation: null,

    /**
     * Initialize the evaluations component
     */
    async init() {
        await this.loadEvaluations();
        this.render();
    },

    /**
     * Load evaluations from API
     */
    async loadEvaluations() {
        try {
            const result = await API.getEvaluations();
            this.evaluations = result.evaluations || [];
        } catch (error) {
            console.error('Failed to load evaluations:', error);
        }
    },

    /**
     * Render an APO (prompt optimization) outcome chip from an eval's apo_status.
     * Returns '' when APO wasn't used. Tooltip carries the detail message.
     */
    apoBadge(e) {
        const s = e && e.apo_status;
        if (!s) return '';
        const map = {
            applied: ['success', 'APO ✓'],
            partial: ['warning', 'APO partial'],
            failed:  ['error',   'APO failed'],
            skipped: ['neutral', 'APO skipped'],
        };
        const [cls, label] = map[s] || ['neutral', `APO ${s}`];
        const tip = (e.apo_message || '').replace(/"/g, '&quot;');
        return ` <span class="ds-chip ds-chip--${cls}" style="margin-left:0.25rem;font-size:0.7rem;" title="${tip}">${label}</span>`;
    },

    /**
     * Render the evaluations tab (Streamlit-like layout)
     */
    render() {
        const container = document.getElementById('evaluations-content');
        const completedEvals = this.evaluations.filter(e =>
            e.status === 'completed' || e.status === 'failed'
        );

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <!-- Completed Evaluations -->
                <h3 class="st-subheader" style="margin-top: 0;">Completed Evaluations</h3>
                ${completedEvals.length > 0 ? '<p class="st-caption ds-mb-2">Click a row to view its details.</p>' : ''}
                ${completedEvals.length > 0 ? `
                    <div class="ds-card ds-card--flush" style="max-height: 300px; overflow-y: auto;">
                        <table class="ds-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Status</th>
                                    <th>Task Type</th>
                                    <th>Created</th>
                                    <th>Duration</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${completedEvals.map(e => `
                                    <tr class="eval-row${this.selectedEvaluation?.id === e.id ? ' is-selected' : ''}" data-eval-id="${e.id}">
                                        <td>${e.name}</td>
                                        <td><span class="ds-chip ds-chip--${e.status === 'completed' ? 'success' : 'error'}">${e.status}</span>${this.apoBadge(e)}</td>
                                        <td>${e.task_type || 'N/A'}</td>
                                        <td>${e.created_at ? new Date(e.created_at).toLocaleDateString() : 'N/A'}</td>
                                        <td>${e.duration ? this.formatDuration(e.duration) : 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <button class="ds-btn ds-btn--secondary ds-mt-3" id="refresh-evaluations-btn">
                        <i class="bi bi-arrow-clockwise"></i> Refresh
                    </button>
                ` : `
                    <div class="ds-empty">
                        <i class="bi bi-inbox ds-empty__icon"></i>
                        <div class="ds-empty__title">No Completed Evaluations</div>
                        <div class="ds-empty__description">Run some evaluations to see results here.</div>
                    </div>
                `}

                <!-- Selected Evaluation Details -->
                <div id="eval-details-container">
                    ${this.selectedEvaluation ? this.renderEvaluationDetails(this.selectedEvaluation) : ''}
                </div>
            </div>
        `;

        this.bindEvents();
    },

    /**
     * Render evaluation details (Streamlit-like layout)
     */
    renderEvaluationDetails(evaluation) {
        const hasUnprocessed = evaluation.unprocessed_files?.length > 0;

        return `
            <hr class="ds-divider ds-divider--accent">

            <!-- Unprocessed Warning (conditional) -->
            ${hasUnprocessed ? `
                <div class="st-columns-2" style="align-items: center;">
                    <div>
                        <div class="ds-alert ds-alert--warning">
                            <i class="bi bi-exclamation-triangle ds-alert__icon"></i>
                            <div class="ds-alert__content">
                                <div class="ds-alert__message">This evaluation has ${evaluation.unprocessed_files.length} unprocessed record file(s).</div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <button class="ds-btn ds-btn--secondary" id="goto-unprocessed-btn">
                            View Unprocessed Records
                        </button>
                    </div>
                </div>
            ` : ''}

            <!-- Evaluation Details -->
            <h3 class="st-subheader">Evaluation Details</h3>
            <div class="ds-card ds-card--accent ds-mb-4">
                <div class="st-columns-2">
                    <div>
                        <p><strong>Name:</strong> ${evaluation.name}</p>
                        <p><strong>Status:</strong> <span class="ds-chip ds-chip--${evaluation.status === 'completed' ? 'success' : 'error'}">${evaluation.status}</span></p>
                        ${evaluation.apo_status ? `<p><strong>Prompt Optimization:</strong> ${this.apoBadge(evaluation)}${evaluation.apo_message ? ` <span class="st-caption">${evaluation.apo_message}</span>` : ''}</p>` : ''}
                        <p><strong>Task Type:</strong> ${evaluation.task_type || 'N/A'}</p>
                        <p><strong>Task Criteria:</strong> ${evaluation.task_criteria || 'N/A'}</p>
                    </div>
                    <div>
                        <p><strong>Created:</strong> ${evaluation.created_at ? new Date(evaluation.created_at).toLocaleString() : 'N/A'}</p>
                        <p><strong>Duration:</strong> ${evaluation.duration ? this.formatDuration(evaluation.duration) : 'N/A'}</p>
                        <p><strong>CSV File:</strong> ${evaluation.csv_file_name || 'N/A'}</p>
                        <p><strong>Streaming:</strong> <span class="ds-chip ds-chip--${evaluation.stream_evaluation ? 'success' : 'neutral'}">${evaluation.stream_evaluation ? 'Enabled' : 'Disabled'}</span></p>
                    </div>
                </div>
            </div>

            <!-- Error Display (conditional) -->
            ${evaluation.error ? `
                <div class="ds-alert ds-alert--error ds-mb-4">
                    <i class="bi bi-x-circle ds-alert__icon"></i>
                    <div class="ds-alert__content">
                        <div class="ds-alert__title">Error</div>
                        <div class="ds-alert__message">${evaluation.error}</div>
                    </div>
                </div>
            ` : ''}

            <!-- Optimized Prompts (APO) — shown when optimization produced output -->
            ${['applied', 'partial'].includes(evaluation.apo_status) ? `
                <h4 class="st-section-header">Optimized Prompts (APO) ${this.apoBadge(evaluation)}</h4>
                <div class="ds-card ds-card--flush ds-mb-4" style="padding:0.75rem;">
                    <div id="apo-artifacts">Loading optimized-prompt files…</div>
                </div>
            ` : ''}

            <!-- Models Evaluated -->
            ${evaluation.selected_models?.length > 0 ? `
                <h4 class="st-section-header">Models Evaluated (${evaluation.selected_models.length})</h4>
                <div class="ds-card ds-card--flush" style="max-height: 200px; overflow-y: auto;">
                    <table class="ds-table ds-table--compact">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Region</th>
                                <th>Input Cost</th>
                                <th>Output Cost</th>
                                <th>Service Tier</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${evaluation.selected_models.map(m => `
                                <tr>
                                    <td>${this.extractModelName(m.id || m.model_id)}</td>
                                    <td><span class="ds-chip ds-chip--info">${m.region}</span></td>
                                    <td>$${m.input_cost || m.input_token_cost || 0}</td>
                                    <td>$${m.output_cost || m.output_token_cost || 0}</td>
                                    <td><span class="ds-chip ds-chip--neutral">${m.service_tier || 'default'}</span></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : ''}

            <!-- Judge Configuration -->
            ${evaluation.eval_mode === 'specialist' && evaluation.metric_assignments ? `
                <h4 class="st-section-header">
                    Judge Configuration
                    <span class="ds-chip ds-chip--primary" style="margin-left: 0.5rem; font-size: 0.7rem;">Specialist</span>
                    ${evaluation.has_self_eval_judges ? '<span class="ds-chip ds-chip--warning" style="margin-left: 0.25rem; font-size: 0.7rem;" title="One or more judge models share the same family as the target model">Self-Eval Warning</span>' : ''}
                </h4>
                <div class="ds-card ds-card--flush" style="max-height: 200px; overflow-y: auto;">
                    <table class="ds-table ds-table--compact">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Primary Model</th>
                                <th>Secondary</th>
                                <th>Threshold</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(evaluation.metric_assignments).map(([metric, a]) => `
                                <tr>
                                    <td><strong>${metric}</strong></td>
                                    <td>${a.primary ? this.extractModelName(a.primary.id || a.primary.model_id) : '-'}</td>
                                    <td>${a.secondary ? this.extractModelName(a.secondary.id || a.secondary.model_id) : '-'}</td>
                                    <td>${a.threshold ?? 3}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : evaluation.judge_models?.length > 0 ? `
                <h4 class="st-section-header">
                    Judge Models (${evaluation.judge_models.length})
                    <span class="ds-chip ds-chip--secondary" style="margin-left: 0.5rem; font-size: 0.7rem;">Bundled</span>
                    ${evaluation.has_self_eval_judges ? '<span class="ds-chip ds-chip--warning" style="margin-left: 0.25rem; font-size: 0.7rem;" title="One or more judge models share the same family as the target model">Self-Eval Warning</span>' : ''}
                </h4>
                <div class="ds-card ds-card--flush" style="max-height: 150px; overflow-y: auto;">
                    <table class="ds-table ds-table--compact">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Region</th>
                                <th>Input Cost</th>
                                <th>Output Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${evaluation.judge_models.map(j => `
                                <tr>
                                    <td>${this.extractModelName(j.id || j.model_id)}</td>
                                    <td><span class="ds-chip ds-chip--info">${j.region}</span></td>
                                    <td>$${j.input_cost || j.input_token_cost || 0}</td>
                                    <td>$${j.output_cost || j.output_token_cost || 0}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : ''}

            <!-- Evaluation Configuration -->
            <h4 class="st-section-header">Evaluation Configuration</h4>
            <div class="st-columns-4 ds-mb-4">
                <div class="ds-metric ds-metric--compact">
                    <div class="ds-metric__label">Parallel Calls</div>
                    <div class="ds-metric__value">${evaluation.parallel_calls || 'N/A'}</div>
                </div>
                <div class="ds-metric ds-metric--compact">
                    <div class="ds-metric__label">Invocations/Scenario</div>
                    <div class="ds-metric__value">${evaluation.invocations_per_scenario || 'N/A'}</div>
                </div>
                <div class="ds-metric ds-metric--compact">
                    <div class="ds-metric__label">Experiment Counts</div>
                    <div class="ds-metric__value">${evaluation.experiment_counts || 'N/A'}</div>
                </div>
                <div class="ds-metric ds-metric--compact">
                    <div class="ds-metric__label">Temperature</div>
                    <div class="ds-metric__value">${evaluation.temperature || 'N/A'}</div>
                </div>
            </div>

            <!-- RPM Metrics (conditional) -->
            ${evaluation.rpm_metrics ? `
                <h4 class="st-section-header">RPM Metrics</h4>
                <div class="ds-card ds-card--flush">
                    <table class="ds-table ds-table--compact">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Target RPM</th>
                                <th>Actual RPM</th>
                                <th>Requests</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${Object.entries(evaluation.rpm_metrics).map(([model, metrics]) => `
                                <tr>
                                    <td>${this.extractModelName(model)}</td>
                                    <td>${metrics.target_rpm || 'N/A'}</td>
                                    <td>${metrics.actual_rpm?.toFixed(2) || 'N/A'}</td>
                                    <td>${metrics.total_requests || 'N/A'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <details class="st-expander ds-mt-2">
                    <summary>View RPM Details</summary>
                    <div class="st-expander-content">
                        <pre style="font-size: 0.8rem; max-height: 200px; overflow: auto; background: var(--bg-tertiary); padding: 1rem; border-radius: var(--radius-md);">${JSON.stringify(evaluation.rpm_metrics, null, 2)}</pre>
                    </div>
                </details>
            ` : ''}

            <hr class="ds-divider">

            <!-- Load Config Button -->
            <div class="st-columns-3">
                <div>
                    <button class="ds-btn ds-btn--primary ds-btn--lg" id="load-config-btn" style="width: 100%;">
                        <i class="bi bi-box-arrow-in-down"></i> Load This Configuration
                    </button>
                </div>
                <div></div>
                <div></div>
            </div>
        `;
    },

    /**
     * Bind events
     */
    bindEvents() {
        // Refresh button
        document.getElementById('refresh-evaluations-btn')?.addEventListener('click', async () => {
            await this.loadEvaluations();
            this.render();
            App.showNotification('Refreshed', 'Evaluations list refreshed');
        });

        // Click a row in the Completed Evaluations table to view its details.
        document.querySelectorAll('tr.eval-row').forEach(row => {
            row.addEventListener('click', () => {
                const evalId = row.dataset.evalId;
                this.selectedEvaluation = this.evaluations.find(ev => ev.id === evalId) || null;
                // Highlight the active row without a full re-render.
                document.querySelectorAll('tr.eval-row').forEach(r =>
                    r.classList.toggle('is-selected', r.dataset.evalId === evalId));
                const detailsContainer = document.getElementById('eval-details-container');
                if (detailsContainer) {
                    detailsContainer.innerHTML = this.selectedEvaluation
                        ? this.renderEvaluationDetails(this.selectedEvaluation)
                        : '';
                    this._bindDetailsEvents();
                    detailsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            });
        });

        this._bindDetailsEvents();
    },

    /**
     * Fetch the APO artifact list for the selected eval and render download links.
     */
    async _populateApoArtifacts(evalId) {
        const el = document.getElementById('apo-artifacts');
        if (!el) return;
        try {
            const res = await API.getApoArtifacts(evalId);
            const items = (res && res.artifacts) || [];
            if (!items.length) {
                el.innerHTML = '<span class="st-caption">No optimized-prompt files found for this evaluation.</span>';
                return;
            }
            el.innerHTML = items.map(a => {
                const href = API.apoArtifactUrl(evalId, a.name);
                return `<div class="ds-flex ds-items-center ds-gap-2 ds-mb-1">
                    <i class="bi bi-download"></i>
                    <a href="${href}" download>${a.name}</a>
                    <span class="st-caption">(${a.size_human || ''})</span>
                </div>`;
            }).join('');
        } catch (e) {
            el.innerHTML = `<span class="st-caption">Could not load optimized-prompt files: ${e.message}</span>`;
        }
    },

    /**
     * Bind events for the details panel (called after details update)
     */
    _bindDetailsEvents() {
        // Populate APO download links when the selected eval produced optimized prompts.
        if (this.selectedEvaluation && ['applied', 'partial'].includes(this.selectedEvaluation.apo_status)) {
            this._populateApoArtifacts(this.selectedEvaluation.id);
        }

        // Go to unprocessed button
        document.getElementById('goto-unprocessed-btn')?.addEventListener('click', () => {
            App.switchTab('unprocessed');
        });

        // Load config button
        document.getElementById('load-config-btn')?.addEventListener('click', () => {
            if (this.selectedEvaluation) {
                State.loadFromEvaluation(this.selectedEvaluation);
                App.showNotification('Configuration Loaded', 'Please upload a CSV file and review settings in the Setup tab.');
                App.switchTab('setup');
            }
        });
    },

    /**
     * Format duration in seconds to human readable
     */
    formatDuration(seconds) {
        if (!seconds) return 'N/A';

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        const parts = [];
        if (hours > 0) parts.push(`${hours}h`);
        if (minutes > 0) parts.push(`${minutes}m`);
        parts.push(`${secs}s`);

        return parts.join(' ');
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
window.EvaluationsComponent = EvaluationsComponent;
