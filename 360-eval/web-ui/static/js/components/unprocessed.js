/**
 * Unprocessed tab component
 * Displays failed/unprocessed records grouped by evaluation and error type
 * Uses summary-first loading: lightweight summaries on init, full records on demand
 */

const UnprocessedComponent = {
    data: null,
    expandedEval: null,
    loadedDetails: {},  // eval_id → records array (loaded on demand)

    async init() {
        await this.loadData();
        this.render();
    },

    async loadData() {
        try {
            this.data = await API.getUnprocessed();
        } catch (error) {
            console.error('Failed to load unprocessed data:', error);
            this.data = { eval_summaries: [], summary: {} };
        }
    },

    async loadDetailForEval(evalId) {
        if (this.loadedDetails[evalId]) return this.loadedDetails[evalId];
        try {
            const result = await API.getUnprocessedDetail(evalId);
            this.loadedDetails[evalId] = result.records || [];
            return this.loadedDetails[evalId];
        } catch (error) {
            console.error(`Failed to load unprocessed detail for ${evalId}:`, error);
            return [];
        }
    },

    classifyError(reason) {
        if (!reason) return { type: 'Unknown', hint: '' };
        const r = reason.toLowerCase();

        if (r.includes('authenticationerror') || r.includes('bearer token') || r.includes('expired') || r.includes('invalid credential'))
            return { type: 'Authentication Error', hint: 'Update your API key in the Credentials tab.' };
        if (r.includes('ratelimit') || r.includes('rate limit') || r.includes('throttl'))
            return { type: 'Rate Limit', hint: 'Reduce parallel calls or add a target RPM limit.' };
        if (r.includes('timeout') || r.includes('timed out'))
            return { type: 'Timeout', hint: 'The model took too long to respond. Try again or use a faster model.' };
        if (r.includes('serviceunav') || r.includes('service unavailable') || r.includes('503'))
            return { type: 'Service Unavailable', hint: 'The model service is temporarily down. Try again later.' };
        if (r.includes('badrequest') || r.includes('bad request') || r.includes('400'))
            return { type: 'Bad Request', hint: 'The request was malformed. Check your prompt or model configuration.' };
        if (r.includes('accessdenied') || r.includes('access denied') || r.includes('not authorized'))
            return { type: 'Access Denied', hint: 'Your API key does not have access to this model or region.' };
        if (r.includes('connection') || r.includes('apiconnection'))
            return { type: 'Connection Error', hint: 'Network issue reaching the model API. Try again.' };
        if (r.includes('notfound') || r.includes('not found') || r.includes('404'))
            return { type: 'Model Not Found', hint: 'The model ID may be incorrect or not available in this region.' };
        if (r.includes('malformedjudgeresponseerror'))
            return { type: 'Malformed Judge Response', hint: 'The judge model responded but did not return valid JSON scores. Try a different judge model.' };
        if (r.includes('judge') || r.includes('parsing'))
            return { type: 'Judge Error', hint: 'The judge model failed to evaluate. Check judge model availability.' };

        return { type: 'API Error', hint: '' };
    },

    render() {
        const container = document.getElementById('unprocessed-content');
        const summaries = this.data?.eval_summaries || [];

        if (summaries.length === 0) {
            container.innerHTML = `
                <div class="streamlit-section ds-fade-in">
                    <div class="ds-alert ds-alert--success">
                        <i class="bi bi-check-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__title">All Clear!</div>
                            <div class="ds-alert__message">No unprocessed records found. All evaluations completed successfully.</div>
                        </div>
                    </div>
                </div>`;
            return;
        }

        const summary = this.data.summary;

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 class="st-subheader" style="margin-top: 0;">Unprocessed Records</h3>
                    <button class="ds-btn ds-btn--secondary" id="refresh-unprocessed">
                        <i class="bi bi-arrow-clockwise"></i> Refresh
                    </button>
                </div>

                <!-- Summary Stats -->
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Unprocessed Files</div>
                        <div class="ds-metric__value ds-text-error">${summary.total_files || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Affected Evaluations</div>
                        <div class="ds-metric__value">${summary.affected_experiments || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Status</div>
                        <div class="ds-metric__value" style="font-size: 0.9rem;">Click to expand</div>
                    </div>
                </div>

                <!-- Evaluation Groups -->
                ${summaries.map(es => {
                    const isExpanded = this.expandedEval === es.eval_id;
                    const records = this.loadedDetails[es.eval_id] || [];
                    const hasLoaded = !!this.loadedDetails[es.eval_id];

                    return `
                        <div class="ds-card ds-mb-3">
                            <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" class="eval-group-header" data-eval-id="${es.eval_id}">
                                <div>
                                    <strong style="font-size: 1.05rem;">${this.cleanExperimentName(es.eval_name)}</strong>
                                    <span class="ds-chip ds-chip--error ds-ml-2">${es.file_count} file(s)</span>
                                    <span style="color: var(--text-secondary); font-size: 0.8rem; margin-left: 0.5rem;">${es.created_at ? new Date(es.created_at).toLocaleDateString() : ''}</span>
                                </div>
                                <i class="bi bi-chevron-${isExpanded ? 'up' : 'down'}"></i>
                            </div>

                            <div id="eval-detail-${es.eval_id}" style="display: ${isExpanded ? 'block' : 'none'}; margin-top: 1rem;">
                                ${isExpanded && !hasLoaded ? '<div style="text-align: center; padding: 1rem; color: var(--text-secondary);"><i class="bi bi-hourglass-split"></i> Loading records...</div>' : ''}
                                ${isExpanded && hasLoaded ? this._renderEvalDetail(records) : ''}
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;

        this.bindEvents();
    },

    _renderEvalDetail(records) {
        if (records.length === 0) {
            return '<div style="color: var(--text-secondary); padding: 0.5rem;">No records found in file.</div>';
        }

        const byError = {};
        const byModel = {};
        for (const r of records) {
            const scenario = r.scenario || {};
            const reason = r.reason || '';
            const { type, hint } = this.classifyError(reason);
            if (!byError[type]) byError[type] = { count: 0, hint, example: reason };
            byError[type].count++;

            const modelId = scenario.model_id || 'Unknown';
            const shortModel = this.extractModelName(modelId);
            if (!byModel[shortModel]) byModel[shortModel] = { total: 0, errors: {} };
            byModel[shortModel].total++;
            byModel[shortModel].errors[type] = (byModel[shortModel].errors[type] || 0) + 1;
        }

        return `
            <!-- Error Types -->
            <h4 class="st-section-header" style="margin-top: 0;">Errors (${records.length} records)</h4>
            ${Object.entries(byError).map(([errorType, info]) => `
                <div style="display: flex; align-items: start; gap: 0.75rem; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color, #333);">
                    <span class="ds-chip ds-chip--error">${info.count}</span>
                    <div>
                        <strong>${errorType}</strong>
                        ${info.hint ? `<p style="color: var(--text-secondary); font-size: 0.8rem; margin: 0.25rem 0 0 0;"><i class="bi bi-lightbulb"></i> ${info.hint}</p>` : ''}
                    </div>
                </div>
            `).join('')}

            <!-- Model Breakdown -->
            <h4 class="st-section-header ds-mt-3">Model Breakdown</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                ${Object.entries(byModel).map(([model, info]) => {
                    const errorTypes = Object.entries(info.errors).map(([t, c]) => `${t}: ${c}`).join(', ');
                    return `
                        <div class="ds-card" style="padding: 0.5rem 0.75rem; min-width: 200px;">
                            <strong>${model}</strong>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">${info.total} failure(s) — ${errorTypes}</div>
                        </div>
                    `;
                }).join('')}
            </div>

            <!-- Raw Details (collapsed) -->
            <details class="st-expander ds-mt-3">
                <summary>Raw Records (${records.length})</summary>
                <div class="st-expander-content">
                    <div style="max-height: 300px; overflow-y: auto;">
                        <table class="ds-table ds-table--compact">
                            <thead>
                                <tr><th>Model</th><th>Region</th><th>Error Type</th><th>Reason</th><th>Metadata</th></tr>
                            </thead>
                            <tbody>
                                ${records.slice(0, 100).map(r => {
                                    const scenario = r.scenario || {};
                                    const result = r.result || {};
                                    const model = this.extractModelName(scenario.model_id || '');
                                    const region = scenario.region || 'N/A';
                                    const { type } = this.classifyError(r.reason);
                                    const metadata = {
                                        task_type: scenario.task_types || '',
                                        temperature: scenario.TEMPERATURE || scenario.temperature || '',
                                        invocation: r.invocation || '',
                                        error_classification: r.error_classification || '',
                                        api_status: result.api_call_status || '',
                                    };
                                    const metaStr = Object.entries(metadata).filter(([,v]) => v !== '' && v !== undefined).map(([k,v]) => `${k}: ${v}`).join('\n');
                                    return `
                                        <tr>
                                            <td><strong>${model}</strong></td>
                                            <td><span class="ds-chip ds-chip--info">${region}</span></td>
                                            <td><span class="ds-chip ds-chip--error">${type}</span></td>
                                            <td title="${(r.reason || '').replace(/"/g, '&quot;')}" style="font-size: 0.8rem;">${this.truncate(r.reason || '', 60)}</td>
                                            <td><pre style="font-size: 0.7rem; margin:0; white-space:pre-wrap; max-width:200px; color:var(--text-secondary);">${metaStr}</pre></td>
                                        </tr>
                                    `;
                                }).join('')}
                                ${records.length > 100 ? `<tr><td colspan="5" style="text-align:center; color:var(--text-secondary);">Showing first 100 of ${records.length} records</td></tr>` : ''}
                            </tbody>
                        </table>
                    </div>
                </div>
            </details>
        `;
    },

    bindEvents() {
        // Expand/collapse eval groups — load detail on demand
        document.querySelectorAll('.eval-group-header').forEach(header => {
            header.addEventListener('click', async () => {
                const evalId = header.dataset.evalId;
                if (this.expandedEval === evalId) {
                    // Collapse
                    this.expandedEval = null;
                    const detail = document.getElementById(`eval-detail-${evalId}`);
                    if (detail) detail.style.display = 'none';
                    header.querySelector('i').className = 'bi bi-chevron-down';
                } else {
                    // Collapse previous
                    if (this.expandedEval) {
                        const prev = document.getElementById(`eval-detail-${this.expandedEval}`);
                        if (prev) prev.style.display = 'none';
                        document.querySelector(`.eval-group-header[data-eval-id="${this.expandedEval}"] i`)?.setAttribute('class', 'bi bi-chevron-down');
                    }
                    // Expand new
                    this.expandedEval = evalId;
                    const detail = document.getElementById(`eval-detail-${evalId}`);
                    if (detail) {
                        detail.style.display = 'block';
                        detail.innerHTML = '<div style="text-align: center; padding: 1rem; color: var(--text-secondary);"><i class="bi bi-hourglass-split"></i> Loading records...</div>';
                    }
                    header.querySelector('i').className = 'bi bi-chevron-up';

                    // Load on demand
                    const records = await this.loadDetailForEval(evalId);
                    if (detail && this.expandedEval === evalId) {
                        detail.innerHTML = this._renderEvalDetail(records);
                    }
                }
            });
        });

        // Refresh
        document.getElementById('refresh-unprocessed')?.addEventListener('click', async () => {
            this.loadedDetails = {};
            this.expandedEval = null;
            await this.loadData();
            this.render();
            App.showNotification('Refreshed', 'Unprocessed data refreshed');
        });
    },

    cleanExperimentName(name) {
        if (!name) return 'Unknown';
        return name.replace(/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_/, '');
    },

    extractModelName(modelId) {
        if (!modelId) return 'Unknown';
        let name = modelId.replace(/^bedrock\//, '').replace(/^[a-z]{2}\./, '');
        return name;
    },

    truncate(text, length) {
        if (!text || text.length <= length) return text || '';
        return text.substring(0, length) + '...';
    }
};

window.UnprocessedComponent = UnprocessedComponent;
