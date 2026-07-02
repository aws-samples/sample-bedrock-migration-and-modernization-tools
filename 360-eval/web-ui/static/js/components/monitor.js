/**
 * Monitor tab component
 * Handles evaluation queue monitoring and execution
 * Styled to match Streamlit UI
 */

const MonitorComponent = {
    evaluations: [],
    queueStatus: null,
    refreshInterval: null,
    hasBedrockKey: false,

    /**
     * Initialize the monitor component
     */
    async init() {
        await this.loadEvaluations();
        await this.checkBedrockKey();
        this.render();
        this.startAutoRefresh();
    },

    /**
     * Check if user has a Bedrock API key saved
     */
    async checkBedrockKey() {
        // OFFLINE: Bedrock auth is "ready" if a key is stored OR AWS default creds
        // (SigV4) are available — the backend reports this as `bedrock_ready`.
        try {
            const res = await API.getCredentials();
            const creds = res.credentials || [];
            this.hasBedrockKey = (res.bedrock_ready === true)
                || creds.some(c => c.provider === 'bedrock');
        } catch (e) {
            this.hasBedrockKey = false;
        }
    },

    /**
     * Load evaluations from API
     */
    async loadEvaluations() {
        try {
            const [evalRes, queueRes] = await Promise.all([
                API.getEvaluations(),
                API.getQueueStatus()
            ]);
            this.evaluations = evalRes.evaluations || [];
            this.queueStatus = queueRes;
            State.set('evaluations', this.evaluations);
        } catch (error) {
            console.error('Failed to load evaluations:', error);
        }
    },

    /**
     * Compact signature of the queue (running eval id+status + queued ids) used to
     * detect when a job starts, finishes, or the queue changes.
     */
    _queueSignature(qs) {
        const cur = qs?.current_evaluation;
        const queued = qs?.queued_evaluations || [];
        return `${cur ? cur.id + ':' + cur.status : 'none'}|${queued.map(q => q.id).join(',')}`;
    },

    /**
     * Start auto-refresh for queue status
     */
    startAutoRefresh() {
        // Clear any existing interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }

        // Poll every 5s. The interval fetches ONLY the lightweight queue-status
        // endpoint and patches the progress bar in place. The full evaluations list
        // (large per-eval config payload) is refetched and the tab re-rendered only
        // when a job actually starts/finishes or the queue changes — detected via a
        // queue signature. This removes a full-table DynamoDB query + large JSON
        // payload from every 5s tick for every active user.
        this.refreshInterval = setInterval(async () => {
            // Pause polling when the browser tab is hidden or the monitor sub-tab
            // isn't the active one — no point hitting the API in the background.
            if (document.hidden) return;
            if (!document.getElementById('tab-monitor')?.classList.contains('active')) return;

            try {
                const prevSig = this._queueSignature(this.queueStatus);
                const queueRes = await API.getQueueStatus();
                this.queueStatus = queueRes;
                const newSig = this._queueSignature(queueRes);

                if (prevSig !== newSig) {
                    // A job started/finished or the queue changed — refresh the full
                    // list so started/completed rows reflect their new state, then render.
                    await this.loadEvaluations();
                    this.render();
                } else {
                    // Only progress moved — patch the DOM in place (no list refetch).
                    this._updateProgressInPlace();
                }
            } catch (e) {
                console.error('Monitor poll failed:', e);
            }
        }, 5000);
    },

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    },

    /**
     * Update progress bar and status in-place without full re-render
     */
    _updateProgressInPlace() {
        // Update the running evaluation progress bar
        const runningEval = this.queueStatus?.current_evaluation;
        if (runningEval) {
            const progressBar = document.querySelector('.ds-progress__bar');
            const progressText = document.querySelector('.ds-progress + .st-caption');
            if (progressBar) {
                progressBar.style.width = `${runningEval.progress || 0}%`;
            }
            if (progressText) {
                progressText.textContent = `Progress: ${runningEval.progress || 0}%`;
            }
            const statusMsg = document.getElementById('running-status-message');
            if (statusMsg) {
                statusMsg.textContent = runningEval.status_message || '';
            }
        }

        // Update progress in the evaluations table rows. The list (this.evaluations)
        // is only refetched on a status transition, so for the currently-running eval
        // take the live progress from queue-status instead of the stale list value.
        const runId = runningEval?.id;
        const runProg = runningEval?.progress;
        this.evaluations.forEach(e => {
            const row = document.querySelector(`tr[data-eval-id="${e.id}"]`);
            if (!row) return;
            const prog = (e.id === runId && runProg != null) ? runProg : (e.progress || 0);
            const progressSpan = row.querySelector('.eval-progress');
            if (progressSpan) {
                const status = (e.status || '').toLowerCase();
                if (status === 'running' || status === 'in-progress') {
                    progressSpan.textContent = `${prog}%`;
                }
            }
            // Update inline progress bar in the table
            const inlineBar = row.querySelector('.ds-progress__bar');
            if (inlineBar) {
                inlineBar.style.width = `${prog}%`;
            }
        });

        // Update timestamp
        const timestamp = document.querySelector('.st-timestamp');
        if (timestamp) {
            timestamp.textContent = `Last refreshed: ${new Date().toLocaleTimeString()}`;
        }
    },

    /**
     * Render the monitor tab (Streamlit-like layout)
     */
    render() {
        const container = document.getElementById('monitor-content');
        const lastRefresh = new Date().toLocaleTimeString();

        const runningEval = this.queueStatus?.current_evaluation;
        const queuedEvals = this.queueStatus?.queued_evaluations || [];

        // Filter evaluations by status (normalize to lowercase for comparison).
        // Terminal failure states (eval_failed, pre_eval_failed) are NOT runnable
        // — they require re-creating the eval, not re-running a broken one.
        const TERMINAL_STATUSES = ['completed', 'running', 'queued', 'in-progress',
                                   'eval_failed', 'pre_eval_failed'];
        const runnableEvals = this.evaluations.filter(e => {
            const status = (e.status || '').toLowerCase();
            return !TERMINAL_STATUSES.includes(status);
        });

        // Deletable: anything that's not currently running or queued.
        const DELETABLE_STATUSES = ['configuring', 'failed', 'completed',
                                    'eval_failed', 'pre_eval_failed'];
        const deletableEvals = this.evaluations.filter(e => {
            const status = (e.status || '').toLowerCase();
            return DELETABLE_STATUSES.includes(status);
        });

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <!-- Refresh Button and Timestamp -->
                <div class="st-header-row ds-flex ds-justify-between ds-items-center">
                    <button class="ds-btn ds-btn--secondary" id="refresh-monitor-btn">
                        <i class="bi bi-arrow-clockwise"></i> Refresh Evaluations
                    </button>
                    <span class="st-timestamp ds-text-muted">Last refreshed: ${lastRefresh}</span>
                </div>

                <!-- Queue Status (conditional) -->
                ${runningEval || queuedEvals.length > 0 ? `
                    <h3 class="st-subheader">Execution Queue Status</h3>

                    ${runningEval ? `
                        <div class="ds-card ds-card--accent ds-mb-3">
                            <div class="ds-flex ds-items-center ds-gap-3 ds-mb-3">
                                <span class="ds-chip ds-chip--accent ds-chip--running">Running</span>
                                <strong>${runningEval.name}</strong>
                            </div>
                            <div class="ds-progress ds-progress--animated">
                                <div class="ds-progress__bar" style="width: ${runningEval.progress || 0}%;"></div>
                            </div>
                            <p class="st-caption ds-mt-2">Progress: ${runningEval.progress || 0}%</p>
                            <p class="st-caption st-muted" id="running-status-message">${runningEval.status_message || ''}</p>
                        </div>
                    ` : ''}

                    ${queuedEvals.length > 0 ? `
                        <div class="ds-alert ds-alert--info ds-mb-3">
                            <i class="bi bi-hourglass-split ds-alert__icon"></i>
                            <div class="ds-alert__content">
                                <div class="ds-alert__message"><strong>Queued Evaluations:</strong> ${queuedEvals.length}</div>
                            </div>
                        </div>
                        <details class="st-expander">
                            <summary>View Queued Evaluations</summary>
                            <div class="st-expander-content">
                                ${queuedEvals.map((e, i) => `
                                    <p>${i + 1}. ${e.name}</p>
                                `).join('')}
                            </div>
                        </details>
                    ` : ''}

                    <hr class="ds-divider">
                ` : ''}

                <!-- Processing Evaluations - Show ALL evaluations -->
                <h3 class="st-subheader">Processing Evaluations</h3>
                ${this.evaluations.length > 0 ? `
                    <div class="ds-card ds-card--flush" style="max-height: 350px; overflow-y: auto;">
                        <table class="ds-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Task Type</th>
                                    <th>Models</th>
                                    <th>Stream</th>
                                    <th>Status</th>
                                    <th>Progress</th>
                                    <th>Created</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.evaluations.map(e => {
                                    const taskType = e.task_type || 'Latency Benchmark';
                                    const modelCount = e.selected_models?.length || 0;
                                    const stream = e.stream_evaluation ? 'Yes' : 'No';
                                    const statusLower = (e.status || '').toLowerCase();
                                    let statusDisplay = e.status ? e.status.charAt(0).toUpperCase() + e.status.slice(1).toLowerCase() : 'Unknown';

                                    let progress = '-';
                                    let progressBar = '';
                                    if (statusLower === 'running' || statusLower === 'in-progress') {
                                        const progressVal = e.progress || 0;
                                        progress = `${progressVal}%`;
                                        progressBar = `<div class="ds-progress ds-progress--animated" style="width: 80px; height: 6px; display: inline-block; vertical-align: middle; margin-left: 8px;"><div class="ds-progress__bar" style="width: ${progressVal}%;"></div></div>`;
                                    } else if (statusLower === 'queued') {
                                        progress = 'Queued';
                                    } else if (statusLower === 'completed') {
                                        progress = '100%';
                                    }

                                    const created = e.created_at ? new Date(e.created_at).toLocaleString() : 'N/A';

                                    // Map status to chip variant (use lowercase for lookup)
                                    const statusChipMap = {
                                        'running': 'ds-chip--accent ds-chip--running',
                                        'in-progress': 'ds-chip--accent ds-chip--running',
                                        'completed': 'ds-chip--success',
                                        'failed': 'ds-chip--error',
                                        'eval_failed': 'ds-chip--error',
                                        'pre_eval_failed': 'ds-chip--warning',
                                        'queued': 'ds-chip--info',
                                        'configuring': 'ds-chip--neutral'
                                    };
                                    const statusDisplayMap = {
                                        'eval_failed': 'Eval Failed',
                                        'pre_eval_failed': 'Config Error',
                                    };
                                    statusDisplay = statusDisplayMap[statusLower] || statusDisplay;
                                    const chipClass = statusChipMap[statusLower] || 'ds-chip--neutral';

                                    return `
                                        <tr data-eval-id="${e.id}">
                                            <td><strong>${e.name}</strong></td>
                                            <td>${taskType}</td>
                                            <td><span class="ds-chip ds-chip--neutral">${modelCount}</span></td>
                                            <td>${stream === 'Yes' ? '<span class="ds-chip ds-chip--success">Yes</span>' : '<span class="ds-chip ds-chip--neutral">No</span>'}</td>
                                            <td><span class="ds-chip ${chipClass}">${statusDisplay}</span></td>
                                            <td><span class="eval-progress">${progress}</span>${progressBar}</td>
                                            <td>${created}</td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : `
                    <div class="ds-empty">
                        <i class="bi bi-inbox ds-empty__icon"></i>
                        <div class="ds-empty__title">No Evaluations Available</div>
                        <div class="ds-empty__description">Go to the Setup tab to create new evaluations.</div>
                    </div>
                `}

                <hr class="ds-divider">

                <!-- Run Evaluations -->
                <h3 class="st-subheader">Run Evaluations</h3>
                ${!this.hasBedrockKey ? `
                    <div class="ds-alert ds-alert--warning ds-mb-3">
                        <i class="bi bi-exclamation-triangle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__title">Bedrock authentication required</div>
                            <div class="ds-alert__message">
                                No Bedrock authentication was found. Configure <strong>AWS credentials</strong>
                                (so the app can authenticate to Bedrock via SigV4) — or add a Bedrock API key
                                in the <strong>Credentials</strong> tab.
                                <br><br>
                                <strong>Note:</strong> with a short-term (temporary) API key, evaluations can only run against the AWS region where the key was generated.
                            </div>
                        </div>
                    </div>
                ` : ''}
                ${runnableEvals.length > 0 ? `
                    <div class="ds-card">
                        <label class="st-label">Select evaluations to run (will execute in order selected)</label>
                        <select class="st-multiselect" id="run-eval-select" multiple size="5">
                            ${runnableEvals.map(e => `
                                <option value="${e.id}">${e.name}</option>
                            `).join('')}
                        </select>
                        <p class="st-caption ds-mt-2">Hold Ctrl/Cmd to select multiple evaluations</p>
                        <div id="run-selected-info" style="margin-top: 0.5rem;"></div>
                        <button class="ds-btn ds-btn--primary ds-btn--lg ds-mt-3" id="run-evaluations-btn" ${!this.hasBedrockKey ? 'disabled style="opacity:0.5;cursor:not-allowed;"' : ''}>
                            <i class="bi bi-rocket-takeoff"></i> Execute Evaluation(s)
                        </button>
                    </div>
                ` : `
                    <div class="ds-alert ds-alert--info">
                        <i class="bi bi-info-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">No evaluations available to run. Create new evaluations in the Setup tab or wait for running evaluations to complete.</div>
                        </div>
                    </div>
                `}

                <hr class="ds-divider">

                <!-- Delete Evaluations -->
                <h3 class="st-subheader">Delete Evaluations</h3>
                ${deletableEvals.length > 0 ? `
                    <div class="ds-card">
                        <label class="st-label">Select evaluations to delete permanently</label>
                        <select class="st-multiselect" id="delete-eval-select" multiple size="5">
                            ${deletableEvals.map(e => `
                                <option value="${e.id}">${e.name} (${e.status})</option>
                            `).join('')}
                        </select>
                        <p class="st-caption ds-mt-2">This will permanently delete the evaluation configuration and status files.</p>
                        <div id="delete-warning" style="margin-top: 0.5rem;"></div>
                        <button class="ds-btn ds-btn--danger ds-mt-3" id="delete-evaluations-btn">
                            <i class="bi bi-trash-fill"></i> Delete Selected Evaluations
                        </button>
                    </div>
                ` : `
                    <div class="ds-alert ds-alert--info">
                        <i class="bi bi-info-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">No evaluations available for deletion.</div>
                        </div>
                    </div>
                `}
            </div>
        `;

        this.bindEvents();

        // Ensure multiselects start with no selection (browser compatibility fix)
        const runSelect = document.getElementById('run-eval-select');
        const deleteSelect = document.getElementById('delete-eval-select');
        if (runSelect) {
            runSelect.selectedIndex = -1;
            // Also clear any options that might be selected
            Array.from(runSelect.options).forEach(opt => opt.selected = false);
        }
        if (deleteSelect) {
            deleteSelect.selectedIndex = -1;
            // Also clear any options that might be selected
            Array.from(deleteSelect.options).forEach(opt => opt.selected = false);
        }
    },

    /**
     * Bind events for monitor tab
     */
    bindEvents() {
        // Refresh button
        document.getElementById('refresh-monitor-btn')?.addEventListener('click', async () => {
            const prevStatuses = this.evaluations.map(e => `${e.id}:${e.status}`).join(',');
            await this.loadEvaluations();
            const newStatuses = this.evaluations.map(e => `${e.id}:${e.status}`).join(',');
            if (prevStatuses !== newStatuses) {
                this.render();
            } else {
                this._updateProgressInPlace();
            }
            App.showNotification('Refreshed', 'Monitor data refreshed');
        });

        // Run evaluations select change - show info about selected
        document.getElementById('run-eval-select')?.addEventListener('change', (e) => {
            const select = e.target;
            const selectedCount = select.selectedOptions.length;
            const infoDiv = document.getElementById('run-selected-info');

            if (selectedCount > 0) {
                infoDiv.innerHTML = `<div class="st-info-box"><i class="bi bi-info-circle"></i> Selected ${selectedCount} evaluation(s).</div>`;
            } else {
                infoDiv.innerHTML = '';
            }
        });

        // Delete evaluations select change - show warning
        document.getElementById('delete-eval-select')?.addEventListener('change', (e) => {
            const select = e.target;
            const selectedCount = select.selectedOptions.length;
            const warningDiv = document.getElementById('delete-warning');

            if (selectedCount > 0) {
                warningDiv.innerHTML = `<div class="st-warning-box"><i class="bi bi-exclamation-triangle"></i> You are about to delete ${selectedCount} evaluation(s). This action cannot be undone.</div>`;
            } else {
                warningDiv.innerHTML = '';
            }
        });

        // Run evaluations
        document.getElementById('run-evaluations-btn')?.addEventListener('click', async () => {
            const select = document.getElementById('run-eval-select');
            const selectedIds = Array.from(select.selectedOptions).map(o => o.value);

            if (selectedIds.length === 0) {
                App.showNotification('Error', 'Please select at least one evaluation', 'error');
                return;
            }

            try {
                App.showLoading(true);
                const result = await API.runEvaluations(selectedIds);

                if (result.success) {
                    App.showNotification('Success', result.message);
                    await this.loadEvaluations();
                    this.render();
                } else {
                    App.showNotification('Error', result.error || 'Failed to run evaluations', 'error');
                }
            } catch (error) {
                App.showNotification('Error', `Failed to run evaluations: ${error.message}`, 'error');
            } finally {
                App.showLoading(false);
            }
        });

        // Delete evaluations
        document.getElementById('delete-evaluations-btn')?.addEventListener('click', async () => {
            const select = document.getElementById('delete-eval-select');
            const selectedIds = Array.from(select.selectedOptions).map(o => o.value);

            if (selectedIds.length === 0) {
                App.showNotification('Error', 'Please select at least one evaluation', 'error');
                return;
            }

            if (!confirm(`Are you sure you want to delete ${selectedIds.length} evaluation(s)? This cannot be undone.`)) {
                return;
            }

            try {
                App.showLoading(true);
                let deleted = 0;

                for (const evalId of selectedIds) {
                    try {
                        await API.deleteEvaluation(evalId);
                        deleted++;
                    } catch (error) {
                        console.error(`Failed to delete ${evalId}:`, error);
                    }
                }

                App.showNotification('Success', `Deleted ${deleted} evaluation(s)`);
                await this.loadEvaluations();
                this.render();
            } catch (error) {
                App.showNotification('Error', `Failed to delete evaluations: ${error.message}`, 'error');
            } finally {
                App.showLoading(false);
            }
        });
    }
};

// Export for use in other modules
window.MonitorComponent = MonitorComponent;
