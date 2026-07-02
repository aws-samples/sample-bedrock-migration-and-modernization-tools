/**
 * Admin dashboard component
 * Read-only overview of all platform activity
 * Only visible to admin users
 */

const AdminComponent = {
    data: null,
    isAdmin: false,
    statusFilter: 'all',

    async init() {
        await this.checkAdmin();
        if (this.isAdmin) {
            await this.loadData();
        }
        this.render();
    },

    async checkAdmin() {
        try {
            const res = await API.get('/api/admin/check');
            this.isAdmin = res?.is_admin || false;
        } catch (e) {
            this.isAdmin = false;
        }
    },

    async loadData() {
        try {
            this.data = await API.get('/api/admin/dashboard');
        } catch (e) {
            console.error('Failed to load admin data:', e);
            this.data = null;
        }
    },

    render() {
        const container = document.getElementById('admin-content');
        if (!container) return;

        if (!this.isAdmin) {
            container.innerHTML = `
                <div class="ds-alert ds-alert--warning ds-mt-4">
                    <i class="bi bi-shield-lock ds-alert__icon"></i>
                    <div class="ds-alert__content">
                        <div class="ds-alert__title">Access Denied</div>
                        <div class="ds-alert__message">Admin access is required to view this page.</div>
                    </div>
                </div>`;
            return;
        }

        if (!this.data) {
            container.innerHTML = '<p style="color:var(--text-secondary);">Loading...</p>';
            return;
        }

        const stats = this.data.stats || {};
        const users = this.data.users || [];
        const evals = this.data.evaluations || [];
        const activeJobs = this.data.active_jobs || [];

        const filteredEvals = this.statusFilter === 'all'
            ? evals
            : evals.filter(e => e.status === this.statusFilter);

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 class="st-subheader" style="margin-top: 0;">
                        <i class="bi bi-speedometer2"></i> Platform Overview
                    </h3>
                    <button class="ds-btn ds-btn--secondary" id="admin-refresh-btn">
                        <i class="bi bi-arrow-clockwise"></i> Refresh
                    </button>
                </div>

                <!-- Platform Stats -->
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Total Users</div>
                        <div class="ds-metric__value">${stats.total_users || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Total Evaluations</div>
                        <div class="ds-metric__value">${stats.total_evaluations || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Total Reports</div>
                        <div class="ds-metric__value">${stats.total_reports || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">This Week</div>
                        <div class="ds-metric__value">${stats.evals_this_week || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">This Month</div>
                        <div class="ds-metric__value">${stats.evals_this_month || 0}</div>
                    </div>
                    <div class="ds-metric ds-metric--compact">
                        <div class="ds-metric__label">Active Jobs</div>
                        <div class="ds-metric__value" style="color: ${stats.active_jobs > 0 ? 'var(--accent-orange)' : 'inherit'};">${stats.active_jobs || 0}</div>
                    </div>
                </div>

                <!-- Active Jobs -->
                ${activeJobs.length > 0 ? `
                    <h3 class="st-subheader"><i class="bi bi-activity"></i> Active Jobs</h3>
                    <div class="ds-card ds-card--accent ds-mb-4 ds-card--flush">
                        <table class="ds-table ds-table--compact">
                            <thead>
                                <tr><th>User</th><th>Evaluation</th><th>Status</th><th>Progress</th><th>Started</th></tr>
                            </thead>
                            <tbody>
                                ${activeJobs.map(j => `
                                    <tr>
                                        <td><span class="ds-chip ds-chip--info">${j.user_id}</span></td>
                                        <td><strong>${j.eval_name}</strong></td>
                                        <td><span class="ds-chip ds-chip--accent ds-chip--running">${j.status}</span></td>
                                        <td>
                                            ${j.progress || 0}%
                                            <div class="ds-progress ds-progress--animated" style="width:80px;height:6px;display:inline-block;vertical-align:middle;margin-left:8px;">
                                                <div class="ds-progress__bar" style="width:${j.progress || 0}%;"></div>
                                            </div>
                                        </td>
                                        <td>${j.created_at ? new Date(j.created_at).toLocaleString() : 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : ''}

                <!-- Users -->
                <h3 class="st-subheader"><i class="bi bi-people"></i> Users (${users.length})</h3>
                <div class="ds-card ds-card--flush ds-mb-4" style="max-height: 300px; overflow-y: auto;">
                    <table class="ds-table ds-table--compact">
                        <thead>
                            <tr><th>User</th><th>Evaluations</th><th>Completed</th><th>Failed</th><th>Running</th><th>Reports</th><th>Last Activity</th></tr>
                        </thead>
                        <tbody>
                            ${users.map(u => `
                                <tr>
                                    <td><strong>${u.user_id}</strong></td>
                                    <td>${u.evaluations}</td>
                                    <td><span class="ds-chip ds-chip--success">${u.completed}</span></td>
                                    <td>${u.failed > 0 ? `<span class="ds-chip ds-chip--error">${u.failed}</span>` : '0'}</td>
                                    <td>${u.running > 0 ? `<span class="ds-chip ds-chip--accent ds-chip--running">${u.running}</span>` : '0'}</td>
                                    <td>${u.reports}</td>
                                    <td>${u.last_activity ? new Date(u.last_activity).toLocaleDateString() : 'N/A'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- All Evaluations -->
                <h3 class="st-subheader"><i class="bi bi-list-check"></i> All Evaluations (<span id="admin-evals-count">${filteredEvals.length}</span>)</h3>
                <div style="margin-bottom: 0.75rem;">
                    <label class="st-label" style="display: inline; margin-right: 0.5rem;">Filter:</label>
                    <select class="ds-select" id="admin-eval-filter" style="max-width: 200px; display: inline-block;">
                        <option value="all" ${this.statusFilter === 'all' ? 'selected' : ''}>All</option>
                        <option value="completed" ${this.statusFilter === 'completed' ? 'selected' : ''}>Completed</option>
                        <option value="failed" ${this.statusFilter === 'failed' ? 'selected' : ''}>Failed</option>
                        <option value="running" ${this.statusFilter === 'running' ? 'selected' : ''}>Running</option>
                        <option value="queued" ${this.statusFilter === 'queued' ? 'selected' : ''}>Queued</option>
                    </select>
                </div>
                <div class="ds-card ds-card--flush" style="max-height: 400px; overflow-y: auto;">
                    <table class="ds-table ds-table--compact" id="admin-evals-table">
                        <thead>
                            <tr><th>User</th><th>Evaluation</th><th>Status</th><th>Models</th><th>Created</th><th>Duration</th><th></th></tr>
                        </thead>
                        <tbody>
                            ${filteredEvals.map(e => {
                                const chipMap = {'completed':'ds-chip--success','failed':'ds-chip--error','running':'ds-chip--accent ds-chip--running','queued':'ds-chip--info'};
                                const chip = chipMap[e.status] || 'ds-chip--neutral';
                                const dur = e.duration ? (e.duration > 60 ? `${Math.round(e.duration/60)}m` : `${e.duration}s`) : '-';
                                return `
                                    <tr>
                                        <td><span class="ds-chip ds-chip--info">${e.user_id}</span></td>
                                        <td><strong>${e.eval_name}</strong></td>
                                        <td><span class="ds-chip ${chip}">${e.status}</span></td>
                                        <td>${e.models_count || 0}</td>
                                        <td>${e.created_at ? new Date(e.created_at).toLocaleString() : 'N/A'}</td>
                                        <td>${dur}</td>
                                        <td style="white-space: nowrap; display: flex; gap: 0.25rem;">
                                            <button class="ds-btn ds-btn--primary ds-btn--sm admin-clone-eval" data-user="${e.user_id}" data-eval="${e.eval_id}" title="Clone to my Setup">
                                                <i class="bi bi-copy"></i> Clone
                                            </button>
                                            ${e.results_s3_key ? `
                                                <button class="ds-btn ds-btn--info ds-btn--sm admin-download-results" data-user="${e.user_id}" data-eval="${e.eval_id}" title="Download Results CSV" style="background: #1a73e8; color: #fff; border: none;">
                                                    <i class="bi bi-download"></i> CSV
                                                </button>
                                            ` : ''}
                                            ${e.has_unprocessed ? `
                                                <button class="ds-btn ds-btn--sm admin-download-errors" data-user="${e.user_id}" data-eval="${e.eval_id}" title="Download Error Records" style="background: #e8710a; color: #fff; border: none;">
                                                    <i class="bi bi-download"></i> Errors
                                                </button>
                                            ` : ''}
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        this.bindEvents();
    },

    _updateEvalsTable() {
        const evals = this.data?.evaluations || [];
        const filtered = this.statusFilter === 'all' ? evals : evals.filter(e => e.status === this.statusFilter);
        const tbody = document.querySelector('#admin-evals-table tbody');
        const countLabel = document.getElementById('admin-evals-count');
        if (countLabel) countLabel.textContent = filtered.length;
        if (!tbody) return;

        tbody.innerHTML = filtered.map(e => {
            const chipMap = {'completed':'ds-chip--success','failed':'ds-chip--error','running':'ds-chip--accent ds-chip--running','queued':'ds-chip--info'};
            const chip = chipMap[e.status] || 'ds-chip--neutral';
            const dur = e.duration ? (e.duration > 60 ? `${Math.round(e.duration/60)}m` : `${e.duration}s`) : '-';
            return `
                <tr>
                    <td><span class="ds-chip ds-chip--info">${e.user_id}</span></td>
                    <td><strong>${e.eval_name}</strong></td>
                    <td><span class="ds-chip ${chip}">${e.status}</span></td>
                    <td>${e.models_count || 0}</td>
                    <td>${e.created_at ? new Date(e.created_at).toLocaleString() : 'N/A'}</td>
                    <td>${dur}</td>
                    <td style="white-space: nowrap; display: flex; gap: 0.25rem;">
                        <button class="ds-btn ds-btn--primary ds-btn--sm admin-clone-eval" data-user="${e.user_id}" data-eval="${e.eval_id}" title="Clone to my Setup">
                            <i class="bi bi-copy"></i> Clone
                        </button>
                        ${e.results_s3_key ? `
                            <button class="ds-btn ds-btn--sm admin-download-results" data-user="${e.user_id}" data-eval="${e.eval_id}" title="Download Results CSV" style="background: #1a73e8; color: #fff; border: none;">
                                <i class="bi bi-download"></i> CSV
                            </button>
                        ` : ''}
                        ${e.has_unprocessed ? `
                            <button class="ds-btn ds-btn--sm admin-download-errors" data-user="${e.user_id}" data-eval="${e.eval_id}" title="Download Error Records" style="background: #e8710a; color: #fff; border: none;">
                                <i class="bi bi-download"></i> Errors
                            </button>
                        ` : ''}
                    </td>
                </tr>
            `;
        }).join('');

        // Rebind action buttons
        this._bindActionButtons();
    },

    _bindActionButtons() {
        // Clone buttons
        document.querySelectorAll('.admin-clone-eval').forEach(btn => {
            btn.addEventListener('click', async () => {
                const userId = btn.dataset.user;
                const evalId = btn.dataset.eval;
                try {
                    App.showLoading(true);
                    const result = await API.post('/api/admin/clone-eval', { user_id: userId, eval_id: evalId });
                    if (result.success) {
                        State.updateConfig({
                            name: `Clone-${result.eval_name}`,
                            temp_path: result.temp_s3_key,
                            csv_file_name: result.csv_file_name || 'cloned_data.csv',
                            columns: result.columns || [],
                            preview: result.preview || [],
                            prompt_column: result.config?.prompt_column,
                            golden_answer_column: result.config?.golden_answer_column,
                            task_evaluations: result.config?.task_evaluations || [{task_type: result.config?.task_type || '', task_criteria: result.config?.task_criteria || '', temperature: result.config?.temperature || 0.7, user_defined_metrics: result.config?.user_defined_metrics || ''}],
                            selected_models: result.config?.selected_models || [],
                            judge_models: result.config?.judge_models || [],
                            vision_enabled: result.config?.vision_enabled || false,
                            latency_only_mode: result.config?.latency_only_mode || false,
                            stream_evaluation: result.config?.stream_evaluation !== undefined ? result.config.stream_evaluation : true,
                        });
                        App.showNotification('Success', `Cloned "${result.eval_name}" from ${userId}. Review in Setup tab.`);
                        App.switchTab('setup');
                        SetupComponent.renderEvaluationSetup();
                    } else {
                        App.showNotification('Error', result.error || 'Failed to clone evaluation', 'error');
                    }
                } catch (error) {
                    App.showNotification('Error', `Failed to clone: ${error.message}`, 'error');
                } finally {
                    App.showLoading(false);
                }
            });
        });

        // Download results buttons
        document.querySelectorAll('.admin-download-results').forEach(btn => {
            btn.addEventListener('click', async () => {
                const userId = btn.dataset.user;
                const evalId = btn.dataset.eval;
                try {
                    const result = await API.get(`/api/admin/download/${userId}/${evalId}/results`);
                    if (result.url) {
                        const a = document.createElement('a');
                        a.href = result.url;
                        a.download = result.filename || 'results.csv';
                        a.click();
                    } else {
                        App.showNotification('Error', result.error || 'No results available', 'error');
                    }
                } catch (error) {
                    App.showNotification('Error', `Failed to download: ${error.message}`, 'error');
                }
            });
        });

        // Download error records buttons
        document.querySelectorAll('.admin-download-errors').forEach(btn => {
            btn.addEventListener('click', async () => {
                const userId = btn.dataset.user;
                const evalId = btn.dataset.eval;
                try {
                    const result = await API.get(`/api/admin/download/${userId}/${evalId}/unprocessed`);
                    if (result.files?.length > 0) {
                        for (const file of result.files) {
                            const a = document.createElement('a');
                            a.href = file.url;
                            a.download = file.filename;
                            a.click();
                        }
                    } else {
                        App.showNotification('Error', result.error || 'No error files available', 'error');
                    }
                } catch (error) {
                    App.showNotification('Error', `Failed to download: ${error.message}`, 'error');
                }
            });
        });
    },

    bindEvents() {
        document.getElementById('admin-refresh-btn')?.addEventListener('click', async () => {
            await this.loadData();
            this.render();
            App.showNotification('Refreshed', 'Admin dashboard refreshed');
        });

        document.getElementById('admin-eval-filter')?.addEventListener('change', (e) => {
            this.statusFilter = e.target.value;
            this._updateEvalsTable();
        });

        this._bindActionButtons();
    }
};

window.AdminComponent = AdminComponent;
