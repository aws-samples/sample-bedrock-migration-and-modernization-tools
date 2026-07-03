/**
 * API utility functions for communicating with the Flask backend
 */

const API = {
    // In production, set this to the ALB URL (e.g., https://api.360-eval.amazon.dev)
    // In dev (same-origin), leave empty
    baseUrl: window.__API_BASE_URL__ || '',

    /**
     * Handle 401 responses — user is not authenticated
     */
    _handle401() {
        document.body.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#ccc;font-family:sans-serif;">
                <div style="text-align:center;">
                    <h1>Authentication Required</h1>
                    <p>Please connect via the corporate network or VPN to access 360-Eval.</p>
                </div>
            </div>`;
    },

    /**
     * Make a GET request
     */
    async get(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`);
            if (response.status === 401) { this._handle401(); return; }
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `HTTP error ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`GET ${endpoint} failed:`, error);
            throw error;
        }
    },

    /**
     * Make a POST request with JSON body
     */
    async post(endpoint, data) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data),
                credentials: 'include',
                redirect: 'follow'
            });
            if (response.status === 401) { this._handle401(); return; }
            const contentType = response.headers.get('content-type') || '';
            if (!contentType.includes('application/json')) {
                if (response.redirected) {
                    throw new Error('Session expired. Please refresh the page to re-authenticate.');
                }
                const text = await response.text();
                throw new Error(`Server error (HTTP ${response.status}): ${text.substring(0, 200)}`);
            }
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `HTTP error ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`POST ${endpoint} failed:`, error);
            throw error;
        }
    },

    /**
     * Make a DELETE request
     */
    async delete(endpoint) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'DELETE'
            });
            if (response.status === 401) { this._handle401(); return; }
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `HTTP error ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`DELETE ${endpoint} failed:`, error);
            throw error;
        }
    },

    /**
     * Upload a file
     */
    async uploadFile(endpoint, file) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            if (response.status === 401) { this._handle401(); return; }
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `HTTP error ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Upload to ${endpoint} failed:`, error);
            throw error;
        }
    },

    /**
     * Get raw HTML content
     */
    async getHtml(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return await response.text();
        } catch (error) {
            console.error(`GET HTML ${url} failed:`, error);
            throw error;
        }
    },

    // API endpoints

    /** Get application configuration */
    getConfig() {
        return this.get('/api/config');
    },

    /** Get available models */
    getModels() {
        return this.get('/api/models');
    },

    /** Get available judge models */
    getJudges() {
        return this.get('/api/judges');
    },

    /** Validate API credentials for selected models */
    validateCredentials(models) {
        return this.post('/api/validate-credentials', { models });
    },

    /** Get metric definitions and rubrics */
    getMetrics() {
        return this.get('/api/metrics');
    },

    /** Upload CSV file */
    uploadCsv(file) {
        return this.uploadFile('/api/upload-csv', file);
    },

    // Single-flight guard so the several components that all fetch the evaluations
    // list on boot (Monitor, Evaluations, Reports, debug-info) share one request
    // instead of issuing 4 identical ones. Cleared as soon as the request settles,
    // so there is no stale caching — the next call after it resolves hits the network.
    _evalsInFlight: null,

    /** Get all evaluations (concurrent callers share one in-flight request) */
    getEvaluations() {
        if (this._evalsInFlight) return this._evalsInFlight;
        this._evalsInFlight = this.get('/api/evaluations').finally(() => {
            this._evalsInFlight = null;
        });
        return this._evalsInFlight;
    },

    /** Create a new evaluation */
    createEvaluation(data) {
        return this.post('/api/evaluations', data);
    },

    /** Get a single evaluation */
    getEvaluation(evalId) {
        return this.get(`/api/evaluations/${evalId}`);
    },

    /** List APO (optimized-prompt) artifacts for an evaluation */
    getApoArtifacts(evalId) {
        return this.get(`/api/evaluations/${evalId}/apo`);
    },

    /** Direct download URL for a single APO artifact (served as attachment) */
    apoArtifactUrl(evalId, name) {
        return `${this.baseUrl}/api/evaluations/${evalId}/apo/${encodeURIComponent(name)}`;
    },

    /** Delete an evaluation */
    deleteEvaluation(evalId) {
        return this.delete(`/api/evaluations/${evalId}`);
    },

    /** Run evaluations */
    runEvaluations(evalIds) {
        return this.post('/api/evaluations/run', { evaluation_ids: evalIds });
    },

    /** Get queue status */
    getQueueStatus() {
        return this.get('/api/queue-status');
    },

    /** Validate models */
    validateModels(forceRefresh = false) {
        return this.post('/api/validate-models', { force_refresh: forceRefresh });
    },

    /** Get reports */
    getReports() {
        return this.get('/api/reports');
    },

    /** Generate a new report */
    generateReport(selectedEvaluations = null, selectedModelIds = null, summaryModel = null, summaryRegion = null, selectedSections = null) {
        return this.post('/api/reports/generate', {
            selected_evaluations: selectedEvaluations,
            selected_model_ids: selectedModelIds,
            summary_model: summaryModel,
            summary_region: summaryRegion,
            selected_sections: selectedSections,
        });
    },

    /** Get report HTML content */
    getReportContent(reportPath) {
        return this.getHtml(`/api/reports/${encodeURIComponent(reportPath)}`);
    },

    /** Delete a report */
    deleteReport(statusFile) {
        return this.post('/api/reports/delete', { status_file: statusFile });
    },

    /** Get unprocessed records */
    getUnprocessed() {
        return this.get('/api/unprocessed');
    },

    /** Get unprocessed records for a specific evaluation (on-demand) */
    getUnprocessedDetail(evalId) {
        return this.get(`/api/unprocessed/${evalId}`);
    },

    // --- User & Credentials ---

    /** Get authenticated user profile */
    getUserProfile() {
        return this.get('/api/user/profile');
    },

    /** Get saved credential providers (masked keys) */
    getCredentials() {
        return this.get('/api/credentials');
    },

    /** Save an API key for a provider */
    saveCredential(provider, apiKey) {
        return this.post('/api/credentials', { provider, api_key: apiKey });
    },

    /** Delete a saved credential */
    deleteCredential(provider) {
        return this.delete(`/api/credentials/${provider}`);
    }
};

// Export for use in other modules
window.API = API;
