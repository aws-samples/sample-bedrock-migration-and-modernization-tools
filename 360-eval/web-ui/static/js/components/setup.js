/**
 * Setup tab component
 * Handles evaluation setup, model configuration, and advanced settings
 * Styled to match Streamlit UI
 */

const SetupComponent = {
    modelsData: null,
    judgesData: null,
    metricsData: null,
    selectedRegion: 'us-east-1',

    /**
     * Initialize the setup component
     */
    async init() {
        await this.loadModelsAndJudges();
        this.render();
    },

    /**
     * Render all setup sub-tabs
     */
    render() {
        this.renderEvaluationSetup();
        this.renderModelConfig();
        this.renderAdvancedConfig();
    },

    /**
     * Load models and judges data
     */
    async loadModelsAndJudges() {
        try {
            const [modelsRes, judgesRes, metricsRes] = await Promise.all([
                API.getModels(),
                API.getJudges(),
                API.getMetrics(),
            ]);
            this.modelsData = modelsRes;
            this.judgesData = judgesRes;
            this.metricsData = metricsRes;
        } catch (error) {
            console.error('Failed to load models/judges:', error);
            App.showNotification('Error', 'Failed to load models and judges', 'error');
        }
    },

    /**
     * Render Evaluation Setup sub-tab (Streamlit-like layout)
     */
    renderEvaluationSetup() {
        const config = State.getConfig();
        const container = document.getElementById('evaluation-setup-content');

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <!-- Evaluation Name + CSV Upload — Side by Side -->
                <div class="st-columns-2 ds-mb-4" style="align-items: stretch; gap: 1.5rem;">
                    <div class="ds-card" style="display: flex; flex-direction: column;">
                        <label class="st-label">Evaluation Name</label>
                        <input type="text" class="ds-input ds-input--standard" id="eval-name"
                               value="${config.name || ''}"
                               placeholder="Enter evaluation name">
                        <p class="ds-helper" style="flex: 1;">Give your evaluation a descriptive name</p>
                    </div>
                    <div class="ds-card">
                        <label class="st-label">Upload Evaluation Dataset (CSV)</label>
                        <div class="st-file-uploader" id="csv-upload-zone" style="margin-top: 0.5rem;">
                            <i class="bi bi-cloud-upload"></i>
                            <p>Drag and drop your CSV file here, or click to select</p>
                            <input type="file" id="csv-file-input" accept=".csv" style="display: none;">
                        </div>
                        ${config.csv_file_name ? `
                            <div class="ds-alert ds-alert--success ds-mt-2">
                                <i class="bi bi-check-circle ds-alert__icon"></i>
                                <div class="ds-alert__content">
                                    <div class="ds-alert__message">File loaded: <strong>${config.csv_file_name}</strong> (${config.preview?.length || 0} rows preview)</div>
                                </div>
                            </div>
                            <button class="ds-btn ds-btn--danger ds-btn--sm ds-mt-2" id="clear-csv-btn">
                                <i class="bi bi-trash"></i> Clear File
                            </button>
                        ` : ''}
                    </div>
                </div>

                <!-- Evaluation Mode Toggle -->
                <div class="ds-card ds-mt-4">
                    <label class="st-label">Evaluation Mode</label>
                    <div class="st-radio-group" style="display: flex; gap: 1.5rem; margin-top: 0.5rem;">
                        <label class="st-radio">
                            <input type="radio" name="evaluation-mode" value="single_shot"
                                   ${config.evaluation_mode !== 'multi_shot' ? 'checked' : ''}>
                            <span><strong>Single-shot</strong> — N independent tasks against the same prompt/golden columns</span>
                        </label>
                        <label class="st-radio">
                            <input type="radio" name="evaluation-mode" value="multi_shot"
                                   ${config.evaluation_mode === 'multi_shot' ? 'checked' : ''}>
                            <span><strong>Multi-shot</strong> — N (prompt_K, golden_K) column pairs; each turn becomes its own standalone evaluation (<code>EVAL_1</code>, <code>EVAL_2</code>, …) judged independently in the Monitor</span>
                        </label>
                    </div>
                </div>

                ${(config.columns?.length > 0 && config.evaluation_mode !== 'multi_shot') ? `
                    <!-- Column Selection (single-shot only) -->
                    <div class="st-columns-2 ds-mt-4">
                        <div>
                            <label class="st-label">Prompt Column</label>
                            <select class="ds-select" id="prompt-column">
                                <option value="">Select column...</option>
                                ${config.columns.map(col => `
                                    <option value="${col}" ${config.prompt_column === col ? 'selected' : ''}>${col}</option>
                                `).join('')}
                            </select>
                        </div>
                        <div>
                            <label class="st-label">Golden Answer Column</label>
                            <select class="ds-select" id="golden-answer-column" ${config.golden_answer_mode === 'criteria_only' ? 'disabled' : ''}>
                                <option value="">Select column...</option>
                                ${config.columns.map(col => `
                                    <option value="${col}" ${config.golden_answer_column === col ? 'selected' : ''}>${col}</option>
                                `).join('')}
                            </select>
                        </div>
                    </div>
                ` : ''}

                <!-- Data Preview -->
                ${config.preview?.length > 0 ? `
                    <h3 class="st-subheader">Data Preview</h3>
                    <div class="ds-card ds-card--flush" style="max-height: 300px; overflow: auto;">
                        <table class="ds-table ds-table--compact">
                            <thead>
                                <tr>
                                    ${config.columns.map(col => `<th>${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${config.preview.slice(0, 5).map(row => `
                                    <tr>
                                        ${config.columns.map(col => `
                                            <td>${String(row[col] || '').substring(0, 100)}${String(row[col] || '').length > 100 ? '...' : ''}</td>
                                        `).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : ''}

                <!-- Golden Answer Mode -->
                ${config.columns?.length > 0 ? `
                    <div class="ds-card ds-mt-4 ds-mb-4" style="padding: 0.75rem 1rem;">
                        <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 0.5rem;">
                            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
                                <input type="radio" name="golden-answer-mode" value="golden_answer" ${config.golden_answer_mode !== 'criteria_only' ? 'checked' : ''}>
                                <span>Use golden answer</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
                                <input type="radio" name="golden-answer-mode" value="criteria_only" ${config.golden_answer_mode === 'criteria_only' ? 'checked' : ''}>
                                <span>Use success criteria</span>
                            </label>
                        </div>
                        <div id="success-criteria-form" style="display: ${config.golden_answer_mode === 'criteria_only' ? 'block' : 'none'};">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                                <div>
                                    <label class="st-label">Must Include *</label>
                                    <textarea class="ds-input ds-input--standard" id="criteria-must-include" rows="2"
                                        placeholder="Items the response must contain">${config.success_criteria?.must_include || ''}</textarea>
                                </div>
                                <div>
                                    <label class="st-label">Success Definition *</label>
                                    <textarea class="ds-input ds-input--standard" id="criteria-success-def" rows="2"
                                        placeholder="What a successful response looks like">${config.success_criteria?.success_definition || ''}</textarea>
                                </div>
                                <div>
                                    <label class="st-label">Must NOT Include</label>
                                    <textarea class="ds-input ds-input--standard" id="criteria-must-not-include" rows="2"
                                        placeholder="Items the response must NOT contain (optional)">${config.success_criteria?.must_not_include || ''}</textarea>
                                </div>
                                <div>
                                    <label class="st-label">Edge Cases</label>
                                    <textarea class="ds-input ds-input--standard" id="criteria-edge-cases" rows="2"
                                        placeholder="How edge cases or ambiguity should be handled (optional)">${config.success_criteria?.edge_cases || ''}</textarea>
                                </div>
                            </div>
                        </div>
                    </div>
                ` : ''}

                <!-- Configuration Options — Two Column Layout -->
                <div class="st-columns-2 ds-mt-4" style="align-items: start; gap: 1.5rem;">
                    <!-- Left Panel: Vision + Prompt Optimization -->
                    <div class="ds-card">
                        <h4 class="st-section-header" style="margin-top:0;">Vision Model Configuration</h4>
                        <label class="st-checkbox">
                            <input type="checkbox" id="vision-enabled" ${config.vision_enabled ? 'checked' : ''}>
                            <span>Enable Vision Model Evaluation</span>
                        </label>
                        <div id="vision-panel" style="${config.vision_enabled && config.columns?.length > 0 ? '' : 'display:none;'}">
                            <div class="ds-mt-2">
                                <label class="st-label">Image Column</label>
                                <select class="ds-select" id="image-column">
                                    <option value="">Select column containing image data...</option>
                                    ${(config.columns || []).map(col => `
                                        <option value="${col}" ${config.image_column === col ? 'selected' : ''}>${col}</option>
                                    `).join('')}
                                </select>
                            </div>
                        </div>

                        <hr class="ds-divider" style="margin: 1rem 0;">

                        <h4 class="st-section-header" style="margin-top:0;">Prompt Optimization (APO)</h4>
                        <label class="st-checkbox">
                            <input type="checkbox" id="prompt-opt-enabled" ${config.prompt_optimization_mode !== 'none' ? 'checked' : ''}>
                            <span>Enable Prompt Optimization</span>
                        </label>
                        <div id="prompt-opt-panel" style="${config.prompt_optimization_mode !== 'none' ? '' : 'display:none;'}">
                            <div class="ds-alert ds-alert--info ds-mt-2">
                                <i class="bi bi-info-circle ds-alert__icon"></i>
                                <div class="ds-alert__content">
                                    <div class="ds-alert__message">
                                        Runs an Amazon Bedrock APO job per selected model (20-50 min each).
                                        Extracts the shared system prompt from 5 sample rows, optimizes it against your
                                        chosen evaluator, then substitutes the optimized prompt per model in the eval.
                                        <strong>APO caps at 5 concurrent jobs</strong> — with N models selected, total
                                        APO time ≈ ceil(N/5) × ~30-50 min.
                                        Downloadable artifacts: per-model template + dataset CSV + summary log.
                                    </div>
                                </div>
                            </div>

                            <h5 class="st-section-header" style="margin-top: 1rem;">Apply Mode</h5>
                            <label class="st-checkbox">
                                <input type="radio" name="prompt-opt-mode" value="optimize_only"
                                       ${(config.prompt_optimization_mode || 'optimize_only') === 'optimize_only' ? 'checked' : ''}>
                                <span>Optimize prompts only (replace originals)</span>
                            </label>
                            <label class="st-checkbox">
                                <input type="radio" name="prompt-opt-mode" value="evaluate_both"
                                       ${config.prompt_optimization_mode === 'evaluate_both' ? 'checked' : ''}>
                                <span>Run both original and optimized side-by-side</span>
                            </label>

                            <h5 class="st-section-header" style="margin-top: 1rem;">Evaluator</h5>
                            <label class="st-checkbox">
                                <input type="radio" name="apo-evaluator" value="llmj"
                                       ${(config.apo_evaluator || 'llmj') === 'llmj' ? 'checked' : ''}>
                                <span><strong>LLM-as-Judge</strong> — provide a rubric + judge model</span>
                            </label>
                            <label class="st-checkbox">
                                <input type="radio" name="apo-evaluator" value="steering"
                                       ${config.apo_evaluator === 'steering' ? 'checked' : ''}>
                                <span><strong>Steering criteria</strong> — up to 5 natural-language rules</span>
                            </label>

                            <div id="apo-llmj-panel" class="ds-mt-3" style="${(config.apo_evaluator || 'llmj') === 'llmj' ? '' : 'display:none;'}">
                                <label class="st-label">Judge rubric</label>
                                <textarea class="ds-input ds-input--standard" id="apo-llmj-rubric" rows="4"
                                          placeholder="e.g., Score the response on accuracy and clarity. Return JSON: {score: 1-5, rationale: string}">${config.apo_llmj_rubric || ''}</textarea>
                                <label class="st-label ds-mt-2">Judge model</label>
                                <select class="ds-select" id="apo-llmj-judge-model">
                                    <option value="">Select a judge model...</option>
                                    ${(this.judgesData?.judges || []).map(([modelId, region]) => `
                                        <option value="${modelId}" ${config.apo_llmj_judge_model === modelId ? 'selected' : ''}>${modelId} (${region})</option>
                                    `).join('')}
                                </select>
                            </div>

                            <div id="apo-steering-panel" class="ds-mt-3" style="${config.apo_evaluator === 'steering' ? '' : 'display:none;'}">
                                <label class="st-label">Steering criteria (up to 5)</label>
                                ${[0, 1, 2, 3, 4].map(i => `
                                    <input type="text" class="ds-input ds-input--standard apo-steering-criterion ds-mt-1"
                                           data-index="${i}"
                                           placeholder="Criterion ${i + 1} (e.g., 'response must be ≤ 2 sentences')"
                                           value="${(config.apo_steering_criteria || [])[i] || ''}">
                                `).join('')}
                            </div>
                        </div>
                    </div>

                    <!-- Right Panel: Evaluation Type + Streaming -->
                    <div class="ds-card">
                        <h4 class="st-section-header" style="margin-top:0;">Evaluation Type</h4>
                        <label class="st-checkbox">
                            <input type="checkbox" id="latency-only" ${config.latency_only_mode ? 'checked' : ''}>
                            <span>Latency Only Mode (skip judge evaluation)</span>
                        </label>
                        <div id="latency-only-panel" style="${config.latency_only_mode ? '' : 'display:none;'}">
                            <div class="ds-alert ds-alert--info ds-mt-2">
                                <i class="bi bi-info-circle ds-alert__icon"></i>
                                <div class="ds-alert__content">
                                    <div class="ds-alert__message">In latency-only mode, only response time metrics will be collected. No judge model evaluation will be performed.</div>
                                </div>
                            </div>
                        </div>

                        <hr class="ds-divider" style="margin: 1rem 0;">

                        <h4 class="st-section-header" style="margin-top:0;">Streaming Mode</h4>
                        <label class="st-checkbox">
                            <input type="checkbox" id="streaming-mode" ${config.stream_evaluation ? 'checked' : ''}>
                            <span>Enable Streaming Mode</span>
                        </label>
                        <div id="streaming-panel" style="${config.stream_evaluation ? '' : 'display:none;'}">
                            <div class="ds-alert ds-alert--info ds-mt-2">
                                <i class="bi bi-info-circle ds-alert__icon"></i>
                                <div class="ds-alert__content">
                                    <div class="ds-alert__message">Streaming mode will capture time-to-first-token and token-by-token latency metrics.</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Task Evaluations / Multi-shot Turns -->
                <h3 class="st-subheader">${config.evaluation_mode === 'multi_shot' ? 'Chain Turns' : 'Task Evaluations'}</h3>
                <div class="ds-mb-4">
                    <label class="st-label">${config.evaluation_mode === 'multi_shot' ? 'Number of Turns' : 'Number of Tasks'}</label>
                    <input type="number" class="ds-input ds-input--standard" id="task-count"
                           min="1" max="10" value="${config.task_evaluations?.length || 1}" style="max-width: 150px;">
                </div>

                <div id="task-cards-container">
                    ${this._renderTaskCardsHtml(config)}
                </div>

                ${config.evaluation_mode === 'multi_shot' ? `
                    <div class="ds-alert ds-alert--info ds-mt-4">
                        <i class="bi bi-info-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">
                                Each turn is evaluated <strong>independently</strong> as its own evaluation run.
                                On save you'll see <code>${(config.name || 'EVAL') + '_1, ' + (config.name || 'EVAL') + '_2, …'}</code>
                                in Monitor — one entry per turn, scored standalone.
                            </div>
                        </div>
                    </div>
                ` : ''}

                <hr class="ds-divider">

                <!-- Save Configuration Button -->
                <div class="st-columns-2">
                    <div>
                        <button class="ds-btn ds-btn--primary ds-btn--lg" id="save-eval-config" style="width: 100%;">
                            <i class="bi bi-save"></i> Save Configuration
                        </button>
                    </div>
                    <div>
                        <button class="ds-btn ds-btn--secondary ds-btn--lg" id="reset-eval-config" style="width: 100%;">
                            <i class="bi bi-arrow-counterclockwise"></i> Reset Configuration
                        </button>
                    </div>
                </div>
            </div>
        `;

        this.bindEvaluationSetupEvents();
    },

    /**
     * Bind events for Evaluation Setup
     */
    bindEvaluationSetupEvents() {
        // Evaluation name
        document.getElementById('eval-name')?.addEventListener('change', (e) => {
            State.updateConfig({ name: e.target.value });
        });

        // CSV upload zone
        const uploadZone = document.getElementById('csv-upload-zone');
        const fileInput = document.getElementById('csv-file-input');

        if (uploadZone && fileInput) {
            uploadZone.addEventListener('click', () => fileInput.click());

            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });

            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });

            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) this.handleCsvUpload(file);
            });

            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) this.handleCsvUpload(file);
            });
        }

        // Column selectors
        document.getElementById('prompt-column')?.addEventListener('change', (e) => {
            State.updateConfig({ prompt_column: e.target.value });
        });

        document.getElementById('golden-answer-column')?.addEventListener('change', (e) => {
            State.updateConfig({ golden_answer_column: e.target.value });
        });

        // Golden answer mode radio
        document.querySelectorAll('input[name="golden-answer-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const mode = e.target.value;
                State.updateConfig({ golden_answer_mode: mode });
                const form = document.getElementById('success-criteria-form');
                const gaSelect = document.getElementById('golden-answer-column');
                if (form) form.style.display = mode === 'criteria_only' ? 'block' : 'none';
                if (gaSelect) gaSelect.disabled = mode === 'criteria_only';
            });
        });

        // Success criteria fields
        ['criteria-must-include', 'criteria-success-def', 'criteria-must-not-include', 'criteria-edge-cases'].forEach(id => {
            document.getElementById(id)?.addEventListener('input', () => {
                State.updateConfig({
                    success_criteria: {
                        must_include: document.getElementById('criteria-must-include')?.value || '',
                        success_definition: document.getElementById('criteria-success-def')?.value || '',
                        must_not_include: document.getElementById('criteria-must-not-include')?.value || '',
                        edge_cases: document.getElementById('criteria-edge-cases')?.value || '',
                    }
                });
            });
        });

        // Clear CSV file
        document.getElementById('clear-csv-btn')?.addEventListener('click', async () => {
            // Clear temp files on server
            try {
                await API.post('/api/clear-temp-files', {});
            } catch (e) {
                // Ignore — files may already be gone
            }
            // Reset file-related state
            State.updateConfig({
                csv_file_name: null,
                temp_path: null,
                columns: [],
                preview: [],
                prompt_column: null,
                golden_answer_column: null,
                image_column: null,
            });
            this.renderEvaluationSetup();
        });

        // Vision model — toggle panel
        document.getElementById('vision-enabled')?.addEventListener('change', (e) => {
            State.updateConfig({ vision_enabled: e.target.checked });
            const panel = document.getElementById('vision-panel');
            if (panel) panel.style.display = e.target.checked && State.getConfig().columns?.length > 0 ? '' : 'none';
        });

        document.getElementById('image-column')?.addEventListener('change', (e) => {
            State.updateConfig({ image_column: e.target.value });
        });

        // Prompt optimization — toggle panel
        // APO evaluator radio (LLMJ vs Steering)
        document.querySelectorAll('input[name="apo-evaluator"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                State.updateConfig({ apo_evaluator: e.target.value });
                const llmj = document.getElementById('apo-llmj-panel');
                const steer = document.getElementById('apo-steering-panel');
                if (llmj) llmj.style.display = e.target.value === 'llmj' ? '' : 'none';
                if (steer) steer.style.display = e.target.value === 'steering' ? '' : 'none';
            });
        });
        document.getElementById('apo-llmj-rubric')?.addEventListener('change', (e) => {
            State.updateConfig({ apo_llmj_rubric: e.target.value });
        });
        document.getElementById('apo-llmj-judge-model')?.addEventListener('change', (e) => {
            State.updateConfig({ apo_llmj_judge_model: e.target.value });
        });
        document.querySelectorAll('.apo-steering-criterion').forEach(input => {
            input.addEventListener('change', () => {
                // Re-collect all 5 inputs into a list (trimmed, empties preserved at positions)
                const criteria = Array.from(document.querySelectorAll('.apo-steering-criterion'))
                    .map(i => i.value.trim());
                State.updateConfig({ apo_steering_criteria: criteria });
            });
        });

        document.getElementById('prompt-opt-enabled')?.addEventListener('change', (e) => {
            const enabled = e.target.checked;
            const update = { prompt_optimization_mode: enabled ? 'optimize_only' : 'none' };
            // Persist the default evaluator. The evaluator radio shows 'llmj' selected by
            // default but a default-selected radio never fires a 'change' event, so without
            // this the value is never written to config and the engine can't tell which
            // evaluator to use.
            if (enabled && !(State.getConfig() || {}).apo_evaluator) {
                update.apo_evaluator = 'llmj';
            }
            State.updateConfig(update);
            const panel = document.getElementById('prompt-opt-panel');
            if (panel) panel.style.display = enabled ? '' : 'none';
        });

        document.querySelectorAll('input[name="prompt-opt-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                State.updateConfig({ prompt_optimization_mode: e.target.value });
            });
        });

        // Evaluation type — toggle panel
        document.getElementById('latency-only')?.addEventListener('change', (e) => {
            State.updateConfig({ latency_only_mode: e.target.checked });
            const panel = document.getElementById('latency-only-panel');
            if (panel) panel.style.display = e.target.checked ? '' : 'none';
        });

        // Streaming mode — toggle panel
        document.getElementById('streaming-mode')?.addEventListener('change', (e) => {
            State.updateConfig({ stream_evaluation: e.target.checked });
            const panel = document.getElementById('streaming-panel');
            if (panel) panel.style.display = e.target.checked ? '' : 'none';
        });

        // Task / Turn count — only update the cards container
        document.getElementById('task-count')?.addEventListener('change', (e) => {
            const count = parseInt(e.target.value) || 1;
            const tasks = State.getConfig().task_evaluations || [];
            while (tasks.length < count) {
                tasks.push({
                    task_type: '', task_criteria: '', temperature: 0.7,
                    user_defined_metrics: '', structured_output_format: null,
                    prompt_column: null, golden_answer_column: null,
                });
            }
            while (tasks.length > count) {
                tasks.pop();
            }
            State.updateConfig({ task_evaluations: tasks });
            this._renderTaskCards();
        });

        // Task card inputs
        document.querySelectorAll('.ds-card[data-index]').forEach(card => {
            const index = parseInt(card.dataset.index);

            card.querySelector('.task-type')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { task_type: e.target.value });
            });

            card.querySelector('.task-criteria')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { task_criteria: e.target.value });
            });

            card.querySelector('.task-temperature')?.addEventListener('input', (e) => {
                const temp = parseFloat(e.target.value);
                State.updateTaskEvaluation(index, { temperature: temp });
                const label = e.target.previousElementSibling;
                if (label) {
                    label.textContent = `Temperature: ${temp}`;
                }
            });

            card.querySelector('.task-metrics')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { user_defined_metrics: e.target.value });
            });

            card.querySelector('.task-structured-check')?.addEventListener('change', (e) => {
                const dropdown = card.querySelector('.task-structured-dropdown');
                if (e.target.checked) {
                    dropdown.style.display = '';
                    const format = card.querySelector('.task-structured-format')?.value || 'json';
                    State.updateTaskEvaluation(index, { structured_output_format: format });
                } else {
                    dropdown.style.display = 'none';
                    State.updateTaskEvaluation(index, { structured_output_format: null });
                }
            });

            card.querySelector('.task-structured-format')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { structured_output_format: e.target.value });
            });

            // Per-turn column selectors (multi-shot only — absent in single-shot)
            card.querySelector('.task-prompt-column')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { prompt_column: e.target.value });
            });
            card.querySelector('.task-golden-column')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { golden_answer_column: e.target.value });
            });

            card.querySelector('.remove-task-btn')?.addEventListener('click', () => {
                State.removeTaskEvaluation(index);
                // Update count input and re-render just the cards
                const countInput = document.getElementById('task-count');
                if (countInput) countInput.value = State.getConfig().task_evaluations?.length || 1;
                this._renderTaskCards();
            });
        });

        // Evaluation mode toggle (single_shot vs multi_shot)
        document.querySelectorAll('input[name="evaluation-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                State.updateConfig({ evaluation_mode: e.target.value });
                this.renderEvaluationSetup();
            });
        });

        // Reset button
        document.getElementById('reset-eval-config')?.addEventListener('click', () => {
            State.resetConfig();
            this.renderEvaluationSetup();
            this.renderModelConfig();
            App.showNotification('Reset', 'Evaluation configuration has been reset');
        });

        // Save button
        document.getElementById('save-eval-config')?.addEventListener('click', () => {
            this.saveConfiguration();
        });
    },

    /**
     * Handle CSV file upload
     */
    async handleCsvUpload(file) {
        try {
            App.showLoading(true);
            const result = await API.uploadCsv(file);

            State.updateConfig({
                csv_file_name: result.filename,
                columns: result.columns,
                preview: result.preview,
                temp_path: result.temp_path,
                prompt_column: null,
                golden_answer_column: null
            });

            this.renderEvaluationSetup();
            App.showNotification('Success', `Loaded ${result.row_count} rows from ${result.filename}`);
        } catch (error) {
            App.showNotification('Error', `Failed to upload CSV: ${error.message}`, 'error');
        } finally {
            App.showLoading(false);
        }
    },

    /**
     * Render the inner HTML for the task/turn cards container.
     * Used both during full render and during re-render via _renderTaskCards.
     */
    _renderTaskCardsHtml(config) {
        const isMultiShot = config.evaluation_mode === 'multi_shot';
        const titleLabel = isMultiShot ? 'Turn' : 'Task';
        const columns = config.columns || [];

        const tasks = config.task_evaluations || [{
            task_type: '', task_criteria: '', temperature: 0.7,
            user_defined_metrics: '', structured_output_format: null,
            prompt_column: null, golden_answer_column: null,
        }];

        return tasks.map((task, index) => `
            <div class="ds-card ds-card--accent ds-mb-3" data-index="${index}">
                <div class="ds-card__header">
                    <span class="ds-card__title ds-text-accent">${titleLabel} ${index + 1}</span>
                    ${index > 0 ? `
                        <button class="ds-btn ds-btn--danger ds-btn--sm remove-task-btn" data-index="${index}">
                            <i class="bi bi-trash"></i>
                        </button>
                    ` : ''}
                </div>
                <div class="ds-card__body">
                    ${isMultiShot && columns.length > 0 ? `
                        <div class="st-columns-2 ds-mb-3">
                            <div>
                                <label class="st-label">Prompt Column (turn ${index + 1})</label>
                                <select class="ds-select task-prompt-column" data-index="${index}">
                                    <option value="">Select column...</option>
                                    ${columns.map(col => `
                                        <option value="${col}" ${task.prompt_column === col ? 'selected' : ''}>${col}</option>
                                    `).join('')}
                                </select>
                            </div>
                            <div>
                                <label class="st-label">Golden Answer Column (turn ${index + 1})</label>
                                <select class="ds-select task-golden-column" data-index="${index}">
                                    <option value="">Select column...</option>
                                    ${columns.map(col => `
                                        <option value="${col}" ${task.golden_answer_column === col ? 'selected' : ''}>${col}</option>
                                    `).join('')}
                                </select>
                            </div>
                        </div>
                    ` : ''}
                    <div class="st-columns-2">
                        <div>
                            <label class="st-label">Task Type <i style="font-weight:normal;">(What is the Target Model Supposed to Do?)</i></label>
                            <input type="text" class="ds-input ds-input--standard task-type"
                                   value="${task.task_type || ''}"
                                   placeholder="e.g., summarization, classification">
                        </div>
                        <div>
                            <label class="st-label">Temperature: ${task.temperature || 0.7}</label>
                            <input type="range" class="form-range task-temperature"
                                   min="0.01" max="1" step="0.01" value="${task.temperature || 0.7}"
                                   style="margin-top: 0.5rem;">
                        </div>
                    </div>
                    <div class="ds-mt-3">
                        <label class="st-label">Task Criteria <i style="font-weight:normal;">(Describe what defines success and describe User-Defined Metrics if applicable)</i></label>
                        <textarea class="ds-input ds-input--standard task-criteria" rows="3"
                                  placeholder="Describe what makes a good response...">${task.task_criteria || ''}</textarea>
                    </div>
                    <div class="ds-mt-3">
                        <label class="st-label">User-Defined Metrics (optional)</label>
                        <input type="text" class="ds-input ds-input--standard task-metrics"
                               value="${task.user_defined_metrics || ''}"
                               placeholder="e.g., accuracy, relevance, coherence">
                    </div>
                    <div class="ds-mt-3">
                        <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                            <input type="checkbox" class="task-structured-check" data-index="${index}"
                                   ${task.structured_output_format ? 'checked' : ''}>
                            <span class="st-label" style="margin: 0;">Data Structured Analysis</span>
                        </label>
                        <div class="task-structured-dropdown ds-mt-2" style="${task.structured_output_format ? '' : 'display:none;'}">
                            <label class="st-label">Expected Output Format</label>
                            <select class="ds-select task-structured-format" data-index="${index}" style="max-width: 250px;">
                                <option value="json" ${task.structured_output_format === 'json' ? 'selected' : ''}>JSON</option>
                                <option value="csv-comma" ${task.structured_output_format === 'csv-comma' ? 'selected' : ''}>CSV (Comma)</option>
                                <option value="csv-pipe" ${task.structured_output_format === 'csv-pipe' ? 'selected' : ''}>CSV (Pipe)</option>
                                <option value="markdown" ${task.structured_output_format === 'markdown' ? 'selected' : ''}>Markdown</option>
                                <option value="yaml" ${task.structured_output_format === 'yaml' ? 'selected' : ''}>YAML</option>
                                <option value="html" ${task.structured_output_format === 'html' ? 'selected' : ''}>HTML</option>
                                <option value="xml" ${task.structured_output_format === 'xml' ? 'selected' : ''}>XML</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    },

    /**
     * Re-render only the task cards container and rebind their events
     */
    _renderTaskCards() {
        const config = State.getConfig();
        const container = document.getElementById('task-cards-container');
        if (!container) return;
        container.innerHTML = this._renderTaskCardsHtml(config);
        this._bindTaskCardEvents();
    },

    /**
     * Bind events for task card inputs
     */
    _bindTaskCardEvents() {
        document.querySelectorAll('.ds-card[data-index]').forEach(card => {
            const index = parseInt(card.dataset.index);

            card.querySelector('.task-type')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { task_type: e.target.value });
            });

            card.querySelector('.task-criteria')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { task_criteria: e.target.value });
            });

            card.querySelector('.task-temperature')?.addEventListener('input', (e) => {
                const temp = parseFloat(e.target.value);
                State.updateTaskEvaluation(index, { temperature: temp });
                const label = e.target.previousElementSibling;
                if (label) {
                    label.textContent = `Temperature: ${temp}`;
                }
            });

            card.querySelector('.task-metrics')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { user_defined_metrics: e.target.value });
            });

            card.querySelector('.task-structured-check')?.addEventListener('change', (e) => {
                const dropdown = card.querySelector('.task-structured-dropdown');
                if (e.target.checked) {
                    dropdown.style.display = '';
                    const format = card.querySelector('.task-structured-format')?.value || 'json';
                    State.updateTaskEvaluation(index, { structured_output_format: format });
                } else {
                    dropdown.style.display = 'none';
                    State.updateTaskEvaluation(index, { structured_output_format: null });
                }
            });

            card.querySelector('.task-structured-format')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { structured_output_format: e.target.value });
            });

            // Per-turn column selectors (multi-shot only — absent in single-shot)
            card.querySelector('.task-prompt-column')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { prompt_column: e.target.value });
            });
            card.querySelector('.task-golden-column')?.addEventListener('change', (e) => {
                State.updateTaskEvaluation(index, { golden_answer_column: e.target.value });
            });

            card.querySelector('.remove-task-btn')?.addEventListener('click', () => {
                State.removeTaskEvaluation(index);
                const countInput = document.getElementById('task-count');
                if (countInput) countInput.value = State.getConfig().task_evaluations?.length || 1;
                this._renderTaskCards();
            });
        });
    },

    /**
     * Render Model Configuration sub-tab (Streamlit-like layout)
     */
    renderModelConfig() {
        const config = State.getConfig();
        const container = document.getElementById('model-config-content');

        const regions = State.get('config')?.aws_regions || ['us-east-1', 'us-west-2', 'eu-west-1'];
        const bedrockModels = this.getModelsForRegion(this.selectedRegion);
        const otherModels = this.modelsData?.openai_models || [];
        const judges = this.getJudgesForRegion(this.selectedRegion);

        // Validation status
        const hasModels = config.selected_models?.length > 0;
        const hasJudges = config.judge_models?.length > 0 || config.latency_only_mode;
        const isValid = hasModels && hasJudges;

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <!-- Validation Banner -->
                ${!isValid ? `
                    <div class="ds-alert ds-alert--warning ds-mb-4">
                        <i class="bi bi-exclamation-triangle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__title">Configuration Incomplete</div>
                            <div class="ds-alert__message">
                                ${!hasModels ? '• Select at least one model to evaluate<br>' : ''}
                                ${!hasJudges ? '• Select at least one judge model (or enable latency-only mode)' : ''}
                            </div>
                        </div>
                    </div>
                ` : `
                    <div class="ds-alert ds-alert--success ds-mb-4">
                        <i class="bi bi-check-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">Configuration valid - ready to save</div>
                        </div>
                    </div>
                `}

                <!-- Region Selection -->
                <div id="region-selector-panel">
                    <label class="st-label ds-mt-4">AWS Region</label>
                    <select class="ds-select" id="aws-region" style="max-width: 300px;">
                        ${regions.map(r => `
                            <option value="${r}" ${this.selectedRegion === r ? 'selected' : ''}>${r}</option>
                        `).join('')}
                    </select>
                </div>

                <!-- Model Selection Tabs -->
                <h3 class="st-subheader">Model Selection</h3>
                <div class="st-tabs" id="model-tabs">
                    <button class="st-tab active" data-model-tab="bedrock">Bedrock</button>
                    <button class="st-tab" data-model-tab="other">Other</button>
                </div>

                <!-- Bedrock Models Tab -->
                <div class="st-tab-content active" id="model-tab-bedrock">
                    ${this.renderModelSelector(bedrockModels, 'bedrock')}
                </div>

                <!-- Other Models Tab -->
                <div class="st-tab-content" id="model-tab-other">
                    <div class="ds-card ds-mb-3">
                        <div class="st-columns-5" style="margin-bottom: 1rem; align-items: end;">
                            <div>
                                <label class="st-label">Provider</label>
                                <select class="ds-select" id="other-provider-select">
                                    <option value="openai/">OpenAI</option>
                                    <option value="gemini/">Google Gemini</option>
                                    <option value="azure/">Azure OpenAI</option>
                                    <option value="anthropic/">Anthropic</option>
                                    <option value="custom">Custom...</option>
                                </select>
                            </div>
                            <div>
                                <label class="st-label">Model ID</label>
                                <input type="text" class="ds-input ds-input--standard" id="other-model-id"
                                       placeholder="e.g., gpt-4o, gemini-2.0-flash">
                            </div>
                            <div>
                                <label class="st-label">Input Cost (per 1M)</label>
                                <input type="number" class="ds-input ds-input--standard" id="other-input-cost" step="0.01" value="0">
                            </div>
                            <div>
                                <label class="st-label">Output Cost (per 1M)</label>
                                <input type="number" class="ds-input ds-input--standard" id="other-output-cost" step="0.01" value="0">
                            </div>
                            <div>
                                <label class="st-label">Region</label>
                                <input type="text" class="ds-input ds-input--standard" id="other-region" value="N/A"
                                       placeholder="N/A or region">
                            </div>
                        </div>
                        <div id="other-custom-provider" style="display:none; margin-bottom: 1rem;">
                            <label class="st-label">Custom Provider Prefix</label>
                            <input type="text" class="ds-input ds-input--standard" id="other-custom-prefix"
                                   placeholder="e.g., together_ai/" style="max-width: 250px;">
                        </div>
                        <button class="ds-btn ds-btn--primary" id="other-add-model">
                            <i class="bi bi-plus"></i> Add Model
                        </button>
                    </div>
                </div>

                <!-- Selected Models -->
                <div id="selected-models-container">
                <h3 class="st-subheader">Selected Models (${config.selected_models?.length || 0})</h3>
                ${config.selected_models?.length > 0 ? `
                    <div class="ds-card ds-card--flush" style="max-height: 250px; overflow-y: auto;">
                        <table class="ds-table ds-table--compact">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Region</th>
                                    <th>Input Cost</th>
                                    <th>Output Cost</th>
                                    <th>Service Tier</th>
                                    <th>RPM</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                ${config.selected_models.map((model, index) => `
                                    <tr>
                                        <td><strong>${this.extractModelName(model.id)}</strong></td>
                                        <td><span class="ds-chip ds-chip--info">${model.region}</span></td>
                                        <td>$${model.input_cost}</td>
                                        <td>$${model.output_cost}</td>
                                        <td><span class="ds-chip ds-chip--neutral">${model.service_tier || 'default'}</span></td>
                                        <td>${model.target_rpm || '-'}</td>
                                        <td>
                                            <button class="ds-btn ds-btn--danger ds-btn--sm ds-btn--icon remove-model" data-index="${index}">
                                                <i class="bi bi-x"></i>
                                            </button>
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <button class="ds-btn ds-btn--secondary ds-mt-2" id="clear-models">
                        Clear All Models
                    </button>
                    ${(() => {
                        const allBedrockItems = [
                            ...(config.selected_models || []).filter(m => m.id && m.id.includes('bedrock/')),
                            ...(config.judge_models || []).filter(m => m.id && m.id.includes('bedrock/'))
                        ];
                        const bedrockRegions = new Set(allBedrockItems.map(m => m.region));
                        if (bedrockRegions.size > 1) {
                            return `
                                <div class="ds-alert ds-alert--warning ds-mt-3">
                                    <i class="bi bi-exclamation-triangle ds-alert__icon"></i>
                                    <div class="ds-alert__content">
                                        <div class="ds-alert__title">Multi-Region Evaluation</div>
                                        <div class="ds-alert__message">
                                            You have selected Bedrock models and/or judges across <strong>${bedrockRegions.size} regions</strong> (${[...bedrockRegions].join(', ')}).
                                            <br><br>
                                            Short-term (temporary) Bedrock API keys only allow inference in the region where they were generated.
                                            To evaluate across multiple regions, you must use a <strong>long-term API key</strong>.
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                        return '';
                    })()}
                ` : `
                    <div class="ds-alert ds-alert--info">
                        <i class="bi bi-info-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">No models selected. Add models above to begin evaluation.</div>
                        </div>
                    </div>
                `}
                </div>

                <!-- Judge Models -->
                <h3 class="st-subheader">Judge Models</h3>
                ${config.latency_only_mode ? `
                    <div class="ds-alert ds-alert--info">
                        <i class="bi bi-info-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">Judge models are not required in latency-only mode.</div>
                        </div>
                    </div>
                ` : `
                    <!-- Evaluation Mode Toggle -->
                    <div class="ds-card ds-mb-4">
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            <label class="st-label" style="margin: 0;">Evaluation Mode:</label>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="ds-btn ds-btn--sm ${config.eval_mode !== 'specialist' ? 'ds-btn--primary' : 'ds-btn--secondary'}" id="mode-bundled">
                                    Bundled
                                </button>
                                <button class="ds-btn ds-btn--sm ${config.eval_mode === 'specialist' ? 'ds-btn--primary' : 'ds-btn--secondary'}" id="mode-specialist">
                                    Specialist
                                </button>
                            </div>
                            <span style="color: var(--text-secondary); font-size: 0.8rem;">
                                ${config.eval_mode === 'specialist'
                                    ? 'One model per metric — more precise, parallel calls'
                                    : 'All metrics scored by each judge in a single call'}
                            </span>
                        </div>
                    </div>

                    <!-- Bundled Mode: flat judge list -->
                    <div id="bundled-judge-section" style="display: ${config.eval_mode === 'specialist' ? 'none' : 'block'};">
                        <div class="ds-card ds-mb-4">
                            <div class="st-columns-4" style="margin-bottom: 1rem;">
                                <div style="grid-column: span 2;">
                                    <label class="st-label">Select Judge Model</label>
                                    <select class="ds-select" id="judge-select">
                                        <option value="">Choose a judge model...</option>
                                        ${judges.map(([modelId, region]) => `
                                            <option value="${modelId}" data-region="${region || 'N/A'}">
                                                ${this.extractModelName(modelId)}
                                            </option>
                                        `).join('')}
                                    </select>
                                </div>
                                <div>
                                    <label class="st-label">Input Cost (per 1M)</label>
                                    <input type="number" class="ds-input ds-input--standard" id="judge-input-cost" step="0.0001" value="0">
                                </div>
                                <div>
                                    <label class="st-label">Output Cost (per 1M)</label>
                                    <input type="number" class="ds-input ds-input--standard" id="judge-output-cost" step="0.0001" value="0">
                                </div>
                            </div>
                            <button class="ds-btn ds-btn--primary" id="add-judge">
                                <i class="bi bi-plus"></i> Add Judge
                            </button>
                        </div>

                        ${config.judge_models?.length > 0 ? `
                            <div class="ds-card ds-card--flush" style="max-height: 200px; overflow-y: auto;">
                                <table class="ds-table ds-table--compact">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Region</th>
                                            <th>Input Cost</th>
                                            <th>Output Cost</th>
                                            <th></th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${config.judge_models.map((judge, index) => `
                                            <tr>
                                                <td><strong>${this.extractModelName(judge.id)}</strong></td>
                                                <td><span class="ds-chip ds-chip--info">${judge.region}</span></td>
                                                <td>$${judge.input_cost}</td>
                                                <td>$${judge.output_cost}</td>
                                                <td>
                                                    <button class="ds-btn ds-btn--danger ds-btn--sm ds-btn--icon remove-judge" data-index="${index}">
                                                        <i class="bi bi-x"></i>
                                                    </button>
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                            <button class="ds-btn ds-btn--secondary ds-mt-2" id="clear-judges">
                                Clear All Judges
                            </button>
                        ` : ''}
                    </div>

                    <!-- Specialist Mode: metric-to-model assignment table -->
                    <div id="specialist-judge-section" style="display: ${config.eval_mode === 'specialist' ? 'block' : 'none'};">
                        <div class="ds-card ds-mb-4">
                            <table class="ds-table ds-table--compact" id="metric-assignment-table">
                                <thead>
                                    <tr>
                                        <th style="width: 180px;">Metric</th>
                                        <th>Primary Model</th>
                                        <th>Secondary (optional)</th>
                                        <th style="width: 70px;">Threshold</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${this._renderMetricAssignmentRows(config, judges)}
                                </tbody>
                            </table>
                        </div>

                        <!-- Custom Metric Builder -->
                        <div class="ds-card ds-mb-4" id="custom-metric-section">
                            <h4 class="st-section-header" style="margin-top: 0;">Custom Metrics</h4>
                            ${(config.custom_metrics || []).map((cm, idx) => `
                                <div class="ds-card ds-mb-2" style="padding: 0.5rem 0.75rem; display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>${cm.metric_name}</strong>
                                        <span style="color: var(--text-secondary); font-size: 0.8rem; margin-left: 0.5rem;">${cm.definition?.substring(0, 60)}${cm.definition?.length > 60 ? '...' : ''}</span>
                                    </div>
                                    <button class="ds-btn ds-btn--danger ds-btn--sm ds-btn--icon remove-custom-metric" data-index="${idx}">
                                        <i class="bi bi-x"></i>
                                    </button>
                                </div>
                            `).join('')}
                            <button class="ds-btn ds-btn--secondary ds-btn--sm ds-mt-2" id="add-custom-metric-btn">
                                <i class="bi bi-plus"></i> Add Custom Metric
                            </button>
                        </div>
                    </div>
                `}

                <!-- Validation Status -->
                ${!isValid ? `
                    <div class="ds-alert ds-alert--warning ds-mt-4">
                        <i class="bi bi-exclamation-triangle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">Please complete the configuration above before saving.</div>
                        </div>
                    </div>
                ` : ''}

                <hr class="ds-divider">

                <!-- Save/Reset Buttons -->
                <div class="st-columns-2">
                    <div>
                        <button class="ds-btn ds-btn--primary ds-btn--lg" id="save-model-config" style="width: 100%;">
                            <i class="bi bi-save"></i> Save Configuration
                        </button>
                    </div>
                    <div>
                        <button class="ds-btn ds-btn--secondary ds-btn--lg" id="reset-model-config" style="width: 100%;">
                            <i class="bi bi-arrow-counterclockwise"></i> Reset Configuration
                        </button>
                    </div>
                </div>
            </div>
        `;

        this.bindModelConfigEvents();
    },

    /**
     * Render model selector form (Streamlit-like 5-column layout)
     */
    renderModelSelector(models, type) {
        // Get initial costs and tiers for the first model in the list
        const firstModel = models.length > 0 ? models[0][0] : '';
        const region = this.selectedRegion;
        const costKey = `${firstModel}|${region}`;
        const fallbackKey = `${firstModel}|N/A`;
        const initialCosts = this.modelsData?.cost_map?.[costKey] || this.modelsData?.cost_map?.[fallbackKey] || {};
        const initialTiers = this.modelsData?.service_tiers?.[costKey] || ['default'];

        return `
            <div class="ds-card ds-mb-3">
                <div class="st-columns-5" style="margin-bottom: 1rem; align-items: end;">
                    <div>
                        <label class="st-label">Model</label>
                        <select class="ds-select" id="${type}-model-select">
                            <option value="">Choose a model...</option>
                            ${models.map(([modelId, region]) => `
                                <option value="${modelId}" data-region="${region || 'N/A'}">
                                    ${this.extractModelName(modelId)}
                                </option>
                            `).join('')}
                        </select>
                    </div>
                    <div>
                        <label class="st-label">Input Cost</label>
                        <input type="number" class="ds-input ds-input--standard" id="${type}-input-cost" step="0.0001" value="${initialCosts.input || 0}">
                    </div>
                    <div>
                        <label class="st-label">Output Cost</label>
                        <input type="number" class="ds-input ds-input--standard" id="${type}-output-cost" step="0.0001" value="${initialCosts.output || 0}">
                    </div>
                    <div>
                        <label class="st-label">Service Tier</label>
                        <select class="ds-select" id="${type}-service-tier">
                            ${initialTiers.map(t => `<option value="${t}">${t.charAt(0).toUpperCase() + t.slice(1)}</option>`).join('')}
                        </select>
                    </div>
                    <div>
                        <label class="st-label">Target RPM</label>
                        <input type="number" class="ds-input ds-input--standard" id="${type}-target-rpm" placeholder="Optional">
                    </div>
                </div>
                <button class="ds-btn ds-btn--primary" id="${type}-add-model">
                    <i class="bi bi-plus"></i> Add Model
                </button>
            </div>
        `;
    },

    /**
     * Bind events for Model Configuration
     */
    bindModelConfigEvents() {
        // Region change — update model and judge dropdowns
        document.getElementById('aws-region')?.addEventListener('change', (e) => {
            this.selectedRegion = e.target.value;
            const bedrockModels = this.getModelsForRegion(this.selectedRegion);
            const bedrockSelect = document.getElementById('bedrock-model-select');
            if (bedrockSelect) {
                bedrockSelect.innerHTML = '<option value="">Choose a model...</option>' +
                    bedrockModels.map(([modelId, region]) =>
                        `<option value="${modelId}" data-region="${region || 'N/A'}">${this.extractModelName(modelId)}</option>`
                    ).join('');
            }
            // Also update judge dropdown
            const judgeModels = this.getJudgesForRegion(this.selectedRegion);
            const judgeSelect = document.getElementById('judge-select');
            if (judgeSelect) {
                judgeSelect.innerHTML = '<option value="">Choose a judge model...</option>' +
                    judgeModels.map(([modelId, region]) =>
                        `<option value="${modelId}" data-region="${region || 'N/A'}">${this.extractModelName(modelId)}</option>`
                    ).join('');
                // Reset judge costs
                const judgeInputCost = document.getElementById('judge-input-cost');
                const judgeOutputCost = document.getElementById('judge-output-cost');
                if (judgeInputCost) judgeInputCost.value = 0;
                if (judgeOutputCost) judgeOutputCost.value = 0;
            }
        });

        // Model tab switching
        document.querySelectorAll('#model-tabs .st-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.dataset.modelTab;

                // Update tabs
                document.querySelectorAll('#model-tabs .st-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update content
                document.getElementById('model-tab-bedrock')?.classList.toggle('active', tabId === 'bedrock');
                document.getElementById('model-tab-other')?.classList.toggle('active', tabId === 'other');

                // Hide region selector on "Other" tab (non-Bedrock models don't have regions)
                const regionPanel = document.getElementById('region-selector-panel');
                if (regionPanel) regionPanel.style.display = tabId === 'other' ? 'none' : '';
            });
        });

        // Bedrock model selection handler
        const bedrockSelect = document.getElementById('bedrock-model-select');
        bedrockSelect?.addEventListener('change', (e) => {
            const modelId = e.target.value;
            const region = this.selectedRegion;
            const costKey = `${modelId}|${region}`;
            const fallbackKey = `${modelId}|N/A`;

            // Update service tier dropdown
            const tiers = this.modelsData?.service_tiers?.[costKey] || ['default'];
            const tierSelect = document.getElementById('bedrock-service-tier');
            if (tierSelect) {
                tierSelect.innerHTML = tiers.map(t =>
                    `<option value="${t}">${t.charAt(0).toUpperCase() + t.slice(1)}</option>`
                ).join('');
            }

            // Update costs using tier_pricing (default tier)
            this._updateBedrockCosts(modelId, region, 'default');
        });

        // Service tier change — update costs from tier_pricing
        document.getElementById('bedrock-service-tier')?.addEventListener('change', (e) => {
            const modelId = document.getElementById('bedrock-model-select')?.value;
            if (modelId) {
                this._updateBedrockCosts(modelId, this.selectedRegion, e.target.value);
            }
        });

        document.getElementById('bedrock-add-model')?.addEventListener('click', () => {
            this.addModel('bedrock');
        });

        // Other provider — show/hide custom prefix input
        document.getElementById('other-provider-select')?.addEventListener('change', (e) => {
            const customDiv = document.getElementById('other-custom-provider');
            if (customDiv) {
                customDiv.style.display = e.target.value === 'custom' ? '' : 'none';
            }
        });

        // Other provider — add model
        document.getElementById('other-add-model')?.addEventListener('click', () => {
            this.addOtherModel();
        });

        // Judge model selection — populate costs
        document.getElementById('judge-select')?.addEventListener('change', (e) => {
            const modelId = e.target.value;
            const region = this.selectedRegion;
            const costKey = `${modelId}|${region}`;
            const fallbackKey = `${modelId}|N/A`;
            const costs = this.judgesData?.cost_map?.[costKey] || this.judgesData?.cost_map?.[fallbackKey];
            if (costs) {
                document.getElementById('judge-input-cost').value = costs.input || 0;
                document.getElementById('judge-output-cost').value = costs.output || 0;
            }
        });

        // Add judge handler
        document.getElementById('add-judge')?.addEventListener('click', () => {
            this.addJudge();
        });

        // Clear buttons — targeted DOM update instead of full re-render
        document.getElementById('clear-models')?.addEventListener('click', () => {
            State.clearSelectedModels();
            document.querySelectorAll('.remove-model').forEach(btn => btn.closest('tr')?.remove());
            const clearBtn = document.getElementById('clear-models');
            if (clearBtn) clearBtn.closest('.ds-card')?.previousElementSibling?.remove();
            clearBtn?.parentElement?.remove();
        });

        document.getElementById('clear-judges')?.addEventListener('click', () => {
            State.clearJudgeModels();
            document.querySelectorAll('.remove-judge').forEach(btn => btn.closest('tr')?.remove());
            const clearBtn = document.getElementById('clear-judges');
            if (clearBtn) clearBtn.closest('.ds-card')?.previousElementSibling?.remove();
            clearBtn?.parentElement?.remove();
        });

        // Evaluation mode toggle
        document.getElementById('mode-bundled')?.addEventListener('click', () => {
            State.setEvalMode('bundled');
            document.getElementById('bundled-judge-section').style.display = 'block';
            document.getElementById('specialist-judge-section').style.display = 'none';
            document.getElementById('mode-bundled').className = 'ds-btn ds-btn--sm ds-btn--primary';
            document.getElementById('mode-specialist').className = 'ds-btn ds-btn--sm ds-btn--secondary';
        });
        document.getElementById('mode-specialist')?.addEventListener('click', () => {
            State.setEvalMode('specialist');
            document.getElementById('bundled-judge-section').style.display = 'none';
            document.getElementById('specialist-judge-section').style.display = 'block';
            document.getElementById('mode-bundled').className = 'ds-btn ds-btn--sm ds-btn--secondary';
            document.getElementById('mode-specialist').className = 'ds-btn ds-btn--sm ds-btn--primary';
        });

        // Specialist metric assignment changes
        document.querySelectorAll('.metric-primary-select').forEach(select => {
            select.addEventListener('change', (e) => {
                const metric = e.target.dataset.metric;
                const modelId = e.target.value;
                if (!modelId) {
                    State.updateMetricAssignment(metric, { primary: null });
                    return;
                }
                const region = this.selectedRegion;
                const costKey = `${modelId}|${region}`;
                const fallbackKey = `${modelId}|N/A`;
                const costs = this.judgesData?.cost_map?.[costKey] || this.judgesData?.cost_map?.[fallbackKey] || {};
                State.updateMetricAssignment(metric, {
                    primary: { id: modelId, region, input_cost: costs.input || 0, output_cost: costs.output || 0 }
                });
            });
        });
        document.querySelectorAll('.metric-secondary-select').forEach(select => {
            select.addEventListener('change', (e) => {
                const metric = e.target.dataset.metric;
                const modelId = e.target.value;
                if (!modelId) {
                    State.updateMetricAssignment(metric, { secondary: null });
                    return;
                }
                const region = this.selectedRegion;
                const costKey = `${modelId}|${region}`;
                const fallbackKey = `${modelId}|N/A`;
                const costs = this.judgesData?.cost_map?.[costKey] || this.judgesData?.cost_map?.[fallbackKey] || {};
                State.updateMetricAssignment(metric, {
                    secondary: { id: modelId, region, input_cost: costs.input || 0, output_cost: costs.output || 0 }
                });
            });
        });
        document.querySelectorAll('.metric-threshold').forEach(input => {
            input.addEventListener('change', (e) => {
                const metric = e.target.dataset.metric;
                State.updateMetricAssignment(metric, { threshold: parseInt(e.target.value) || 3 });
            });
        });

        // Custom metric buttons
        document.getElementById('add-custom-metric-btn')?.addEventListener('click', () => {
            this._showCustomMetricForm();
        });
        document.querySelectorAll('.remove-custom-metric').forEach(btn => {
            btn.addEventListener('click', () => {
                State.removeCustomMetric(parseInt(btn.dataset.index));
                // Remove the metric card and its row in the assignment table
                btn.closest('.ds-card')?.remove();
                const metricName = btn.closest('.ds-card')?.querySelector('strong')?.textContent;
                if (metricName) {
                    document.querySelector(`.metric-primary-select[data-metric="${metricName}"]`)?.closest('tr')?.remove();
                }
            });
        });

        // Remove model/judge buttons — targeted row removal
        document.querySelectorAll('.remove-model').forEach(btn => {
            btn.addEventListener('click', () => {
                State.removeSelectedModel(parseInt(btn.dataset.index));
                btn.closest('tr')?.remove();
            });
        });

        document.querySelectorAll('.remove-judge').forEach(btn => {
            btn.addEventListener('click', () => {
                State.removeJudgeModel(parseInt(btn.dataset.index));
                btn.closest('tr')?.remove();
            });
        });

        // Save button
        document.getElementById('save-model-config')?.addEventListener('click', () => {
            this.saveConfiguration();
        });

        // Reset button
        document.getElementById('reset-model-config')?.addEventListener('click', () => {
            State.clearSelectedModels();
            State.clearJudgeModels();
            this.renderModelConfig();
            App.showNotification('Reset', 'Model configuration has been reset');
        });
    },

    /**
     * Add a model to selection
     */
    _updateBedrockCosts(modelId, region, tier) {
        const costKey = `${modelId}|${region}`;
        const fallbackKey = `${modelId}|N/A`;

        // Try tier_pricing first for the selected tier
        const tierPricing = this.modelsData?.tier_pricing?.[costKey] || this.modelsData?.tier_pricing?.[fallbackKey];
        if (tierPricing && tierPricing[tier]) {
            document.getElementById('bedrock-input-cost').value = tierPricing[tier].input || 0;
            document.getElementById('bedrock-output-cost').value = tierPricing[tier].output || 0;
            return;
        }

        // Fallback to cost_map (default pricing)
        const costs = this.modelsData?.cost_map?.[costKey] || this.modelsData?.cost_map?.[fallbackKey];
        if (costs) {
            document.getElementById('bedrock-input-cost').value = costs.input || 0;
            document.getElementById('bedrock-output-cost').value = costs.output || 0;
        } else {
            document.getElementById('bedrock-input-cost').value = 0;
            document.getElementById('bedrock-output-cost').value = 0;
        }
    },

    _updateSelectedModelsTable() {
        const config = State.getConfig();
        const container = document.getElementById('selected-models-container');
        if (!container) return;

        container.innerHTML = `
            <h3 class="st-subheader">Selected Models (${config.selected_models?.length || 0})</h3>
            ${config.selected_models?.length > 0 ? `
                <div class="ds-card ds-card--flush" style="max-height: 250px; overflow-y: auto;">
                    <table class="ds-table ds-table--compact">
                        <thead>
                            <tr><th>Model</th><th>Region</th><th>Input Cost</th><th>Output Cost</th><th>Service Tier</th><th>RPM</th><th></th></tr>
                        </thead>
                        <tbody>
                            ${config.selected_models.map((model, index) => `
                                <tr>
                                    <td><strong>${this.extractModelName(model.id)}</strong></td>
                                    <td><span class="ds-chip ds-chip--info">${model.region}</span></td>
                                    <td>$${model.input_cost}</td>
                                    <td>$${model.output_cost}</td>
                                    <td><span class="ds-chip ds-chip--neutral">${model.service_tier || 'default'}</span></td>
                                    <td>${model.target_rpm || '-'}</td>
                                    <td><button class="ds-btn ds-btn--danger ds-btn--sm ds-btn--icon remove-model" data-index="${index}"><i class="bi bi-x"></i></button></td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <button class="ds-btn ds-btn--secondary ds-mt-2" id="clear-models">Clear All Models</button>
                ${(() => {
                    const allBedrockItems = [
                        ...(config.selected_models || []).filter(m => m.id && m.id.includes('bedrock/')),
                        ...(config.judge_models || []).filter(m => m.id && m.id.includes('bedrock/'))
                    ];
                    const bedrockRegions = new Set(allBedrockItems.map(m => m.region));
                    if (bedrockRegions.size > 1) {
                        return '<div class="ds-alert ds-alert--warning ds-mt-3"><i class="bi bi-exclamation-triangle ds-alert__icon"></i><div class="ds-alert__content"><div class="ds-alert__title">Multi-Region Evaluation</div><div class="ds-alert__message">You have selected Bedrock models and/or judges across <strong>' + bedrockRegions.size + ' regions</strong> (' + [...bedrockRegions].join(', ') + ').<br><br>Short-term (temporary) Bedrock API keys only allow inference in the region where they were generated. To evaluate across multiple regions, you must use a <strong>long-term API key</strong>.</div></div></div>';
                    }
                    return '';
                })()}
            ` : `
                <div class="ds-alert ds-alert--info"><i class="bi bi-info-circle ds-alert__icon"></i><div class="ds-alert__content"><div class="ds-alert__message">No models selected. Add models above to begin evaluation.</div></div></div>
            `}
        `;

        // Rebind remove and clear buttons
        document.querySelectorAll('.remove-model').forEach(btn => {
            btn.addEventListener('click', () => {
                State.removeSelectedModel(parseInt(btn.dataset.index));
                this._updateSelectedModelsTable();
            });
        });
        document.getElementById('clear-models')?.addEventListener('click', () => {
            State.clearSelectedModels();
            this._updateSelectedModelsTable();
        });
    },

    addOtherModel() {
        const providerSelect = document.getElementById('other-provider-select');
        const modelIdInput = document.getElementById('other-model-id');
        const modelId = modelIdInput?.value?.trim();

        if (!modelId) {
            App.showNotification('Error', 'Please enter a model ID', 'error');
            return;
        }

        // Build full model ID with provider prefix
        let prefix = providerSelect.value;
        if (prefix === 'custom') {
            prefix = document.getElementById('other-custom-prefix')?.value?.trim() || '';
            if (!prefix) {
                App.showNotification('Error', 'Please enter a custom provider prefix', 'error');
                return;
            }
            if (!prefix.endsWith('/')) prefix += '/';
        }

        const fullModelId = `${prefix}${modelId}`;
        const region = document.getElementById('other-region')?.value?.trim() || 'N/A';
        const inputCost = parseFloat(document.getElementById('other-input-cost')?.value) || 0;
        const outputCost = parseFloat(document.getElementById('other-output-cost')?.value) || 0;

        State.addSelectedModel({
            id: fullModelId,
            region: region,
            input_cost: inputCost,
            output_cost: outputCost,
            service_tier: 'default',
            target_rpm: null
        });

        // Clear the input for next entry
        modelIdInput.value = '';
        this._updateSelectedModelsTable();
        App.showNotification('Added', `Model ${fullModelId} added`);
    },

    addModel(type) {
        const select = document.getElementById(`${type}-model-select`);
        const modelId = select.value;

        if (!modelId) {
            App.showNotification('Error', 'Please select a model', 'error');
            return;
        }

        const region = select.selectedOptions[0].dataset.region || this.selectedRegion;
        const inputCost = parseFloat(document.getElementById(`${type}-input-cost`).value) || 0;
        const outputCost = parseFloat(document.getElementById(`${type}-output-cost`).value) || 0;
        const serviceTier = document.getElementById(`${type}-service-tier`)?.value || 'default';
        const targetRpm = document.getElementById(`${type}-target-rpm`)?.value || null;

        State.addSelectedModel({
            id: modelId,
            region: region,
            input_cost: inputCost,
            output_cost: outputCost,
            service_tier: serviceTier,
            target_rpm: targetRpm ? parseInt(targetRpm) : null
        });

        this._updateSelectedModelsTable();
        App.showNotification('Added', `Model ${this.extractModelName(modelId)} added`);
    },

    /**
     * Add a judge model
     */
    addJudge() {
        const select = document.getElementById('judge-select');
        const modelId = select.value;

        if (!modelId) {
            App.showNotification('Error', 'Please select a judge model', 'error');
            return;
        }

        const region = select.selectedOptions[0].dataset.region || this.selectedRegion;
        const inputCost = parseFloat(document.getElementById('judge-input-cost').value) || 0;
        const outputCost = parseFloat(document.getElementById('judge-output-cost').value) || 0;

        State.addJudgeModel({
            id: modelId,
            region: region,
            input_cost: inputCost,
            output_cost: outputCost
        });

        this.renderModelConfig();
        App.showNotification('Added', `Judge ${this.extractModelName(modelId)} added`);
    },

    /**
     * Render Advanced Configuration sub-tab (Streamlit-like layout)
     */
    renderAdvancedConfig() {
        const config = State.getConfig();
        const container = document.getElementById('advanced-config-content');

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <h3 class="st-subheader" style="margin-top: 0;">Advanced Parameters</h3>

                <div class="st-columns-2">
                    <!-- Column 1 -->
                    <div class="ds-card">
                        <h4 class="ds-card__title ds-mb-4">Execution Settings</h4>

                        <label class="st-label">Parallel API Calls (1-20)</label>
                        <input type="number" class="ds-input ds-input--standard" id="adv-parallel-calls"
                               min="1" max="20" value="${config.parallel_calls || 4}">
                        <p class="ds-helper">Number of concurrent API requests</p>

                        <label class="st-label ds-mt-4">Invocations per Scenario (1-20)</label>
                        <input type="number" class="ds-input ds-input--standard" id="adv-invocations"
                               min="1" max="20" value="${config.invocations_per_scenario || 3}">
                        <p class="ds-helper">Repeat each scenario this many times</p>

                        <label class="st-label ds-mt-4">Pass/Failure Threshold (2-4)</label>
                        <input type="number" class="ds-input ds-input--standard" id="adv-threshold"
                               min="2" max="4" value="${config.failure_threshold || 3}">
                        <p class="ds-helper">Retry threshold for failed invocations</p>
                    </div>

                    <!-- Column 2 -->
                    <div class="ds-card">
                        <h4 class="ds-card__title ds-mb-4">Experiment Settings</h4>

                        <label class="st-label">Sleep Between Invocations (0-300s)</label>
                        <input type="number" class="ds-input ds-input--standard" id="adv-sleep"
                               min="0" max="300" value="${config.sleep_between_invocations || 3}">
                        <p class="ds-helper">Delay between API calls in seconds</p>

                        <label class="st-label ds-mt-4">Experiment Counts (1-10)</label>
                        <input type="number" class="ds-input ds-input--standard" id="adv-experiment-counts"
                               min="1" max="10" value="${config.experiment_counts || 1}">
                        <p class="ds-helper">Number of experiment runs</p>

                        <label class="st-label ds-mt-4">Temperature Variations (0-5)</label>
                        <input type="number" class="ds-input ds-input--standard" id="adv-temp-variations"
                               min="0" max="5" value="${config.temperature_variations || 0}">
                        <p class="ds-helper">Number of temperature variations to test</p>

                        <label class="st-label ds-mt-4">Experiment Wait Time</label>
                        <select class="ds-select" id="adv-wait-time">
                            <option value="0" ${config.experiment_wait_time === 0 ? 'selected' : ''}>No wait</option>
                            <option value="60" ${config.experiment_wait_time === 60 ? 'selected' : ''}>1 minute</option>
                            <option value="300" ${config.experiment_wait_time === 300 ? 'selected' : ''}>5 minutes</option>
                            <option value="600" ${config.experiment_wait_time === 600 ? 'selected' : ''}>10 minutes</option>
                            <option value="1800" ${config.experiment_wait_time === 1800 ? 'selected' : ''}>30 minutes</option>
                            <option value="3600" ${config.experiment_wait_time === 3600 ? 'selected' : ''}>1 hour</option>
                        </select>
                        <p class="ds-helper">Wait time between experiment runs</p>
                    </div>
                </div>
            </div>
        `;

        this.bindAdvancedConfigEvents();
    },

    /**
     * Bind events for Advanced Configuration
     */
    bindAdvancedConfigEvents() {
        document.getElementById('adv-parallel-calls')?.addEventListener('change', (e) => {
            State.updateConfig({ parallel_calls: parseInt(e.target.value) || 4 });
        });

        document.getElementById('adv-invocations')?.addEventListener('change', (e) => {
            State.updateConfig({ invocations_per_scenario: parseInt(e.target.value) || 3 });
        });

        document.getElementById('adv-threshold')?.addEventListener('change', (e) => {
            State.updateConfig({ failure_threshold: parseInt(e.target.value) || 3 });
        });

        document.getElementById('adv-sleep')?.addEventListener('change', (e) => {
            State.updateConfig({ sleep_between_invocations: parseInt(e.target.value) || 3 });
        });

        document.getElementById('adv-experiment-counts')?.addEventListener('change', (e) => {
            State.updateConfig({ experiment_counts: parseInt(e.target.value) || 1 });
        });

        document.getElementById('adv-temp-variations')?.addEventListener('change', (e) => {
            State.updateConfig({ temperature_variations: parseInt(e.target.value) || 0 });
        });

        document.getElementById('adv-wait-time')?.addEventListener('change', (e) => {
            State.updateConfig({ experiment_wait_time: parseInt(e.target.value) || 0 });
        });
    },

    /**
     * Save the current configuration as a new evaluation
     */
    async saveConfiguration() {
        const config = State.getConfig();

        // Validation
        if (!config.name) {
            App.showNotification('Error', 'Please enter an evaluation name', 'error');
            return;
        }

        if (!config.temp_path) {
            App.showNotification('Error', 'Please upload a CSV file', 'error');
            return;
        }

        const isMultiShot = config.evaluation_mode === 'multi_shot';

        if (isMultiShot) {
            const turns = config.task_evaluations || [];
            if (turns.length === 0) {
                App.showNotification('Error', 'Multi-shot evaluation needs at least one turn', 'error');
                return;
            }
            for (let i = 0; i < turns.length; i++) {
                if (!turns[i].prompt_column) {
                    App.showNotification('Error', `Turn ${i + 1}: select a Prompt Column`, 'error');
                    return;
                }
                if (!turns[i].golden_answer_column) {
                    App.showNotification('Error', `Turn ${i + 1}: select a Golden Answer Column`, 'error');
                    return;
                }
            }
        } else {
            if (!config.prompt_column) {
                App.showNotification('Error', 'Please select a prompt column', 'error');
                return;
            }
            if (config.golden_answer_mode !== 'criteria_only' && !config.golden_answer_column) {
                App.showNotification('Error', 'Please select a golden answer column (or switch to success criteria mode)', 'error');
                return;
            }
            if (config.golden_answer_mode === 'criteria_only') {
                const sc = config.success_criteria || {};
                if (!sc.must_include?.trim() || !sc.success_definition?.trim()) {
                    App.showNotification('Error', 'Success criteria requires "Must Include" and "Success Definition"', 'error');
                    return;
                }
            }
        }

        if (config.selected_models?.length === 0) {
            App.showNotification('Error', 'Please add at least one model', 'error');
            return;
        }

        if (!config.latency_only_mode) {
            if (config.eval_mode === 'specialist') {
                // Validate specialist: at least one metric must have a primary model assigned
                const assignments = config.metric_assignments || {};
                const hasAssignment = Object.values(assignments).some(a => a.primary?.id);
                if (!hasAssignment) {
                    App.showNotification('Error', 'Please assign at least one primary model in the metric table', 'error');
                    return;
                }
            } else if (config.judge_models?.length === 0) {
                App.showNotification('Error', 'Please add at least one judge model (or enable latency-only mode)', 'error');
                return;
            }
        }

        // Validate task type and criteria (required unless latency-only mode)
        if (!config.latency_only_mode) {
            const tasks = config.task_evaluations || [];
            for (let i = 0; i < tasks.length; i++) {
                if (!tasks[i].task_type?.trim()) {
                    App.showNotification('Error', `Task ${i + 1}: Please enter a Task Type`, 'error');
                    return;
                }
                if (!tasks[i].task_criteria?.trim()) {
                    App.showNotification('Error', `Task ${i + 1}: Please enter Task Criteria`, 'error');
                    return;
                }
            }
        }

        // APO validation — only when prompt optimization is enabled
        if ((config.prompt_optimization_mode || 'none') !== 'none') {
            const evaluator = config.apo_evaluator || 'llmj';
            if (evaluator === 'llmj') {
                if (!(config.apo_llmj_rubric || '').trim()) {
                    App.showNotification('Error', 'Prompt Optimization (LLM-as-Judge): rubric is required', 'error');
                    return;
                }
                if (!(config.apo_llmj_judge_model || '').trim()) {
                    App.showNotification('Error', 'Prompt Optimization (LLM-as-Judge): pick a judge model', 'error');
                    return;
                }
            } else if (evaluator === 'steering') {
                const criteria = (config.apo_steering_criteria || []).filter(c => (c || '').trim());
                if (criteria.length === 0) {
                    App.showNotification('Error', 'Prompt Optimization (Steering): provide at least one criterion', 'error');
                    return;
                }
            }
        }

        try {
            // Validate API credentials before saving
            App.showLoading(true);
            App.showNotification('Validating', 'Checking API credentials...', 'info');
            const allModels = [...(config.selected_models || []), ...(config.judge_models || [])];
            if (config.eval_mode === 'specialist') {
                Object.values(config.metric_assignments || {}).forEach(a => {
                    if (a.primary) allModels.push(a.primary);
                    if (a.secondary) allModels.push(a.secondary);
                });
            }
            const credCheck = await API.validateCredentials(allModels);
            if (!credCheck.valid) {
                App.showLoading(false);
                App.showNotification('Credential Error', credCheck.errors.join('\n'), 'error');
                return;
            }

            // Build the payload. In multi-shot mode, transform task_evaluations
            // into a turns spec; the backend creates N independent evaluations
            // (one per turn) named {base_name}_1, {base_name}_2, ...
            let payload = config;
            if (isMultiShot) {
                const turns = (config.task_evaluations || []).map(t => ({
                    prompt_column: t.prompt_column,
                    golden_answer_column: t.golden_answer_column,
                    task_type: t.task_type || '',
                    task_criteria: t.task_criteria || '',
                    temperature: t.temperature ?? 0.7,
                    user_defined_metrics: t.user_defined_metrics || '',
                    structured_output_format: t.structured_output_format || null,
                }));
                payload = { ...config, evaluation_mode: 'multi_shot', turns };
            } else {
                payload = { ...config, evaluation_mode: 'single_shot' };
            }
            const result = await API.createEvaluation(payload);

            if (result.success) {
                App.showNotification('Success', `Created ${result.evaluations.length} evaluation(s)`);
                State.resetConfig();
                this.renderEvaluationSetup();
                this.renderModelConfig();

                // Refresh monitor and switch to it
                await MonitorComponent.loadEvaluations();
                MonitorComponent.render();
                App.switchTab('monitor');
            } else {
                App.showNotification('Error', result.error || 'Failed to create evaluation', 'error');
            }
        } catch (error) {
            App.showNotification('Error', `Failed to save configuration: ${error.message}`, 'error');
        } finally {
            App.showLoading(false);
        }
    },

    /**
     * Get models available in a region
     */
    getModelsForRegion(region) {
        const regionModels = this.modelsData?.region_to_models?.[region] || [];
        // Return unique models available in this region as [modelId, region] pairs
        const seen = new Set();
        return regionModels
            .filter(modelId => { if (seen.has(modelId)) return false; seen.add(modelId); return true; })
            .map(modelId => [modelId, region]);
    },

    /**
     * Render metric-to-model assignment rows for specialist mode
     */
    _renderMetricAssignmentRows(config, judges) {
        const standardMetrics = ['Correctness', 'Completeness', 'Relevance', 'Format', 'Coherence', 'Following-instructions'];
        const customMetrics = (config.custom_metrics || []).map(cm => cm.metric_name);
        const allMetrics = [...standardMetrics, ...customMetrics];
        const assignments = config.metric_assignments || {};
        const metricsInfo = this.metricsData?.metrics || {};

        return allMetrics.map(metric => {
            const a = assignments[metric] || {};
            const primaryId = a.primary?.id || a.primary?.model_id || '';
            const secondaryId = a.secondary?.id || a.secondary?.model_id || '';
            const threshold = a.threshold ?? 3;
            const info = metricsInfo[metric];
            const tooltipContent = info
                ? `${info.definition}&#10;Does NOT assess: ${info.boundary || 'N/A'}&#10;Rubric: 1=${info.rubric?.['1'] || ''}, 5=${info.rubric?.['5'] || ''}`
                : '';

            return `
                <tr>
                    <td>
                        <strong>${metric}</strong>
                        ${tooltipContent ? `<span title="${tooltipContent}" style="cursor: help; color: var(--text-secondary); margin-left: 4px;">&#9432;</span>` : ''}
                    </td>
                    <td>
                        <select class="ds-select metric-primary-select" data-metric="${metric}" style="font-size: 0.85rem;">
                            <option value="">Select model...</option>
                            ${judges.map(([modelId]) => `
                                <option value="${modelId}" ${modelId === primaryId ? 'selected' : ''}>
                                    ${this.extractModelName(modelId)}
                                </option>
                            `).join('')}
                        </select>
                    </td>
                    <td>
                        <select class="ds-select metric-secondary-select" data-metric="${metric}" style="font-size: 0.85rem;">
                            <option value="">-- none --</option>
                            ${judges.map(([modelId]) => `
                                <option value="${modelId}" ${modelId === secondaryId ? 'selected' : ''}>
                                    ${this.extractModelName(modelId)}
                                </option>
                            `).join('')}
                        </select>
                    </td>
                    <td>
                        <input type="number" class="ds-input ds-input--standard metric-threshold"
                            data-metric="${metric}" value="${threshold}" min="1" max="5" style="width: 60px; font-size: 0.85rem;">
                    </td>
                </tr>
            `;
        }).join('');
    },

    /**
     * Show custom metric builder modal/form
     */
    _showCustomMetricForm() {
        const judges = this.getJudgesForRegion(this.selectedRegion);
        const overlay = document.createElement('div');
        overlay.id = 'custom-metric-overlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);z-index:1000;display:flex;align-items:center;justify-content:center;';
        overlay.innerHTML = `
            <div class="ds-card" style="width: 500px; max-height: 80vh; overflow-y: auto; padding: 1.5rem;">
                <h3 style="margin-top:0;">Add Custom Metric</h3>
                <div style="display:flex;flex-direction:column;gap:0.75rem;">
                    <div>
                        <label class="st-label">Metric Name *</label>
                        <input type="text" class="ds-input ds-input--standard" id="cm-name" placeholder="e.g. SQL_accuracy">
                    </div>
                    <div>
                        <label class="st-label">Definition *</label>
                        <textarea class="ds-input ds-input--standard" id="cm-definition" rows="2" placeholder="What does this metric measure?"></textarea>
                    </div>
                    <div>
                        <label class="st-label">Boundary (optional)</label>
                        <input type="text" class="ds-input ds-input--standard" id="cm-boundary" placeholder="Does NOT assess...">
                    </div>
                    <div>
                        <label class="st-label">Rubric (all levels required) *</label>
                        ${[1,2,3,4,5].map(level => `
                            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
                                <span style="font-weight:600;width:20px;">${level}</span>
                                <input type="text" class="ds-input ds-input--standard cm-rubric" data-level="${level}" placeholder="Score ${level} description">
                            </div>
                        `).join('')}
                    </div>
                    <div>
                        <label class="st-label">Primary Model *</label>
                        <select class="ds-select" id="cm-primary">
                            <option value="">Select model...</option>
                            ${judges.map(([modelId]) => `
                                <option value="${modelId}">${this.extractModelName(modelId)}</option>
                            `).join('')}
                        </select>
                    </div>
                    <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:0.5rem;">
                        <button class="ds-btn ds-btn--secondary" id="cm-cancel">Cancel</button>
                        <button class="ds-btn ds-btn--primary" id="cm-save">Add Metric</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);

        document.getElementById('cm-cancel').addEventListener('click', () => overlay.remove());
        document.getElementById('cm-save').addEventListener('click', () => {
            const name = document.getElementById('cm-name').value.trim().replace(/\s+/g, '-');
            const definition = document.getElementById('cm-definition').value.trim();
            const boundary = document.getElementById('cm-boundary').value.trim();
            const primaryId = document.getElementById('cm-primary').value;
            const rubric = {};
            let rubricValid = true;
            document.querySelectorAll('.cm-rubric').forEach(el => {
                const val = el.value.trim();
                if (!val) rubricValid = false;
                rubric[el.dataset.level] = val;
            });

            if (!name || !definition || !rubricValid || !primaryId) {
                App.showNotification('Error', 'Please fill in all required fields (name, definition, all 5 rubric levels, primary model)', 'error');
                return;
            }

            const region = this.selectedRegion;
            const costKey = `${primaryId}|${region}`;
            const fallbackKey = `${primaryId}|N/A`;
            const costs = this.judgesData?.cost_map?.[costKey] || this.judgesData?.cost_map?.[fallbackKey] || {};

            const metric = {
                metric_name: name,
                definition,
                boundary: boundary || undefined,
                rubric,
                primary: { id: primaryId, region, input_cost: costs.input || 0, output_cost: costs.output || 0 },
            };

            State.addCustomMetric(metric);
            // Also add metric assignment
            State.updateMetricAssignment(name, {
                primary: metric.primary,
                threshold: 3,
            });

            overlay.remove();
            this.renderModelConfig();
            App.showNotification('Added', `Custom metric "${name}" added`);
        });
    },

    /**
     * Get judges available in a region
     */
    getJudgesForRegion(region) {
        const regionJudges = this.judgesData?.region_to_models?.[region] || [];
        const seen = new Set();
        return regionJudges
            .filter(modelId => { if (seen.has(modelId)) return false; seen.add(modelId); return true; })
            .map(modelId => [modelId, region]);
    },

    /**
     * Get all available judges across all regions
     */
    getAllJudges() {
        const allJudges = this.judgesData?.judges || [];
        const seen = new Set();
        return allJudges.filter(([modelId, region]) => {
            const key = `${modelId}|${region}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
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
window.SetupComponent = SetupComponent;
