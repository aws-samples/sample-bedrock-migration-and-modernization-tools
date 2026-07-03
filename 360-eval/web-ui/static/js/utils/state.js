/**
 * Frontend state management
 * Mirrors backend state and persists to localStorage
 */

const State = {
    // Default state values
    defaults: {
        currentEvaluationConfig: {
            name: '',
            csv_file_name: null,
            temp_path: null,
            columns: [],
            preview: [],
            prompt_column: null,
            golden_answer_column: null,
            vision_enabled: false,
            image_column: null,
            latency_only_mode: false,
            stream_evaluation: true,
            prompt_optimization_mode: 'none',
            // APO config (used only when prompt_optimization_mode != 'none')
            apo_evaluator: 'llmj',
            apo_llmj_rubric: '',
            apo_llmj_judge_model: '',
            apo_steering_criteria: ['', '', '', '', ''],
            task_evaluations: [{ task_type: '', task_criteria: '', temperature: 0.7, user_defined_metrics: '', structured_output_format: null, prompt_column: null, golden_answer_column: null }],
            evaluation_mode: 'single_shot',  // 'single_shot' or 'multi_shot' (multi-shot now = N independent single-shot evals)
            selected_models: [],
            judge_models: [],
            eval_mode: 'bundled',  // 'specialist' or 'bundled'
            metric_assignments: {},  // metric_name → { primary: {...}, secondary: null, threshold: 3 }
            custom_metrics: [],  // [{ metric_name, definition, rubric: {1:..,5:..}, boundary?, primary: {...}, secondary? }]
            golden_answer_mode: 'golden_answer',  // 'golden_answer' or 'criteria_only'
            success_criteria: { must_include: '', success_definition: '', must_not_include: '', edge_cases: '' },
            parallel_calls: 4,
            invocations_per_scenario: 3,
            sleep_between_invocations: 3,
            experiment_counts: 1,
            temperature_variations: 0,
            failure_threshold: 3,
            experiment_wait_time: 0
        },
        evaluations: [],
        reports: [],
        config: null,
        models: null,
        judges: null,
        activeTab: 'setup'
    },

    // Current state
    state: {},

    /**
     * Initialize state from localStorage or defaults
     */
    init() {
        const saved = localStorage.getItem('360eval_state');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                // Merge with defaults to handle new fields
                this.state = {
                    ...this.defaults,
                    ...parsed,
                    currentEvaluationConfig: {
                        ...this.defaults.currentEvaluationConfig,
                        ...(parsed.currentEvaluationConfig || {})
                    }
                };
            } catch (e) {
                console.error('Failed to parse saved state:', e);
                this.state = { ...this.defaults };
            }
        } else {
            this.state = { ...this.defaults };
        }
    },

    /**
     * Save state to localStorage.
     * Only persists UI-local state (config form, active tab).
     * Evaluations and reports are always fetched from the API.
     */
    save() {
        try {
            const toSave = {
                currentEvaluationConfig: {
                    ...this.state.currentEvaluationConfig,
                    preview: [] // Don't save preview data
                },
                activeTab: this.state.activeTab,
                // evaluations and reports are NOT persisted — they come from DynamoDB
            };
            localStorage.setItem('360eval_state', JSON.stringify(toSave));
        } catch (e) {
            console.error('Failed to save state:', e);
        }
    },

    /**
     * Get a state value
     */
    get(key) {
        return this.state[key];
    },

    /**
     * Set a state value
     */
    set(key, value) {
        this.state[key] = value;
        this.save();
    },

    /**
     * Update current evaluation config
     */
    updateConfig(updates) {
        this.state.currentEvaluationConfig = {
            ...this.state.currentEvaluationConfig,
            ...updates
        };
        this.save();
    },

    /**
     * Get current evaluation config
     */
    getConfig() {
        return this.state.currentEvaluationConfig;
    },

    /**
     * Reset current evaluation config to defaults
     */
    resetConfig() {
        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
        this.state.currentEvaluationConfig = {
            ...this.defaults.currentEvaluationConfig,
            name: `Evaluation-${timestamp}`
        };
        this.save();
    },

    /**
     * Add a task evaluation
     */
    addTaskEvaluation() {
        const tasks = this.state.currentEvaluationConfig.task_evaluations || [];
        tasks.push({ task_type: '', task_criteria: '', temperature: 0.7, user_defined_metrics: '' });
        this.updateConfig({ task_evaluations: tasks });
    },

    /**
     * Remove a task evaluation
     */
    removeTaskEvaluation(index) {
        const tasks = this.state.currentEvaluationConfig.task_evaluations || [];
        if (tasks.length > 1) {
            tasks.splice(index, 1);
            this.updateConfig({ task_evaluations: tasks });
        }
    },

    /**
     * Update a task evaluation
     */
    updateTaskEvaluation(index, updates) {
        const tasks = [...(this.state.currentEvaluationConfig.task_evaluations || [])];
        tasks[index] = { ...tasks[index], ...updates };
        this.updateConfig({ task_evaluations: tasks });
    },

    /**
     * Add a model to selected models
     */
    addSelectedModel(model) {
        const models = [...(this.state.currentEvaluationConfig.selected_models || [])];
        // Check if already exists
        const exists = models.some(m => m.id === model.id && m.region === model.region);
        if (!exists) {
            models.push(model);
            this.updateConfig({ selected_models: models });
        }
    },

    /**
     * Remove a model from selected models
     */
    removeSelectedModel(index) {
        const models = [...(this.state.currentEvaluationConfig.selected_models || [])];
        models.splice(index, 1);
        this.updateConfig({ selected_models: models });
    },

    /**
     * Clear all selected models
     */
    clearSelectedModels() {
        this.updateConfig({ selected_models: [] });
    },

    /**
     * Add a judge model
     */
    addJudgeModel(judge) {
        const judges = [...(this.state.currentEvaluationConfig.judge_models || [])];
        const exists = judges.some(j => j.id === judge.id && j.region === judge.region);
        if (!exists) {
            judges.push(judge);
            this.updateConfig({ judge_models: judges });
        }
    },

    /**
     * Remove a judge model
     */
    removeJudgeModel(index) {
        const judges = [...(this.state.currentEvaluationConfig.judge_models || [])];
        judges.splice(index, 1);
        this.updateConfig({ judge_models: judges });
    },

    /**
     * Clear all judge models
     */
    clearJudgeModels() {
        this.updateConfig({ judge_models: [] });
    },

    /**
     * Set evaluation mode
     */
    setEvalMode(mode) {
        this.updateConfig({ eval_mode: mode });
    },

    /**
     * Update metric assignment for specialist mode
     */
    updateMetricAssignment(metricName, assignment) {
        const assignments = { ...(this.state.currentEvaluationConfig.metric_assignments || {}) };
        assignments[metricName] = { ...(assignments[metricName] || {}), ...assignment };
        this.updateConfig({ metric_assignments: assignments });
    },

    /**
     * Add a custom metric
     */
    addCustomMetric(metric) {
        const metrics = [...(this.state.currentEvaluationConfig.custom_metrics || [])];
        const exists = metrics.some(m => m.metric_name === metric.metric_name);
        if (!exists) {
            metrics.push(metric);
            this.updateConfig({ custom_metrics: metrics });
        }
    },

    /**
     * Remove a custom metric
     */
    removeCustomMetric(index) {
        const metrics = [...(this.state.currentEvaluationConfig.custom_metrics || [])];
        const removed = metrics.splice(index, 1)[0];
        // Also remove from metric_assignments
        if (removed) {
            const assignments = { ...(this.state.currentEvaluationConfig.metric_assignments || {}) };
            delete assignments[removed.metric_name];
            this.updateConfig({ custom_metrics: metrics, metric_assignments: assignments });
        } else {
            this.updateConfig({ custom_metrics: metrics });
        }
    },

    /**
     * Load configuration from an existing evaluation
     * Matches Streamlit's _load_configuration behavior
     */
    loadFromEvaluation(evaluation) {
        // Generate new timestamp-based name
        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
        const newName = `Evaluation-${timestamp}`;

        // Create task evaluations from source config
        const taskEvaluations = [{
            task_type: evaluation.task_type || '',
            task_criteria: evaluation.task_criteria || '',
            temperature: evaluation.temperature || 0.7,
            user_defined_metrics: evaluation.user_defined_metrics || ''
        }];

        // Normalize models (handle both old and new formats)
        const normalizedModels = this._normalizeModels(evaluation.selected_models || []);
        const normalizedJudges = this._normalizeJudges(evaluation.judge_models || []);

        // Create new config with all parameters from source evaluation
        const newConfig = {
            // New evaluation metadata
            name: newName,

            // Data fields - need new upload
            csv_file_name: null,
            temp_path: null,
            columns: [],
            preview: [],
            prompt_column: null,
            golden_answer_column: null,

            // Copy task configuration
            task_evaluations: taskEvaluations,

            // Copy model configurations
            selected_models: normalizedModels,
            judge_models: normalizedJudges,

            // Copy evaluation settings
            vision_enabled: evaluation.vision_enabled || false,
            image_column: null,  // User will select after upload
            latency_only_mode: evaluation.latency_only_mode || false,
            stream_evaluation: evaluation.stream_evaluation !== undefined ? evaluation.stream_evaluation : true,
            prompt_optimization_mode: evaluation.prompt_optimization_mode || 'none',

            // Copy specialist/bundled config
            eval_mode: evaluation.eval_mode || 'bundled',
            metric_assignments: evaluation.metric_assignments || {},
            custom_metrics: evaluation.custom_metrics || [],
            golden_answer_mode: evaluation.golden_answer_mode || 'golden_answer',
            success_criteria: evaluation.success_criteria || this.defaults.currentEvaluationConfig.success_criteria,

            // Copy advanced parameters
            parallel_calls: evaluation.parallel_calls || this.defaults.currentEvaluationConfig.parallel_calls,
            invocations_per_scenario: evaluation.invocations_per_scenario || this.defaults.currentEvaluationConfig.invocations_per_scenario,
            sleep_between_invocations: evaluation.sleep_between_invocations || this.defaults.currentEvaluationConfig.sleep_between_invocations,
            experiment_counts: evaluation.experiment_counts || this.defaults.currentEvaluationConfig.experiment_counts,
            temperature_variations: evaluation.temperature_variations || this.defaults.currentEvaluationConfig.temperature_variations,
            failure_threshold: evaluation.failure_threshold || this.defaults.currentEvaluationConfig.failure_threshold,
            experiment_wait_time: evaluation.experiment_wait_time || this.defaults.currentEvaluationConfig.experiment_wait_time
        };

        // Replace the entire current config
        this.state.currentEvaluationConfig = newConfig;
        this.save();
    },

    /**
     * Normalize model data structure from loaded configuration
     */
    _normalizeModels(models) {
        if (!models || !Array.isArray(models)) return [];

        return models.map(model => {
            const normalized = {
                id: model.model_id || model.id,
                region: model.region || '',
                input_cost: model.input_token_cost || model.input_cost || 0,
                output_cost: model.output_token_cost || model.output_cost || 0
            };

            // Preserve service_tier if it exists
            if (model.service_tier) {
                normalized.service_tier = model.service_tier;
            }

            // Preserve target_rpm if it exists
            if (model.target_rpm !== undefined && model.target_rpm !== null) {
                normalized.target_rpm = model.target_rpm;
            }

            return normalized;
        });
    },

    /**
     * Normalize judge data structure from loaded configuration.
     * Handles flat list (bundled mode) — specialist mode uses metric_assignments instead.
     */
    _normalizeJudges(judges) {
        if (!judges || !Array.isArray(judges)) return [];

        return judges.map(judge => ({
            id: judge.model_id || judge.id,
            region: judge.region || '',
            input_cost: judge.input_token_cost || judge.input_cost || 0,
            output_cost: judge.output_token_cost || judge.output_cost || 0
        }));
    }
};

// Initialize state on load
State.init();

// Export for use in other modules
window.State = State;
