/**
 * Chart utility functions using Plotly.js
 */

const Charts = {
    // Default dark theme layout
    darkLayout: {
        paper_bgcolor: '#1c2432',
        plot_bgcolor: '#1c2432',
        font: {
            color: '#f2f3f3',
            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        },
        margin: { l: 60, r: 30, t: 50, b: 60 },
        xaxis: {
            gridcolor: '#30363d',
            linecolor: '#30363d',
            zerolinecolor: '#30363d'
        },
        yaxis: {
            gridcolor: '#30363d',
            linecolor: '#30363d',
            zerolinecolor: '#30363d'
        },
        colorway: ['#ff9900', '#00a1c9', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db']
    },

    /**
     * Create a horizontal bar chart for error types
     */
    createErrorTypeChart(containerId, data) {
        const counts = {};
        data.forEach(d => {
            const reason = this.normalizeError(d.reason || 'Unknown');
            counts[reason] = (counts[reason] || 0) + 1;
        });

        const sorted = Object.entries(counts).sort((a, b) => a[1] - b[1]);
        const labels = sorted.map(s => s[0]);
        const values = sorted.map(s => s[1]);

        const trace = {
            type: 'bar',
            orientation: 'h',
            x: values,
            y: labels,
            marker: {
                color: '#ff9900'
            }
        };

        const layout = {
            ...this.darkLayout,
            title: 'Distribution of Error Types',
            height: Math.max(400, labels.length * 35),
            xaxis: {
                ...this.darkLayout.xaxis,
                title: 'Failed Records'
            },
            yaxis: {
                ...this.darkLayout.yaxis,
                title: '',
                automargin: true
            }
        };

        // Plotly.react diffs against an existing plot instead of tearing it down and
        // rebuilding it (cheaper re-renders, preserves zoom/pan); same signature as
        // newPlot and creates the plot when the container is empty.
        Plotly.react(containerId, [trace], layout, { responsive: true });
    },

    /**
     * Create a bar chart for errors by model
     */
    createErrorByModelChart(containerId, data) {
        const counts = {};
        data.forEach(d => {
            const model = d.model_id || 'Unknown';
            counts[model] = (counts[model] || 0) + 1;
        });

        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const labels = sorted.map(s => this.extractModelName(s[0]));
        const values = sorted.map(s => s[1]);

        const trace = {
            type: 'bar',
            x: labels,
            y: values,
            marker: {
                color: '#00a1c9'
            }
        };

        const layout = {
            ...this.darkLayout,
            title: 'Errors by Model',
            height: 400,
            xaxis: {
                ...this.darkLayout.xaxis,
                title: 'Model',
                tickangle: -45
            },
            yaxis: {
                ...this.darkLayout.yaxis,
                title: 'Failed Records'
            }
        };

        // Plotly.react diffs against an existing plot instead of tearing it down and
        // rebuilding it (cheaper re-renders, preserves zoom/pan); same signature as
        // newPlot and creates the plot when the container is empty.
        Plotly.react(containerId, [trace], layout, { responsive: true });
    },

    /**
     * Create a bar chart for errors by task type
     */
    createErrorByTaskChart(containerId, data) {
        const counts = {};
        data.forEach(d => {
            const task = d.task_type || 'Unknown';
            counts[task] = (counts[task] || 0) + 1;
        });

        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const labels = sorted.map(s => s[0]);
        const values = sorted.map(s => s[1]);

        const trace = {
            type: 'bar',
            x: labels,
            y: values,
            marker: {
                color: '#2ecc71'
            }
        };

        const layout = {
            ...this.darkLayout,
            title: 'Errors by Task Type',
            height: 400,
            xaxis: {
                ...this.darkLayout.xaxis,
                title: 'Task Type',
                tickangle: -45
            },
            yaxis: {
                ...this.darkLayout.yaxis,
                title: 'Failed Records'
            }
        };

        // Plotly.react diffs against an existing plot instead of tearing it down and
        // rebuilding it (cheaper re-renders, preserves zoom/pan); same signature as
        // newPlot and creates the plot when the container is empty.
        Plotly.react(containerId, [trace], layout, { responsive: true });
    },

    /**
     * Normalize error messages for grouping
     */
    normalizeError(errorMsg) {
        if (!errorMsg || errorMsg === 'Unknown') {
            return 'Unknown';
        }
        errorMsg = String(errorMsg);

        // Match common error type patterns
        const match = errorMsg.match(/^(\w*Error|\w*Exception|\w*Timeout|\w*Failure)/i);
        if (match) {
            return match[1];
        }

        // Truncate to first 50 chars or up to first colon/hash
        const truncated = errorMsg.split(/[:\-#]/)[0].trim();
        return truncated.length > 50 ? truncated.substring(0, 50) : truncated;
    },

    /**
     * Extract model name from full model ID
     */
    extractModelName(modelId) {
        if (!modelId) return 'Unknown';

        // Remove bedrock/ prefix
        let name = modelId.replace(/^bedrock\//, '');

        // Remove region prefix (e.g., us.)
        name = name.replace(/^[a-z]{2}\./, '');

        // Truncate if too long
        if (name.length > 30) {
            name = name.substring(0, 27) + '...';
        }

        return name;
    },

    /**
     * Create a simple progress gauge
     */
    createProgressGauge(containerId, value, title = 'Progress') {
        const trace = {
            type: 'indicator',
            mode: 'gauge+number',
            value: value,
            title: { text: title, font: { size: 14 } },
            number: { suffix: '%' },
            gauge: {
                axis: { range: [0, 100], tickcolor: '#f2f3f3' },
                bar: { color: '#00a1c9' },
                bgcolor: '#232f3e',
                bordercolor: '#30363d',
                steps: [
                    { range: [0, 50], color: '#1c2432' },
                    { range: [50, 100], color: '#232f3e' }
                ]
            }
        };

        const layout = {
            ...this.darkLayout,
            height: 200,
            margin: { t: 50, b: 30, l: 30, r: 30 }
        };

        // Plotly.react diffs against an existing plot instead of tearing it down and
        // rebuilding it (cheaper re-renders, preserves zoom/pan); same signature as
        // newPlot and creates the plot when the container is empty.
        Plotly.react(containerId, [trace], layout, { responsive: true });
    },

    /**
     * Create a pie chart
     */
    createPieChart(containerId, labels, values, title = '') {
        const trace = {
            type: 'pie',
            labels: labels,
            values: values,
            hole: 0.4,
            textinfo: 'label+percent',
            textposition: 'outside',
            marker: {
                colors: this.darkLayout.colorway
            }
        };

        const layout = {
            ...this.darkLayout,
            title: title,
            height: 400,
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.1
            }
        };

        // Plotly.react diffs against an existing plot instead of tearing it down and
        // rebuilding it (cheaper re-renders, preserves zoom/pan); same signature as
        // newPlot and creates the plot when the container is empty.
        Plotly.react(containerId, [trace], layout, { responsive: true });
    }
};

// Export for use in other modules
window.Charts = Charts;
