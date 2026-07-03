/**
 * Main application logic for 360-eval dashboard
 */

const App = {
    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing 360-eval Dashboard...');

        // Load user profile and config in parallel
        await Promise.all([
            this.loadUserProfile(),
            this.loadConfig(),
        ]);

        // Initialize components
        await this.initComponents();

        // Setup navigation
        this.setupNavigation();

        // Setup sub-tabs
        this.setupSubTabs();

        // Update debug info (non-blocking)
        this.updateDebugInfo();

        console.log('Dashboard initialized successfully');
    },

    /**
     * Load authenticated user profile and display in sidebar
     */
    async loadUserProfile() {
        try {
            const profile = await API.getUserProfile();
            if (profile) {
                State.set('userProfile', profile);
                const display = document.getElementById('user-display');
                if (display) {
                    display.textContent = profile.user_id || profile.email || 'Unknown';
                }
            }
        } catch (error) {
            console.error('Failed to load user profile:', error);
            const display = document.getElementById('user-display');
            if (display) {
                display.textContent = 'Not authenticated';
            }
        }
    },

    /**
     * Load application configuration
     */
    async loadConfig() {
        try {
            const config = await API.getConfig();
            State.set('config', config);

            // Update debug info
            document.getElementById('debug-project-root').textContent =
                config.project_root || 'Unknown';
            document.getElementById('debug-output-dir').textContent =
                config.output_dir || 'Unknown';
        } catch (error) {
            console.error('Failed to load config:', error);
            this.showNotification('Error', 'Failed to load application configuration', 'error');
        }
    },

    /**
     * Initialize all components
     */
    async initComponents() {
        try {
            // Initialize setup first (loads models/judges needed by other UI)
            // and monitor+credentials in parallel (independent data)
            await Promise.all([
                SetupComponent.init(),
                MonitorComponent.init(),
                CredentialsComponent.init(),
            ]);

            // Initialize remaining components in parallel
            // These are non-critical tabs — load them without blocking the UI
            Promise.all([
                EvaluationsComponent.init(),
                ReportsComponent.init(),
                UnprocessedComponent.init(),
                AdminComponent.init().then(() => {
                    if (AdminComponent.isAdmin) {
                        const adminNav = document.getElementById('admin-nav-item');
                        if (adminNav) adminNav.style.display = '';
                    }
                }),
            ]).catch(err => console.error('Deferred component init error:', err));
        } catch (error) {
            console.error('Failed to initialize components:', error);
        }
    },

    /**
     * Setup navigation between tabs
     */
    setupNavigation() {
        // Radio-style navigation
        const navRadios = document.querySelectorAll('#main-nav input[type="radio"]');
        navRadios.forEach(radio => {
            radio.addEventListener('change', () => {
                if (radio.checked) {
                    this.switchTab(radio.value);
                }
            });
        });

        // Restore active tab from state
        const activeTab = State.get('activeTab') || 'setup';
        this.switchTab(activeTab);
    },

    /**
     * Setup sub-tabs for Setup page
     */
    setupSubTabs() {
        const setupTabs = document.querySelectorAll('#setup-tabs .st-tab');
        setupTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.dataset.tab;

                // Update tab buttons
                setupTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Update tab content
                document.querySelectorAll('.setup-content .st-tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(tabId)?.classList.add('active');

                // Re-render the sub-tab content when switching
                if (tabId === 'model-config') {
                    SetupComponent.renderModelConfig();
                }
            });
        });
    },

    /**
     * Switch to a specific tab
     */
    switchTab(tabId) {
        // Update radio buttons
        const navRadios = document.querySelectorAll('#main-nav input[type="radio"]');
        navRadios.forEach(radio => {
            radio.checked = (radio.value === tabId);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tabId}`);
        });

        // Save active tab
        State.set('activeTab', tabId);

        // Refresh tab data if needed
        this.refreshTabData(tabId);
    },

    /**
     * Refresh data for a specific tab
     */
    async refreshTabData(tabId) {
        switch (tabId) {
            case 'setup':
                // Re-render setup component to pick up any loaded configuration
                if (typeof SetupComponent !== 'undefined') {
                    SetupComponent.render();
                }
                break;
            case 'monitor': {
                const prevMonitor = MonitorComponent.evaluations?.map(e => `${e.id}:${e.status}:${e.progress}`).join(',');
                await MonitorComponent.loadEvaluations();
                await MonitorComponent.checkBedrockKey();
                const newMonitor = MonitorComponent.evaluations?.map(e => `${e.id}:${e.status}:${e.progress}`).join(',');
                if (prevMonitor !== newMonitor) MonitorComponent.render();
                break;
            }
            case 'evaluations': {
                const prevEvals = JSON.stringify(EvaluationsComponent.evaluations?.map(e => `${e.id}:${e.status}`));
                await EvaluationsComponent.loadEvaluations();
                const newEvals = JSON.stringify(EvaluationsComponent.evaluations?.map(e => `${e.id}:${e.status}`));
                if (prevEvals !== newEvals) EvaluationsComponent.render();
                break;
            }
            case 'reports':
                await ReportsComponent.loadData();
                ReportsComponent.render();
                break;
            case 'unprocessed': {
                const prevUnprocessed = UnprocessedComponent.data?.records?.length;
                await UnprocessedComponent.loadData();
                const newUnprocessed = UnprocessedComponent.data?.records?.length;
                if (prevUnprocessed !== newUnprocessed) UnprocessedComponent.render();
                break;
            }
            case 'credentials':
                await CredentialsComponent.loadCredentials();
                CredentialsComponent.render();
                break;
            case 'admin': {
                const prevAdmin = JSON.stringify(AdminComponent.data?.users?.length);
                await AdminComponent.loadData();
                const newAdmin = JSON.stringify(AdminComponent.data?.users?.length);
                if (prevAdmin !== newAdmin) AdminComponent.render();
                break;
            }
        }
    },

    /**
     * Update debug information
     */
    async updateDebugInfo() {
        try {
            const result = await API.getEvaluations();
            document.getElementById('debug-eval-count').textContent =
                result.evaluations?.length || 0;
        } catch (error) {
            console.error('Failed to update debug info:', error);
        }
    },

    /**
     * Show a toast notification
     */
    showNotification(title, message, type = 'info') {
        const toast = document.getElementById('notification-toast');
        const toastTitle = document.getElementById('toast-title');
        const toastMessage = document.getElementById('toast-message');

        toastTitle.textContent = title;
        toastMessage.textContent = message;

        // Update toast styling based on type
        toast.classList.remove('bg-success', 'bg-danger', 'bg-warning');
        if (type === 'error') {
            toast.classList.add('bg-danger');
        } else if (type === 'success') {
            toast.classList.add('bg-success');
        }

        const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
        bsToast.show();
    },

    /**
     * Show/hide loading overlay
     */
    showLoading(show) {
        let overlay = document.getElementById('loading-overlay');

        if (show) {
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'loading-overlay';
                overlay.className = 'loading-overlay';
                overlay.innerHTML = `
                    <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                `;
                document.body.appendChild(overlay);
            }
            overlay.classList.remove('hidden');
        } else if (overlay) {
            overlay.classList.add('hidden');
        }
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

// Export for use in other modules
window.App = App;
