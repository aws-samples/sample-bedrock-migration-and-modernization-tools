/**
 * Credentials tab component
 * Manages user API keys for LLM providers
 */

const CredentialsComponent = {
    credentials: [],
    notificationEmail: null,

    providers: [
        { id: 'bedrock', name: 'Amazon Bedrock', icon: 'bi-cloud', placeholder: 'Enter your Bedrock API key (also used for OpenAI/Mantle models)' },
        { id: 'openai', name: 'OpenAI', icon: 'bi-braces', placeholder: 'sk-...' },
        { id: 'google', name: 'Google Gemini', icon: 'bi-google', placeholder: 'Enter your Google API key' },
        { id: 'azure', name: 'Azure OpenAI', icon: 'bi-microsoft', placeholder: 'Enter your Azure API key' },
    ],

    async init() {
        await this.loadCredentials();
        this.render();
    },

    async loadCredentials() {
        try {
            const res = await API.getCredentials();
            this.credentials = res.credentials || [];
            this.notificationEmail = res.notification_email || null;
        } catch (error) {
            console.error('Failed to load credentials:', error);
            this.credentials = [];
        }
    },

    getCredential(providerId) {
        return this.credentials.find(c => c.provider === providerId);
    },

    render() {
        const container = document.getElementById('credentials-content');
        if (!container) return;

        container.innerHTML = `
            <div class="streamlit-section ds-fade-in">
                <h3 class="st-subheader" style="margin-top: 0;">
                    <i class="bi bi-key"></i> API Key Management
                </h3>
                <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                    Provide your own API keys for each LLM provider you want to evaluate against.
                    Keys are encrypted with AWS KMS before storage and only decrypted when running evaluations.
                </p>

                <div class="ds-card ds-mb-4">
                    ${this.providers.map(p => {
                        const cred = this.getCredential(p.id);
                        const hasCred = !!cred;
                        return `
                            <div class="credential-row" style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 0; border-bottom: 1px solid var(--border-color, #333);">
                                <div style="min-width: 160px;">
                                    <i class="bi ${p.icon}" style="margin-right: 0.5rem;"></i>
                                    <strong>${p.name}</strong>
                                </div>
                                <div style="flex: 1;">
                                    ${hasCred
                                        ? `<span class="ds-chip ds-chip--success" style="font-family: monospace;">${cred.key_alias}</span>
                                           <span style="color: var(--text-secondary); font-size: 0.8rem; margin-left: 0.5rem;">Updated ${cred.updated_at ? new Date(cred.updated_at).toLocaleDateString() : 'N/A'}</span>`
                                        : `<span style="color: var(--text-secondary);">No key configured</span>`
                                    }
                                </div>
                                <div style="display: flex; gap: 0.5rem;">
                                    <button class="ds-btn ds-btn--sm ds-btn--primary" data-action="edit" data-provider="${p.id}">
                                        <i class="bi bi-pencil"></i> ${hasCred ? 'Update' : 'Add'}
                                    </button>
                                    ${hasCred ? `
                                        <button class="ds-btn ds-btn--sm ds-btn--danger" data-action="delete" data-provider="${p.id}">
                                            <i class="bi bi-trash"></i>
                                        </button>
                                    ` : ''}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>

                <!-- Edit Modal (hidden by default) -->
                <div id="credential-edit-modal" style="display: none;">
                    <div class="ds-card ds-mb-4" style="border: 1px solid var(--accent-orange, #ff9900);">
                        <h4 id="credential-modal-title" class="st-section-header"></h4>
                        <div style="display: flex; gap: 0.75rem; align-items: flex-end;">
                            <div style="flex: 1;">
                                <label class="st-label">API Key</label>
                                <input type="password" id="credential-key-input" class="ds-input" style="width: 100%; font-family: monospace;" autocomplete="off">
                            </div>
                            <button class="ds-btn ds-btn--primary" id="credential-save-btn">
                                <i class="bi bi-shield-lock"></i> Save & Encrypt
                            </button>
                            <button class="ds-btn ds-btn--secondary" id="credential-cancel-btn">
                                Cancel
                            </button>
                        </div>
                        <input type="hidden" id="credential-provider-input">
                    </div>
                </div>

                <div class="st-info-box ds-mt-3">
                    <i class="bi bi-shield-check"></i>
                    <strong>Security:</strong> API keys are encrypted with AWS KMS and stored in DynamoDB.
                    They are only decrypted at evaluation runtime and injected into the evaluation container.
                    Keys are never logged or exposed in the UI after saving.
                </div>

                <!-- Email Notifications -->
                <h3 class="st-subheader ds-mt-4">
                    <i class="bi bi-envelope"></i> Email Notifications
                </h3>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Get notified via email when your evaluations complete or fail.
                </p>
                <div class="ds-card ds-mb-4">
                    <div style="display: flex; gap: 0.75rem; align-items: flex-end;">
                        <div style="flex: 1;">
                            <label class="st-label">Notification Email</label>
                            <input type="email" id="notification-email-input" class="ds-input" style="width: 100;"
                                   placeholder="your-alias@amazon.com"
                                   value="${this.notificationEmail || ''}">
                        </div>
                        <button class="ds-btn ds-btn--primary" id="save-notification-email-btn">
                            <i class="bi bi-bell"></i> ${this.notificationEmail ? 'Update' : 'Subscribe'}
                        </button>
                        ${this.notificationEmail ? `
                            <button class="ds-btn ds-btn--danger" id="remove-notification-email-btn">
                                <i class="bi bi-bell-slash"></i> Unsubscribe
                            </button>
                        ` : ''}
                    </div>
                    ${this.notificationEmail ? `
                        <div class="ds-alert ds-alert--success ds-mt-2">
                            <i class="bi bi-check-circle ds-alert__icon"></i>
                            <div class="ds-alert__content">
                                <div class="ds-alert__message">Subscribed: <strong>${this.notificationEmail}</strong></div>
                            </div>
                        </div>
                    ` : ''}
                    <div class="ds-alert ds-alert--info ds-mt-2">
                        <i class="bi bi-info-circle ds-alert__icon"></i>
                        <div class="ds-alert__content">
                            <div class="ds-alert__message">
                                After subscribing, you will receive a <strong>confirmation email from AWS SNS</strong>.
                                You must click the confirmation link in that email to activate notifications.
                                Only <strong>@amazon.com</strong> emails are accepted.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.bindEvents();
    },

    bindEvents() {
        // Edit/Add buttons
        document.querySelectorAll('[data-action="edit"]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const providerId = e.currentTarget.dataset.provider;
                const provider = this.providers.find(p => p.id === providerId);
                this.showEditModal(provider);
            });
        });

        // Delete buttons
        document.querySelectorAll('[data-action="delete"]').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const providerId = e.currentTarget.dataset.provider;
                const provider = this.providers.find(p => p.id === providerId);
                if (confirm(`Delete the API key for ${provider.name}?`)) {
                    await this.deleteCredential(providerId);
                }
            });
        });

        // Save button
        document.getElementById('credential-save-btn')?.addEventListener('click', async () => {
            await this.saveCredential();
        });

        // Cancel button
        document.getElementById('credential-cancel-btn')?.addEventListener('click', () => {
            this.hideEditModal();
        });

        // Enter key in input
        document.getElementById('credential-key-input')?.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                await this.saveCredential();
            }
        });

        // Save notification email
        document.getElementById('save-notification-email-btn')?.addEventListener('click', async () => {
            const email = document.getElementById('notification-email-input')?.value?.trim();
            if (!email) {
                App.showNotification('Error', 'Please enter an email address', 'error');
                return;
            }
            if (!email.endsWith('@amazon.com')) {
                App.showNotification('Error', 'Only @amazon.com email addresses are accepted', 'error');
                return;
            }
            try {
                App.showLoading(true);
                await API.post('/api/notifications/subscribe', { email });
                await this.loadCredentials();
                this.render();
                App.showNotification('Success', `Subscribed ${email}. Check your inbox for a confirmation email from AWS SNS.`);
            } catch (error) {
                App.showNotification('Error', `Failed to subscribe: ${error.message}`, 'error');
            } finally {
                App.showLoading(false);
            }
        });

        // Remove notification email
        document.getElementById('remove-notification-email-btn')?.addEventListener('click', async () => {
            if (confirm('Unsubscribe from email notifications?')) {
                try {
                    App.showLoading(true);
                    await API.post('/api/notifications/unsubscribe', {});
                    await this.loadCredentials();
                    this.render();
                    App.showNotification('Success', 'Unsubscribed from email notifications');
                } catch (error) {
                    App.showNotification('Error', `Failed to unsubscribe: ${error.message}`, 'error');
                } finally {
                    App.showLoading(false);
                }
            }
        });
    },

    showEditModal(provider) {
        const modal = document.getElementById('credential-edit-modal');
        const title = document.getElementById('credential-modal-title');
        const input = document.getElementById('credential-key-input');
        const hiddenProvider = document.getElementById('credential-provider-input');

        title.textContent = `Enter API Key for ${provider.name}`;
        input.placeholder = provider.placeholder;
        input.value = '';
        hiddenProvider.value = provider.id;
        modal.style.display = 'block';
        input.focus();
    },

    hideEditModal() {
        const modal = document.getElementById('credential-edit-modal');
        const input = document.getElementById('credential-key-input');
        modal.style.display = 'none';
        input.value = '';
    },

    async saveCredential() {
        const provider = document.getElementById('credential-provider-input').value;
        const apiKey = document.getElementById('credential-key-input').value.trim();

        if (!apiKey) {
            App.showNotification('Error', 'Please enter an API key', 'error');
            return;
        }

        try {
            App.showLoading(true);
            await API.saveCredential(provider, apiKey);
            this.hideEditModal();
            await this.loadCredentials();
            this.render();
            const providerName = this.providers.find(p => p.id === provider)?.name || provider;
            App.showNotification('Success', `API key for ${providerName} saved and encrypted`);
        } catch (error) {
            App.showNotification('Error', `Failed to save API key: ${error.message}`, 'error');
        } finally {
            App.showLoading(false);
        }
    },

    async deleteCredential(providerId) {
        try {
            App.showLoading(true);
            await API.deleteCredential(providerId);
            await this.loadCredentials();
            this.render();
            const providerName = this.providers.find(p => p.id === providerId)?.name || providerId;
            App.showNotification('Success', `API key for ${providerName} deleted`);
        } catch (error) {
            App.showNotification('Error', `Failed to delete API key: ${error.message}`, 'error');
        } finally {
            App.showLoading(false);
        }
    }
};

window.CredentialsComponent = CredentialsComponent;
