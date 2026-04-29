// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { listProfiles } from '../api/profiles';
import { invoke } from '../api/gateway';
import type { Profile, Message, InvokeResponse } from '../types';
import MessageBubble from '../components/MessageBubble';

interface ChatMessage {
  role: string;
  text: string;
}

interface LoadTestStats {
  totalRequests: number;
  completed: number;
  succeeded: number;
  failed: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  errors: string[];
}

const INITIAL_STATS: LoadTestStats = {
  totalRequests: 0,
  completed: 0,
  succeeded: 0,
  failed: 0,
  totalInputTokens: 0,
  totalOutputTokens: 0,
  totalCost: 0,
  totalLatencyMs: 0,
  errors: [],
};

const InvokePage: React.FC = () => {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [apiMessages, setApiMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMetrics, setLastMetrics] = useState<InvokeResponse | null>(null);

  // Load test state
  const [showLoadTest, setShowLoadTest] = useState(false);
  const [loadTestPrompt, setLoadTestPrompt] = useState('Say "hello" in one word.');
  const [loadTestCount, setLoadTestCount] = useState(10);
  const [loadTestDurationMinutes, setLoadTestDurationMinutes] = useState(5);
  const [loadTestMode, setLoadTestMode] = useState<'count' | 'duration'>('count');
  const [loadTestRunning, setLoadTestRunning] = useState(false);
  const [loadTestStats, setLoadTestStats] = useState<LoadTestStats>(INITIAL_STATS);
  const loadTestAbortRef = useRef(false);
  const loadTestTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await listProfiles();
        setProfiles(data);
      } catch {
        // Silently handle
      }
    };
    load();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || !selectedProfileId) return;

    setError(null);
    const userMsg: Message = { role: 'user', content: [{ text }] };
    const updatedApiMessages = [...apiMessages, userMsg];
    setApiMessages(updatedApiMessages);
    setChatMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setSending(true);

    try {
      const response = await invoke(selectedProfileId, updatedApiMessages);
      const assistantText = response.output?.content?.[0]?.text ?? '';
      setChatMessages((prev) => [...prev, { role: 'assistant', text: assistantText }]);
      setApiMessages((prev) => [
        ...prev,
        { role: 'assistant', content: [{ text: assistantText }] },
      ]);
      setLastMetrics(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Invoke failed';
      setError(message);
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setChatMessages([]);
    setApiMessages([]);
    setLastMetrics(null);
    setError(null);
  };

  const selectedProfile = profiles.find((t) => t.tenant_id === selectedProfileId);

  const runLoadTest = useCallback(async () => {
    if (!selectedProfileId || !loadTestPrompt.trim()) return;
    loadTestAbortRef.current = false;
    setLoadTestRunning(true);
    setLoadTestStats(INITIAL_STATS);

    const messages: Message[] = [{ role: 'user', content: [{ text: loadTestPrompt.trim() }] }];

    if (loadTestMode === 'count') {
      // Fire N requests spread evenly across the duration
      const intervalMs = loadTestCount > 1
        ? (loadTestDurationMinutes * 60 * 1000) / (loadTestCount - 1)
        : 0;

      for (let i = 0; i < loadTestCount; i++) {
        if (loadTestAbortRef.current) break;

        setLoadTestStats((prev) => ({ ...prev, totalRequests: i + 1 }));
        try {
          const resp = await invoke(selectedProfileId, messages);
          setLoadTestStats((prev) => ({
            ...prev,
            completed: prev.completed + 1,
            succeeded: prev.succeeded + 1,
            totalInputTokens: prev.totalInputTokens + (resp.usage?.inputTokens ?? 0),
            totalOutputTokens: prev.totalOutputTokens + (resp.usage?.outputTokens ?? 0),
            totalCost: prev.totalCost + (resp.cost?.totalCost ?? 0),
            totalLatencyMs: prev.totalLatencyMs + (resp.latencyMs ?? 0),
          }));
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : 'Unknown error';
          setLoadTestStats((prev) => ({
            ...prev,
            completed: prev.completed + 1,
            failed: prev.failed + 1,
            errors: prev.errors.length < 5 ? [...prev.errors, msg] : prev.errors,
          }));
        }

        // Wait for next interval (skip wait on last request)
        if (i < loadTestCount - 1 && intervalMs > 0 && !loadTestAbortRef.current) {
          await new Promise<void>((resolve) => {
            loadTestTimerRef.current = setTimeout(resolve, intervalMs);
          });
        }
      }
    } else {
      // Duration mode: keep sending until time runs out
      const endTime = Date.now() + loadTestDurationMinutes * 60 * 1000;
      let count = 0;

      while (Date.now() < endTime && !loadTestAbortRef.current) {
        count++;
        setLoadTestStats((prev) => ({ ...prev, totalRequests: count }));
        try {
          const resp = await invoke(selectedProfileId, messages);
          setLoadTestStats((prev) => ({
            ...prev,
            completed: prev.completed + 1,
            succeeded: prev.succeeded + 1,
            totalInputTokens: prev.totalInputTokens + (resp.usage?.inputTokens ?? 0),
            totalOutputTokens: prev.totalOutputTokens + (resp.usage?.outputTokens ?? 0),
            totalCost: prev.totalCost + (resp.cost?.totalCost ?? 0),
            totalLatencyMs: prev.totalLatencyMs + (resp.latencyMs ?? 0),
          }));
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : 'Unknown error';
          setLoadTestStats((prev) => ({
            ...prev,
            completed: prev.completed + 1,
            failed: prev.failed + 1,
            errors: prev.errors.length < 5 ? [...prev.errors, msg] : prev.errors,
          }));
        }
      }
    }

    setLoadTestRunning(false);
  }, [selectedProfileId, loadTestPrompt, loadTestCount, loadTestDurationMinutes, loadTestMode]);

  const stopLoadTest = useCallback(() => {
    loadTestAbortRef.current = true;
    if (loadTestTimerRef.current) {
      clearTimeout(loadTestTimerRef.current);
      loadTestTimerRef.current = null;
    }
  }, []);

  const inputStyle: React.CSSProperties = {
    padding: '8px 10px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
  };

  const statBoxStyle: React.CSSProperties = {
    padding: '8px 12px',
    backgroundColor: '#fff',
    borderRadius: '6px',
    border: '1px solid #e5e7eb',
    textAlign: 'center' as const,
    minWidth: '100px',
  };

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '24px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 100px)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h1 style={{ margin: 0 }}>Invoke Model</h1>
        <button
          onClick={handleClear}
          style={{
            padding: '6px 14px',
            backgroundColor: '#fff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '13px',
          }}
        >
          Clear Chat
        </button>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ fontWeight: 600, fontSize: '14px', marginRight: '8px' }}>Profile:</label>
        <select
          value={selectedProfileId}
          onChange={(e) => {
            setSelectedProfileId(e.target.value);
            handleClear();
          }}
          style={{
            padding: '8px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            fontSize: '14px',
            minWidth: '250px',
          }}
        >
          <option value="">Select a profile...</option>
          {profiles.map((t) => (
            <option key={t.tenant_id} value={t.tenant_id}>
              {t.tenant_name} ({t.status})
            </option>
          ))}
        </select>
        {selectedProfile && (
          <span style={{ marginLeft: '12px', fontSize: '13px', color: '#6b7280' }}>
            Model: {selectedProfile.model_id}
          </span>
        )}
      </div>

      {error && (
        <div
          style={{
            padding: '10px 14px',
            backgroundColor: '#fef2f2',
            color: '#dc2626',
            borderRadius: '6px',
            marginBottom: '12px',
            fontSize: '14px',
          }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          flex: 1,
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          padding: '16px',
          overflowY: 'auto',
          backgroundColor: '#fff',
          marginBottom: '12px',
          minHeight: '200px',
        }}
      >
        {chatMessages.length === 0 ? (
          <p style={{ color: '#9ca3af', textAlign: 'center', marginTop: '40px' }}>
            {selectedProfileId ? 'Send a message to start the conversation.' : 'Select a profile to begin.'}
          </p>
        ) : (
          chatMessages.map((msg, idx) => (
            <MessageBubble key={idx} role={msg.role} text={msg.text} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {lastMetrics && (
        <div
          style={{
            display: 'flex',
            gap: '20px',
            padding: '10px 14px',
            backgroundColor: '#f9fafb',
            borderRadius: '6px',
            marginBottom: '12px',
            fontSize: '13px',
            color: '#4b5563',
            flexWrap: 'wrap',
          }}
        >
          <span>Input Tokens: <strong>{lastMetrics.usage.inputTokens}</strong></span>
          <span>Output Tokens: <strong>{lastMetrics.usage.outputTokens}</strong></span>
          <span>Cost: <strong>${lastMetrics.cost.totalCost.toFixed(6)}</strong></span>
          <span>Latency: <strong>{lastMetrics.latencyMs}ms</strong></span>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px' }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={selectedProfileId ? 'Type a message...' : 'Select a profile first'}
          disabled={!selectedProfileId || sending}
          rows={2}
          style={{
            flex: 1,
            padding: '10px 12px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            fontSize: '14px',
            resize: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!selectedProfileId || !input.trim() || sending}
          style={{
            padding: '10px 20px',
            backgroundColor: sending ? '#93c5fd' : '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            cursor: !selectedProfileId || !input.trim() || sending ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            alignSelf: 'flex-end',
          }}
        >
          {sending ? 'Sending...' : 'Send'}
        </button>
      </div>

      {/* Load Test Panel */}
      <div style={{ marginTop: '24px', borderTop: '1px solid #e5e7eb', paddingTop: '16px' }}>
        <button
          onClick={() => setShowLoadTest(!showLoadTest)}
          style={{
            padding: '8px 16px',
            backgroundColor: showLoadTest ? '#f3f4f6' : '#fff',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 600,
          }}
        >
          {showLoadTest ? 'Hide' : 'Show'} Load Test
        </button>

        {showLoadTest && (
          <div style={{ marginTop: '12px', padding: '16px', backgroundColor: '#f9fafb', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '16px' }}>Load Test</h3>

            {/* Prompt */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontWeight: 500, fontSize: '13px', marginBottom: '4px' }}>Prompt</label>
              <input
                type="text"
                value={loadTestPrompt}
                onChange={(e) => setLoadTestPrompt(e.target.value)}
                disabled={loadTestRunning}
                style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }}
              />
            </div>

            {/* Mode selector */}
            <div style={{ display: 'flex', gap: '16px', marginBottom: '12px', alignItems: 'flex-end' }}>
              <div>
                <label style={{ display: 'block', fontWeight: 500, fontSize: '13px', marginBottom: '4px' }}>Mode</label>
                <select
                  value={loadTestMode}
                  onChange={(e) => setLoadTestMode(e.target.value as 'count' | 'duration')}
                  disabled={loadTestRunning}
                  style={inputStyle}
                >
                  <option value="count">Fixed count over duration</option>
                  <option value="duration">Continuous for duration</option>
                </select>
              </div>

              {loadTestMode === 'count' && (
                <div>
                  <label style={{ display: 'block', fontWeight: 500, fontSize: '13px', marginBottom: '4px' }}>Invocations</label>
                  <input
                    type="number"
                    min={1}
                    max={1000}
                    value={loadTestCount}
                    onChange={(e) => setLoadTestCount(Number(e.target.value))}
                    disabled={loadTestRunning}
                    style={{ ...inputStyle, width: '80px' }}
                  />
                </div>
              )}

              <div>
                <label style={{ display: 'block', fontWeight: 500, fontSize: '13px', marginBottom: '4px' }}>Duration (minutes)</label>
                <input
                  type="number"
                  min={1}
                  max={1440}
                  value={loadTestDurationMinutes}
                  onChange={(e) => setLoadTestDurationMinutes(Number(e.target.value))}
                  disabled={loadTestRunning}
                  style={{ ...inputStyle, width: '80px' }}
                />
              </div>

              {!loadTestRunning ? (
                <button
                  onClick={runLoadTest}
                  disabled={!selectedProfileId || !loadTestPrompt.trim()}
                  style={{
                    padding: '8px 20px',
                    backgroundColor: !selectedProfileId ? '#9ca3af' : '#16a34a',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: !selectedProfileId ? 'not-allowed' : 'pointer',
                    fontWeight: 600,
                    fontSize: '14px',
                  }}
                >
                  Start
                </button>
              ) : (
                <button
                  onClick={stopLoadTest}
                  style={{
                    padding: '8px 20px',
                    backgroundColor: '#dc2626',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: 600,
                    fontSize: '14px',
                  }}
                >
                  Stop
                </button>
              )}
            </div>

            {loadTestMode === 'count' && (
              <p style={{ fontSize: '12px', color: '#6b7280', margin: '0 0 12px 0' }}>
                {loadTestCount} request{loadTestCount !== 1 ? 's' : ''} spread evenly over {loadTestDurationMinutes} minute{loadTestDurationMinutes !== 1 ? 's' : ''} (~1 request every {loadTestCount > 1 ? ((loadTestDurationMinutes * 60) / (loadTestCount - 1)).toFixed(1) : '0'}s)
              </p>
            )}

            {/* Stats */}
            {(loadTestRunning || loadTestStats.completed > 0) && (
              <div>
                {/* Progress bar */}
                {loadTestMode === 'count' && (
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                      <span>Progress</span>
                      <span>{loadTestStats.completed} / {loadTestStats.totalRequests}{loadTestRunning ? ` (${loadTestStats.totalRequests < loadTestCount ? 'queued' : 'sending'})` : ' done'}</span>
                    </div>
                    <div style={{ height: '6px', backgroundColor: '#e5e7eb', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${(loadTestStats.completed / loadTestCount) * 100}%`,
                        backgroundColor: loadTestStats.failed > 0 ? '#f59e0b' : '#16a34a',
                        borderRadius: '3px',
                        transition: 'width 0.3s ease',
                      }} />
                    </div>
                  </div>
                )}

                {/* Stat cards */}
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#16a34a' }}>{loadTestStats.succeeded}</div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>Succeeded</div>
                  </div>
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: loadTestStats.failed > 0 ? '#dc2626' : '#9ca3af' }}>{loadTestStats.failed}</div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>Failed</div>
                  </div>
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#2563eb' }}>{loadTestStats.totalInputTokens.toLocaleString()}</div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>Input Tokens</div>
                  </div>
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#2563eb' }}>{loadTestStats.totalOutputTokens.toLocaleString()}</div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>Output Tokens</div>
                  </div>
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#7c3aed' }}>${loadTestStats.totalCost.toFixed(4)}</div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>Total Cost</div>
                  </div>
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#0891b2' }}>
                      {loadTestStats.succeeded > 0 ? (loadTestStats.totalLatencyMs / loadTestStats.succeeded).toFixed(0) : '—'}ms
                    </div>
                    <div style={{ fontSize: '11px', color: '#6b7280' }}>Avg Latency</div>
                  </div>
                </div>

                {/* Errors */}
                {loadTestStats.errors.length > 0 && (
                  <div style={{ marginTop: '10px', padding: '8px 12px', backgroundColor: '#fef2f2', borderRadius: '6px', fontSize: '12px', color: '#dc2626' }}>
                    <strong>Errors:</strong>
                    {loadTestStats.errors.map((e, i) => (
                      <div key={i} style={{ marginTop: '2px' }}>{e}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default InvokePage;
