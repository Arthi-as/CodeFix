import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Editor, { loader } from '@monaco-editor/react';
import axios from 'axios';
import './App.css';

// Serve Monaco from this origin instead of a CDN. The files are copied into
// public/vs by scripts/copy-monaco.js, so the editor still mounts with no
// internet access.
const PUBLIC_URL = process.env.PUBLIC_URL || '';
loader.config({ paths: { vs: `${PUBLIC_URL}/vs` } });

const API_BASE = process.env.REACT_APP_API_URL || '';
const VALIDATION_FAILED_NOTICE = 'Suggested fix — validation unsuccessful.';

const DEFAULT_CODE = `age = input("Enter your age: ")

if age > 18:
    print("Adult")
else:
    print("Minor")`;

const SEVERITY_COLORS = {
  High: {
    bg: 'var(--error-bg)',
    border: 'var(--error-border)',
    text: 'var(--error-text)',
    ruler: '#f85149',
    icon: '●',
  },
  Medium: {
    bg: 'var(--warning-bg)',
    border: 'var(--warning-border)',
    text: 'var(--warning-text)',
    ruler: '#d29922',
    icon: '●',
  },
  Low: {
    bg: 'var(--info-bg)',
    border: 'var(--info-border)',
    text: 'var(--info-text)',
    ruler: '#58a6ff',
    icon: '●',
  },
};

const DEFAULT_SEVERITY = SEVERITY_COLORS.Medium;

function severityStyle(severity) {
  return SEVERITY_COLORS[severity] || DEFAULT_SEVERITY;
}

function App() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [analysis, setAnalysis] = useState(null);
  // The exact source that produced `analysis`. Kept so the panel can tell a
  // result for what is in the editor from a result for something the user has
  // since replaced.
  const [analyzedCode, setAnalyzedCode] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [editorRef, setEditorRef] = useState(null);
  const [monaco, setMonaco] = useState(null);
  const decorationsRef = useRef([]);
  const latestCodeRef = useRef(code);

  latestCodeRef.current = code;

  const isStale = Boolean(analysis) && code !== analyzedCode;

  /* ---------------------------------------------------------------- */
  /* Error-line decorations                                            */
  /* ---------------------------------------------------------------- */

  // Decorations must be driven by `monaco`, not `window.monaco`: the library
  // does not expose the API globally, and reading it before the editor mounts
  // throws. Ids live in a ref so the effect does not re-run on its own writes.
  useEffect(() => {
    const editor = editorRef;
    if (!editor || !monaco) return undefined;

    // Line numbers only mean something for the source that was analyzed, so a
    // stale result must not keep marking lines in the edited file.
    const issues =
      !isStale && analysis && analysis.all_issues && analysis.all_issues.length
        ? analysis.all_issues
        : !isStale && analysis && analysis.has_error && analysis.line
          ? [{ line: analysis.line, severity: analysis.severity }]
          : [];

    decorationsRef.current = editor.deltaDecorations(decorationsRef.current, []);

    if (!issues.length) return undefined;

    const created = issues.map((issue) => {
      const palette = severityStyle(issue.severity);
      const line = Math.min(Math.max(issue.line, 1), editor.getModel()?.getLineCount() || 1);
      return {
        range: new monaco.Range(line, 1, line, 1),
        options: {
          isWholeLine: true,
          className: 'error-line-highlight',
          overviewRuler: {
            color: palette.ruler,
            position: monaco.editor.OverviewRulerLane.Full,
          },
          marginClassName: `issue-margin issue-margin-${String(issue.severity || 'medium').toLowerCase()}`,
        },
      };
    });

    decorationsRef.current = editor.deltaDecorations([], created);
    return undefined;
  }, [editorRef, monaco, analysis, isStale]);

  // Drop decorations when the editor unmounts.
  useEffect(
    () => () => {
      if (editorRef) {
        editorRef.deltaDecorations(decorationsRef.current, []);
      }
    },
    [editorRef],
  );

  /* ---------------------------------------------------------------- */
  /* Actions                                                           */
  /* ---------------------------------------------------------------- */

  const handleAnalyze = useCallback(async (targetCode) => {
    const source = targetCode !== undefined ? targetCode : latestCodeRef.current;

    if (!source || !source.trim()) {
      setError('Enter some Python code to analyze.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/analyze`, { code: source }, { timeout: 180000 });
      setAnalysis(response.data);
      setAnalyzedCode(source);
    } catch (err) {
      console.error('Analysis error:', err);
      if (axios.isCancel(err) || err.code === 'ERR_CANCELED') return;
      const detail = err.response?.data?.detail;
      if (err.response) {
        setError(detail || `Analysis failed (HTTP ${err.response.status}).`);
      } else if (err.code === 'ECONNABORTED') {
        setError('The analysis timed out. The AI model can be slow on first use — try again.');
      } else {
        setError('Cannot reach the backend. Is it running on port 8000?');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Applying a fix re-runs the analysis so the panel reflects reality instead
  // of a stale result for code that no longer exists in the editor.
  const handleApplyFix = useCallback(async () => {
    if (!analysis || !analysis.validated || !analysis.corrected_code) return;
    // Never write a correction computed for older source over newer source.
    if (code !== analyzedCode) return;
    const fixed = analysis.corrected_code;
    setCode(fixed);
    setAnalysis(null);
    setAnalyzedCode(null);
    await handleAnalyze(fixed);
  }, [analysis, handleAnalyze, analyzedCode, code]);

  const handleClear = useCallback(() => {
    setCode('');
    setAnalysis(null);
    setAnalyzedCode(null);
    setError(null);
  }, []);

  const handleReset = useCallback(() => {
    setCode(DEFAULT_CODE);
    setAnalysis(null);
    setAnalyzedCode(null);
    setError(null);
  }, []);

  const handleCodeChange = useCallback((value) => {
    setCode(value || '');
  }, []);

  // Ctrl/Cmd+Enter runs the analysis.
  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (!loading) handleAnalyze();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [loading, handleAnalyze]);

  /* ---------------------------------------------------------------- */
  /* Derived view data                                                 */
  /* ---------------------------------------------------------------- */

  const palette = analysis?.has_error ? severityStyle(analysis.severity) : DEFAULT_SEVERITY;

  // A correction is only offered while it still corresponds to what the editor
  // holds; once the code is edited the old snippet would replace the new code.
  const hasProposedCode = Boolean(
    analysis && !isStale && analysis.corrected_code && analysis.corrected_code !== code,
  );

  // Distinguish "we could not fix this" from "the fix failed validation", and
  // both from "the code changed since this result".
  const fixState = useMemo(() => {
    if (!analysis || !analysis.has_error) return 'none';
    if (isStale) return 'stale';
    if (analysis.validated && hasProposedCode) return 'validated';
    if (hasProposedCode) return 'unvalidated';
    return 'unavailable';
  }, [analysis, hasProposedCode, isStale]);

  const otherIssues = (analysis?.all_issues || []).filter(
    (issue) => !(analysis.line === issue.line && analysis.error_type === issue.error_type),
  );

  const model = analysis?.ai;

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="logo">CodeFix AI</h1>
          <span className="tagline">AI-powered Python debugging assistant</span>
        </div>
        <div className="header-right">
          {model && (
            <span
              className={`model-chip model-${model.model_loaded ? 'ready' : 'off'}`}
              title={model.note}
            >
              {model.model}
              {model.model_loaded ? ' · ready' : ' · not loaded'}
            </span>
          )}
          <button className="btn btn-secondary" onClick={handleReset} disabled={loading}>
            Reset
          </button>
          <button className="btn btn-secondary" onClick={handleClear} disabled={loading}>
            Clear
          </button>
          <button
            className="btn btn-primary"
            onClick={() => handleAnalyze()}
            disabled={loading || !code.trim()}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Analyzing…
              </>
            ) : (
              'Analyze Code'
            )}
          </button>
        </div>
      </header>

      {error && (
        <div className="toast error" role="alert">
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      <main className="main">
        <section className="editor-panel" aria-label="Code Editor">
          <div className="panel-header">
            <h2>Code Editor</h2>
            <div className="editor-toolbar">
              <span className="language-badge">Python</span>
              <span className="kbd-hint" title="Analyze">
                Ctrl+Enter
              </span>
            </div>
          </div>
          <div className="editor-container">
            <Editor
              height="100%"
              language="python"
              value={code}
              onChange={handleCodeChange}
              beforeMount={setMonaco}
              onMount={(editor) => setEditorRef(editor)}
              loading={<div className="loading-state">Loading editor…</div>}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                insertSpaces: true,
                theme: 'vs-dark',
                wordWrap: 'on',
                bracketPairColorization: { enabled: true },
                guides: { bracketPairs: true },
                renderLineHighlight: 'all',
                matchBrackets: 'always',
                autoClosingBrackets: 'always',
                formatOnPaste: true,
              }}
            />
          </div>
        </section>

        <aside className="analysis-panel" aria-label="Analysis Results">
          <div className="panel-header">
            <h2>Analysis</h2>
          </div>

          <div className="analysis-content" aria-live="polite" aria-busy={loading}>
            {loading && (
              <div className="loading-state">
                <div className="spinner-large" />
                <p>Analyzing your code…</p>
                <p className="loading-hint">Static analysis &amp; AI model</p>
              </div>
            )}

            {!loading && !analysis && (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <p>
                  Enter Python code and click <strong>Analyze Code</strong>
                </p>
                <p className="empty-hint">
                  CodeFix AI detects errors, explains them, and suggests validated fixes
                </p>
              </div>
            )}

            {analysis && !analysis.has_error && (
              <div className="success-state">
                <div className="success-icon">✓</div>
                <h3>No Issues Found</h3>
                <p>
                  The code parses and Pylint reported no errors or warnings. Static analysis
                  cannot prove the logic is correct at runtime.
                </p>
                {model && (
                  <p className="attribution">
                    Model: {model.model} · {model.model_loaded ? 'loaded' : 'not loaded'}
                  </p>
                )}
              </div>
            )}

            {analysis && analysis.has_error && (
              <div className="error-analysis">
                <div className="error-header" style={{ background: palette.bg, borderColor: palette.border }}>
                  <div className="error-icon" style={{ color: palette.text }}>
                    {palette.icon}
                  </div>
                  <div className="error-title">
                    <h3>Bug Found</h3>
                    <span className="error-line">Line {analysis.line}</span>
                  </div>
                  <div
                    className={`severity-badge severity-${String(analysis.severity || '').toLowerCase()}`}
                    style={{ background: palette.bg, borderColor: palette.border, color: palette.text }}
                  >
                    {analysis.severity}
                  </div>
                </div>

                <div className="error-meta">
                  <div className="meta-item">
                    <span className="meta-label">Type</span>
                    <span className="meta-value">{analysis.error_type}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Category</span>
                    <span className="meta-value">{analysis.category || 'Static Analysis'}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Found by</span>
                    <span className="meta-value">{analysis.detected_by}</span>
                  </div>
                </div>

                <div className="analysis-section">
                  <h4>Message</h4>
                  <div className="explanation">{analysis.message}</div>
                </div>

                {analysis.explanation && (
                  <div className="analysis-section">
                    <h4>{analysis.explanation_title || 'Why?'}</h4>
                    <div className="explanation">{analysis.explanation}</div>
                  </div>
                )}

                {analysis.suggestion && (
                  <div className="analysis-section">
                    <h4>Suggested Fix</h4>
                    <div className="suggestion">{analysis.suggestion}</div>
                  </div>
                )}

                {hasProposedCode && (
                  <div className="analysis-section">
                    <h4>Corrected Code</h4>
                    {analysis.fix_summary && (
                      <div className="fix-summary">
                        {analysis.fix_summary}
                        {model?.fix_source && (
                          <span className="source-tag">via {model.fix_source}</span>
                        )}
                      </div>
                    )}
                    <div className="corrected-code">
                      <pre>
                        <code>{analysis.corrected_code}</code>
                      </pre>
                    </div>
                  </div>
                )}

                {analysis.diff && analysis.diff.length > 0 && (
                  <div className="analysis-section">
                    <h4>Changes</h4>
                    <div className="diff-view">
                      {analysis.diff.map((entry, index) => (
                        <div key={`${entry.line}-${index}`} className={`diff-row diff-${entry.kind}`}>
                          <span className="diff-line">{entry.line}</span>
                          <span className="diff-marker">
                            {entry.kind === 'added' ? '+' : entry.kind === 'removed' ? '−' : ' '}
                          </span>
                          <code>{entry.text}</code>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="validation-status">
                  {fixState === 'validated' && (
                    <div className="validation-badge validated">
                      <span className="validation-icon">✓</span>
                      <span>Fix validated successfully</span>
                    </div>
                  )}
                  {fixState === 'unvalidated' && (
                    <div className="validation-badge not-validated">
                      <span className="validation-icon">⚠</span>
                      <span>{VALIDATION_FAILED_NOTICE}</span>
                    </div>
                  )}
                  {fixState === 'unavailable' && (
                    <div className="validation-badge not-validated">
                      <span className="validation-icon">⚠</span>
                      <span>No automatic correction is safe to offer here.</span>
                    </div>
                  )}
                  {fixState === 'stale' && (
                    <div className="validation-badge not-validated">
                      <span className="validation-icon">⚠</span>
                      <span>
                        The code changed after this analysis. Run Analyze Code again to get a
                        result for what the editor holds now.
                      </span>
                    </div>
                  )}
                  {analysis.validation_details?.reason && fixState !== 'validated' && (
                    <div className="validation-warnings">
                      <div className="warning-item">{analysis.validation_details.reason}</div>
                    </div>
                  )}
                  {(analysis.validation_details?.warnings || []).map((warning, index) => (
                    <div key={index} className="validation-warnings">
                      <div className="warning-item">⚠ {warning}</div>
                    </div>
                  ))}
                </div>

                {fixState === 'validated' && (
                  <button className="btn btn-success btn-full" onClick={handleApplyFix} disabled={loading}>
                    Apply Fix
                  </button>
                )}

                {fixState === 'unvalidated' && (
                  <div className="validation-note">
                    This suggestion was not validated. Review the corrected code above and apply it
                    manually if it makes sense.
                  </div>
                )}

                {fixState === 'unavailable' && (
                  <div className="validation-note">
                    Fixing this automatically would mean guessing. Use the explanation above to
                    decide what the value should be.
                  </div>
                )}

                {fixState === 'stale' && (
                  <div className="validation-note">
                    The correction below belongs to the code you analyzed, not to the code now in
                    the editor, so it is hidden rather than offered for this file.
                  </div>
                )}

                {otherIssues.length > 0 && (
                  <div className="analysis-section">
                    <h4>Other findings ({otherIssues.length})</h4>
                    <ul className="issue-list">
                      {otherIssues.map((issue, index) => (
                        <li key={`${issue.line}-${issue.error_type}-${index}`}>
                          <button
                            type="button"
                            className="issue-row"
                            onClick={() => {
                              if (!editorRef) return;
                              editorRef.revealLineInCenter(issue.line);
                              editorRef.setPosition({ lineNumber: issue.line, column: 1 });
                              editorRef.focus();
                            }}
                          >
                            <span className="issue-line-no">L{issue.line}</span>
                            <span className="issue-type">{issue.error_type}</span>
                            <span className="issue-category">{issue.category}</span>
                            <span className={`issue-sev issue-sev-${String(issue.severity || '').toLowerCase()}`}>
                              {issue.severity}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {model && (
                  <p className="attribution">
                    {model.explanation_source} · analysis took {analysis.analysis_ms} ms
                  </p>
                )}
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;