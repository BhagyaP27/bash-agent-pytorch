"use client";
import {useState, useRef, useEffect} from 'react';
import styles from './page.module.css';
 
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const EXAMPLES = [
  'make a c file named {your_file_name}.c',
  'list all files',
  'find all python files',
  'show last 20 lines of server.log',
  'create directory called src',
  'make {your_file_name}.py executable',
  'compress folder named myproject',
  'remove file data.csv',
];

export default function Home() {
  const [history, setHistory] = useState([
    { type: 'welcome', text: 'BashAgent v2.0 — Seq2Seq + Entity Extraction Pipeline' },
    { type: 'welcome', text: 'Type a description and get the bash command.' },
    { type: 'divider' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ connected: false, label: 'connecting...' });
 
  const outputRef = useRef(null);
  const inputRef = useRef(null);
 
  // Health check on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(d => {
        setStatus({
          connected: d.model_loaded,
          label: d.model_loaded
            ? `model ready · vocab ${d.input_vocab_size}`
            : 'model not loaded — run train.py',
        });
      })
      .catch(() => setStatus({ connected: false, label: 'API offline — run: uvicorn api:app --reload' }));
  }, []);
 
  // Auto-scroll
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [history]);
 
  const addLine = (entry) => {
    setHistory(prev => [...prev, entry]);
  };
 
  const handleSubmit = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setLoading(true);
 
    addLine({ type: 'input', text });
 
    try {
      const res = await fetch(`${API_URL}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, debug: true }),
      });
 
      if (!res.ok) {
        const err = await res.json();
        addLine({ type: 'error', text: err.detail || 'Translation failed' });
      } else {
        const data = await res.json();
        addLine({ type: 'output', text: data.command });
 
        // Show extracted entities if any
        if (data.debug_info?.entities) {
          const e = data.debug_info.entities;
          const parts = [];
          if (e.name) parts.push(`name=${e.name}`);
          if (e.extension) parts.push(`ext=.${e.extension}`);
          if (e.directory) parts.push(`dir=${e.directory}`);
          if (e.number) parts.push(`n=${e.number}`);
          if (parts.length) {
            addLine({ type: 'info', text: `↳ entities: ${parts.join(', ')}` });
          }
        }
      }
    } catch {
      addLine({ type: 'error', text: `Cannot reach API at ${API_URL}` });
      addLine({ type: 'info', text: 'Start backend: uvicorn api:app --host 0.0.0.0 --port 8000 --reload' });
      setStatus({ connected: false, label: 'disconnected' });
    }
 
    setLoading(false);
    inputRef.current?.focus();
  };
 
  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <div className={styles.terminal}>
 
          {/* Title bar */}
          <div className={styles.titleBar}>
            <span className={`${styles.dot} ${styles.dotRed}`} />
            <span className={`${styles.dot} ${styles.dotYellow}`} />
            <span className={`${styles.dot} ${styles.dotGreen}`} />
            <span className={styles.titleLabel}>bash-agent — natural language → shell</span>
          </div>
 
          {/* Output area */}
          <div className={styles.output} ref={outputRef}>
            {history.map((line, i) => (
              <HistoryLine key={i} line={line} />
            ))}
            {loading && (
              <div className={styles.line}>
                <span className={styles.info}>⟳ translating...</span>
              </div>
            )}
          </div>
 
          {/* Input */}
          <div className={styles.inputArea}>
            <span className={styles.prompt}>$&nbsp;</span>
            <input
              ref={inputRef}
              className={styles.inputField}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              placeholder="describe what you want to do..."
              autoFocus
            />
            <button
              className={styles.runBtn}
              onClick={handleSubmit}
              disabled={loading || !input.trim()}
            >
              ↵ run
            </button>
          </div>
 
          {/* Status bar */}
          <div className={styles.statusBar}>
            <span className={`${styles.statusDot} ${status.connected ? styles.statusOn : styles.statusOff}`} />
            <span className={styles.statusText}>{status.label}</span>
            <span className={`${styles.statusText} ${styles.statusRight}`}>{API_URL}</span>
          </div>
        </div>
 
        {/* Examples */}
        <div className={styles.examplesSection}>
          <p className={styles.examplesLabel}>try an example</p>
          <div className={styles.chips}>
            {EXAMPLES.map(ex => (
              <button
                key={ex}
                className={styles.chip}
                onClick={() => { setInput(ex); inputRef.current?.focus(); }}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
 
function HistoryLine({ line }) {
  const [copied, setCopied] = useState(false);
 
  if (line.type === 'divider') {
    return <div style={{ color: '#334155', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, marginBottom: 8 }}>─────────────────────────────────────────────────</div>;
  }
  if (line.type === 'welcome') {
    return <div style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, marginBottom: 2 }}>{line.text}</div>;
  }
  if (line.type === 'input') {
    return (
      <div style={{ display: 'flex', gap: 8, margin: '8px 0 2px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}>
        <span style={{ color: '#4ade80', flexShrink: 0 }}>$</span>
        <span style={{ color: '#e2e8f0' }}>{line.text}</span>
      </div>
    );
  }
  if (line.type === 'output') {
    const copy = () => {
      navigator.clipboard.writeText(line.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    };
    return (
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#1a2a1a', border: '0.5px solid #2a4a2a', borderRadius: 6, padding: '6px 10px', margin: '4px 0 4px 16px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}>
        <span style={{ color: '#86efac', letterSpacing: '0.5px' }}>{line.text}</span>
        <button onClick={copy} style={{ background: 'none', border: '0.5px solid #2a4a2a', color: copied ? '#a3e635' : '#4ade80', fontFamily: 'JetBrains Mono, monospace', fontSize: 10, padding: '2px 8px', borderRadius: 4, cursor: 'pointer' }}>
          {copied ? 'copied!' : 'copy'}
        </button>
      </div>
    );
  }
  if (line.type === 'info') {
    return <div style={{ color: '#475569', fontFamily: 'JetBrains Mono, monospace', fontSize: 12, marginLeft: 16, marginBottom: 2 }}>{line.text}</div>;
  }
  if (line.type === 'error') {
    return <div style={{ color: '#f87171', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, marginLeft: 16, marginBottom: 2 }}>✗ {line.text}</div>;
  }
  return null;
}
