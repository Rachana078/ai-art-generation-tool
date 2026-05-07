import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Image as ImageIcon, Download, History, AlertCircle, RefreshCw, Sparkles, Loader2, Trash2 } from 'lucide-react';
import './studio.css';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const STYLES = [
  { id: 'watercolor',     name: 'Watercolor',  icon: '🎨', gradient: 'linear-gradient(135deg,#3b82f6,#06b6d4)' },
  { id: 'cyberpunk',      name: 'Cyberpunk',   icon: '⚡', gradient: 'linear-gradient(135deg,#eab308,#ec4899)' },
  { id: 'charcoal-sketch',name: 'Charcoal',    icon: '✏️', gradient: 'linear-gradient(135deg,#64748b,#334155)' },
  { id: 'pop-art',        name: 'Pop Art',     icon: '🎭', gradient: 'linear-gradient(135deg,#f472b6,#f97316)' },
  { id: 'renaissance',    name: 'Classic Oil', icon: '🖼️', gradient: 'linear-gradient(135deg,#f59e0b,#92400e)' },
  { id: 'pixel-art',      name: 'Pixel Art',  icon: '👾', gradient: 'linear-gradient(135deg,#22c55e,#06b6d4)' },
  { id: 'ukiyo-e',        name: 'Ukiyo-e',    icon: '🌊', gradient: 'linear-gradient(135deg,#ef4444,#f97316)' },
  { id: 'surrealism',     name: 'Surrealism',  icon: '✦', gradient: 'linear-gradient(135deg,#a855f7,#6366f1)' },
  { id: 'custom',         name: 'Custom',      icon: '🎲', gradient: 'linear-gradient(135deg,#8b5cf6,#c084fc)' },
];

const MOODS = ['Vibrant', 'Ethereal', 'Melancholic', 'Dramatic', 'Serene', 'Chaotic', 'Custom...'];

const PALETTES = [
  { id: 'colorful',   name: 'Colorful', dot: 'linear-gradient(135deg,#ef4444,#3b82f6)' },
  { id: 'warm',       name: 'Warm',     dot: 'linear-gradient(135deg,#f97316,#ef4444)' },
  { id: 'cool',       name: 'Cool',     dot: 'linear-gradient(135deg,#3b82f6,#06b6d4)' },
  { id: 'monochrome', name: 'Mono',     dot: 'linear-gradient(135deg,#94a3b8,#475569)' },
  { id: 'neon',       name: 'Neon',     dot: 'linear-gradient(135deg,#22c55e,#ec4899)'  },
  { id: 'earthy',     name: 'Earthy',   dot: 'linear-gradient(135deg,#92400e,#65a30d)'  },
];

const GALLERY_PAGE = 20;

function StarRating({ itemId, initialRating }) {
  const [rating, setRating] = useState(initialRating ?? 0);
  const [hover, setHover] = useState(0);

  const handleRate = async (value) => {
    try {
      await axios.patch(`${API_URL}/api/generations/${itemId}/rating`, { rating: value });
      setRating(value);
    } catch (err) {
      console.error('Rating failed', err);
    }
  };

  return (
    <div className="star-row">
      {[1,2,3,4,5].map(star => (
        <button
          key={star}
          className="star-btn"
          style={{ color: star <= (hover || rating) ? '#facc15' : 'rgba(128,128,128,0.4)' }}
          onClick={() => handleRate(star)}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
        >★</button>
      ))}
    </div>
  );
}


function UsernameModal({ onSave }) {
  const [name, setName] = useState('');
  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="logo-mark" style={{ margin: '0 auto 16px' }}>✦</div>
        <h2 className="modal-title">Welcome to AI Art Studio</h2>
        <p className="modal-sub">Enter a name to tag your generations</p>
        <input
          className="modal-input"
          type="text"
          placeholder="e.g. rachana"
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && name.trim() && onSave(name.trim())}
          autoFocus
        />
        <button
          className="generate-btn"
          style={{ marginTop: 12 }}
          disabled={!name.trim()}
          onClick={() => onSave(name.trim())}
        >
          <Sparkles size={14} /> Start Creating
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [prompt, setPrompt]               = useState('');
  const [selectedStyle, setSelectedStyle] = useState(null);
  const [selectedMood, setSelectedMood]   = useState('Vibrant');
  const [selectedPalette, setSelectedPalette] = useState('colorful');
  const [customStyle, setCustomStyle]     = useState('');
  const [customMood, setCustomMood]       = useState('');
  const [image, setImage]                 = useState(null);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);
  const [gallery, setGallery]             = useState([]);
  const [galleryTotal, setGalleryTotal]   = useState(0);
  const [galleryOffset, setGalleryOffset] = useState(0);
  const [loadingMore, setLoadingMore]     = useState(false);
  const [refinedPrompt, setRefinedPrompt] = useState(null);
  const [promptSource, setPromptSource]   = useState(null);
  const [username, setUsername]           = useState(() => localStorage.getItem('studio_username') || '');
  const abortRef = useRef(null);

  const fetchGallery = async (offset = 0, append = false) => {
    try {
      const res = await axios.get(`${API_URL}/gallery?limit=${GALLERY_PAGE}&offset=${offset}`);
      setGalleryTotal(res.data.total);
      setGallery(prev => append ? [...prev, ...res.data.items] : res.data.items);
      setGalleryOffset(offset);
    } catch (err) {
      console.error('Gallery fetch failed', err);
    }
  };

  useEffect(() => { fetchGallery(); }, [image]);

  const handleGenerate = async () => {
    if (!prompt) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const genId = controller; // identity check — only the latest request updates state
    setLoading(true);
    setImage(null);
    setError(null);
    setRefinedPrompt(null);
    setPromptSource(null);
    try {
      const payload = {
        prompt,
        style:          selectedStyle === 'custom' ? customStyle : selectedStyle,
        mood:           selectedMood  === 'Custom...' ? customMood : selectedMood.toLowerCase(),
        palette:        selectedPalette,
        creator: username || null,
      };
      const res = await axios.post(`${API_URL}/generate`, payload, { signal: controller.signal });
      if (abortRef.current !== genId) return; // a newer request took over
      if (res.data?.image_url) {
        setImage(`${res.data.image_url}?t=${Date.now()}`);
        setRefinedPrompt(res.data.refined_prompt ?? null);
        setPromptSource(res.data.prompt_source ?? null);
      }
    } catch (err) {
      if (axios.isCancel(err) || err?.name === 'CanceledError') return;
      if (abortRef.current !== genId) return;
      console.error('API Error:', err);
      setError(err?.response?.data?.detail ?? err?.message ?? 'Unknown error');
    } finally {
      if (abortRef.current === genId) setLoading(false);
    }
  };

  const handleDelete = async (id, filename) => {
    try {
      await axios.delete(`${API_URL}/api/generations/${id}`);
      setGallery(prev => prev.filter(item => item.filename !== filename));
      setGalleryTotal(prev => prev - 1);
    } catch (err) {
      console.error('Delete failed', err);
    }
  };

  const handleLoadMore = async () => {
    setLoadingMore(true);
    await fetchGallery(galleryOffset + GALLERY_PAGE, true);
    setLoadingMore(false);
  };

  const handleSaveUsername = (name) => {
    localStorage.setItem('studio_username', name);
    setUsername(name);
  };

  const [theme, setTheme] = useState(() => localStorage.getItem('studio_theme') || 'dark');
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('studio_theme', theme);
  }, [theme]);

  const activeStyleName = STYLES.find(s => s.id === selectedStyle)?.name;

  return (
    <>
      <div className="bg-layer" />

      {!username && <UsernameModal onSave={handleSaveUsername} />}

      {/* ── Header ── */}
      <header className="studio-header">
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <div className="logo-mark">✦</div>
          <span className="logo-text">AI ART STUDIO</span>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          {username && <span className="creator-badge">{username}</span>}
          <div className="creations-pill">
            <History size={12} /> {galleryTotal} creations
          </div>
          <button className={`theme-toggle ${theme}`} onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle theme" aria-label="Toggle theme">
            <span className="theme-toggle-icon">{theme === 'dark' ? '🌙' : '☀️'}</span>
            <span className="theme-toggle-knob" />
          </button>
        </div>
      </header>

      {/* ── Two-col layout ── */}
      <div className="studio-layout">

        {/* ── Left panel ── */}
        <aside className="left-panel">
          <div className="controls-scroll">

            {/* 1. Vision */}
            <div className="section">
              <div className="section-label">Vision</div>
              <textarea
                className="vision-textarea"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="Describe your vision..."
              />
            </div>

            {/* 2. Style */}
            <div className="section">
              <div className="section-label">Style</div>
              <div className="style-grid">
                {STYLES.map(s => (
                  <button
                    key={s.id}
                    className={`style-card${selectedStyle === s.id ? ' active' : ''}`}
                    onClick={() => setSelectedStyle(s.id)}
                  >
                    <div className="style-card-overlay" style={{ background: s.gradient }} />
                    <span className="style-card-icon">{s.icon}</span>
                    <span className="style-card-name">{s.name}</span>
                  </button>
                ))}
              </div>
              {selectedStyle === 'custom' && (
                <input
                  className="custom-input"
                  type="text"
                  placeholder="e.g. Cyberpunk Noir..."
                  value={customStyle}
                  onChange={e => setCustomStyle(e.target.value)}
                />
              )}
            </div>

            {/* 3. Mood */}
            <div className="section">
              <div className="section-label">Mood</div>
              <div className="chip-row">
                {MOODS.map(m => (
                  <button
                    key={m}
                    className={`chip${selectedMood === m ? ' active' : ''}`}
                    onClick={() => setSelectedMood(m)}
                  >{m}</button>
                ))}
              </div>
              {selectedMood === 'Custom...' && (
                <input
                  className="custom-input"
                  type="text"
                  placeholder="e.g. Melancholic..."
                  value={customMood}
                  onChange={e => setCustomMood(e.target.value)}
                />
              )}
            </div>

            {/* 4. Palette */}
            <div className="section">
              <div className="section-label">Palette</div>
              <div className="chip-row">
                {PALETTES.map(p => (
                  <button
                    key={p.id}
                    className={`chip${selectedPalette === p.id ? ' active' : ''}`}
                    onClick={() => setSelectedPalette(p.id)}
                  >
                    <span className="palette-dot" style={{ background: p.dot }} />
                    {p.name}
                  </button>
                ))}
              </div>
            </div>


          </div>

          {/* Generate button */}
          <div className="generate-area">
            <button
              className="generate-btn"
              onClick={handleGenerate}
              disabled={!prompt}
            >
              {loading
                ? <><Loader2 size={14} className="spin" /> GENERATING...</>
                : <><Sparkles size={14} /> GENERATE</>
              }
            </button>
          </div>
        </aside>

        {/* ── Right panel ── */}
        <div className="right-panel">
          <div className="canvas-area">
            <div className="orb orb-1" />
            <div className="orb orb-2" />
            <div className="orb orb-3" />

            {activeStyleName && (
              <div className="style-tag">{activeStyleName.toUpperCase()}</div>
            )}

            {image ? (
              <div className="canvas-frame">
                <img src={image} alt="Generated art" />
                <a href={image} download className="download-btn">
                  <Download size={18} />
                </a>
              </div>
            ) : error ? (
              <div className="canvas-error">
                <AlertCircle size={40} style={{ opacity:0.5 }} />
                <div>
                  <p style={{ fontWeight:600, marginBottom:4 }}>Generation failed</p>
                  <p style={{ fontSize:12, opacity:0.65 }}>{error}</p>
                </div>
                <button className="retry-btn" onClick={handleGenerate}>
                  <RefreshCw size={13} /> Retry
                </button>
              </div>
            ) : (
              <div className="canvas-empty">
                <ImageIcon size={52} style={{ opacity:0.15 }} />
                <p>Your artwork will appear here</p>
                <span>Configure and generate to begin</span>
              </div>
            )}
          </div>

          {refinedPrompt && (
            <div className="refined-bar">
              <span className={`refined-badge ${promptSource === 'ai' ? 'badge-ai' : 'badge-kw'}`}>
                {promptSource === 'ai' ? 'AI Refined' : 'Keyword Fallback'}
              </span>
              <span className="refined-text">"{refinedPrompt}"</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Gallery ── */}
      {gallery.length > 0 && (
        <section className="gallery-section">
          <div className="gallery-header">
            <History size={16} style={{ color:'var(--muted)' }} />
            <span className="gallery-title">Past Creations</span>
            <span className="gallery-count">{galleryTotal}</span>
          </div>
          <div className="gallery-grid">
            {gallery.map(item => (
              <div key={item.filename} className="gallery-card">
                <div className="gallery-thumb">
                  <img
                    src={`${API_URL}${item.image_path}`}
                    alt={item.original_prompt}
                    onError={e => { e.target.closest('.gallery-card').style.display = 'none'; }}
                  />
                  <div className="gallery-overlay">
                    <p>{item.original_prompt}</p>
                    <div className="gallery-tags">
                      {item.style   && <span className="gallery-tag tag-style">{item.style}</span>}
                      {item.mood    && <span className="gallery-tag tag-mood">{item.mood}</span>}
                      {item.palette && <span className="gallery-tag tag-pal">{item.palette}</span>}
                    </div>
                    {item.creator && <span className="gallery-creator">by {item.creator}</span>}
                  </div>
                </div>
                <div className="gallery-footer">
                  <StarRating itemId={item.id} initialRating={item.rating} />
                  <button
                    className="gallery-delete"
                    onClick={e => { e.stopPropagation(); handleDelete(item.id, item.filename); }}
                    title="Delete"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>

          {gallery.length < galleryTotal && (
            <div style={{ textAlign:'center', marginTop:24 }}>
              <button
                className="load-more-btn"
                onClick={handleLoadMore}
                disabled={loadingMore}
              >
                {loadingMore
                  ? <><Loader2 size={13} className="spin" /> Loading...</>
                  : `Load more (${galleryTotal - gallery.length} remaining)`
                }
              </button>
            </div>
          )}
        </section>
      )}
    </>
  );
}
