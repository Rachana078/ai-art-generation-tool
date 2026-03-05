import { useState, useEffect } from 'react'; // Added useEffect import
import axios from 'axios';
import { Sparkles, Image as ImageIcon, Loader2, Download, History, Palette, Wind, Zap, Smile } from 'lucide-react';

const STYLES = [
  { id: 'watercolor', name: 'Watercolor', icon: <Wind size={16} className="text-blue-400" /> },
  { id: 'cyberpunk', name: 'Cyberpunk', icon: <Zap size={16} className="text-yellow-400" /> },
  { id: 'charcoal-sketch', name: 'Charcoal', icon: <Wind size={16} className="text-slate-400" /> },
  { id: 'pop-art', name: 'Pop Art', icon: <Palette size={16} className="text-pink-400" /> },
  { id: 'renaissance', name: 'Classic Oil', icon: <Palette size={16} className="text-orange-400" /> },
  { id: 'pixel-art', name: 'Pixel Art', icon: <ImageIcon size={16} className="text-green-400" /> },
  { id: 'uquiyo-e', name: 'Ukiyo-e', icon: <Wind size={16} className="text-red-400" /> },
  // Add a comma after the 'surrealism' object below!
  { id: 'surrealism', name: 'Surrealism', icon: <Sparkles size={16} className="text-purple-400" /> }, 
  { id: 'custom', name: 'Custom Style', icon: <Palette size={16}/> },
];

const MOODS = ['Vibrant', 'Dark', 'Dreamy', 'Ethereal', 'Cinematic', 'Custom...']; // Add Custom...

function App() {
  const [prompt, setPrompt] = useState('');
  const [selectedStyle, setSelectedStyle] = useState(null);
  const [selectedMood, setSelectedMood] = useState('Vibrant');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [gallery, setGallery] = useState([]);
  const [customStyle, setCustomStyle] = useState('');
  const [customMood, setCustomMood] = useState('');

  // Fetch gallery on load and whenever a new image is generated
  useEffect(() => {
    const fetchGallery = async () => {
      try {
        const res = await axios.get('http://localhost:8000/gallery');
        setGallery(res.data.items);
      } catch (err) {
        console.error("Gallery fetch failed", err);
      }
    };
    fetchGallery();
  }, [image]);

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    setImage(null); // Clear old image so the user knows a new one is coming

    // 1. You were missing this 'try' keyword!
    try {
      const payload = {
        prompt: prompt,
        // Using your logic to send custom typed words to the backend
        style: selectedStyle === 'custom' ? customStyle : selectedStyle,
        mood: selectedMood === 'Custom...' ? customMood : selectedMood.toLowerCase(),
        palette: "colorful"
      };

      // 2. Wait for the Stable Diffusion process to finish
      const res = await axios.post('http://localhost:8000/generate', payload);

      // 3. Map the backend 'image_url' to the frontend state
      if (res.data && res.data.image_url) {
        setImage(`${res.data.image_url}?t=${Date.now()}`);
      }
    } catch (err) {
      console.error("API Error:", err);
      alert("Generation failed. Check if your backend is running!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-100 font-sans">
      <nav className="border-b border-white/5 bg-black/20 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-gradient-to-br from-purple-500 to-blue-500 p-2 rounded-lg"><Sparkles size={20} /></div>
            <span className="font-black text-xl tracking-tighter uppercase">AI Art Studio</span>
          </div>
          <div className="flex gap-4 text-sm font-medium text-slate-400">
            <button className="flex items-center gap-1 hover:text-white"><History size={16} /> {gallery.length} Creations</button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12 grid grid-cols-1 lg:grid-cols-12 gap-12">
        <div className="lg:col-span-4 space-y-8">
          {/* 1. Prompt */}
          <section className="space-y-4">
            <label className="text-xs font-bold uppercase tracking-widest text-slate-500">1. Vision</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-32 bg-white/5 border border-white/10 rounded-2xl p-4 focus:ring-2 focus:ring-purple-500 outline-none transition-all resize-none"
              placeholder="A futuristic bunny..."
            />
          </section>

          {/* 2. Style Selector */}
          <section className="space-y-4">
            <label className="text-xs font-bold uppercase tracking-widest text-slate-500">2. Style</label>
            <div className="grid grid-cols-2 gap-3">
              {STYLES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedStyle(s.id)}
                  className={`flex items-center gap-2 p-3 rounded-xl border transition-all text-sm ${selectedStyle === s.id ? 'bg-purple-500/20 border-purple-500 text-purple-400' : 'bg-white/5 border-white/5'
                    }`}
                >
                  {s.icon} {s.name}
                </button>
              ))}
            </div>

            {/* CUSTOM STYLE SPACE */}
            {selectedStyle === 'custom' && (
              <input
                type="text"
                placeholder="Type your own style (e.g. 'Cyberpunk Noir')"
                value={customStyle}
                onChange={(e) => setCustomStyle(e.target.value)}
                className="w-full mt-2 bg-white/10 border border-purple-500/50 rounded-xl p-3 text-sm outline-none focus:ring-2 focus:ring-purple-500 transition-all placeholder:text-slate-600"
              />
            )}
          </section>

          {/* 3. Mood Selector */}
          <section className="space-y-4">
            <label className="text-xs font-bold uppercase tracking-widest text-slate-500">3. Mood</label>
            <select
              value={selectedMood}
              onChange={(e) => setSelectedMood(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 outline-none text-slate-300"
            >
              {MOODS.map(m => <option key={m} value={m} className="bg-[#1a1a1a]">{m}</option>)}
            </select>

            {/* CUSTOM MOOD SPACE */}
            {selectedMood === 'Custom...' && (
              <input
                type="text"
                placeholder="Type your own mood (e.g. 'Melancholic')"
                value={customMood}
                onChange={(e) => setCustomMood(e.target.value)}
                className="w-full mt-2 bg-white/10 border border-blue-500/50 rounded-xl p-3 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition-all placeholder:text-slate-600"
              />
            )}
          </section>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 py-4 rounded-2xl font-bold flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <Loader2 className="animate-spin" /> : <Sparkles size={20} />}
            {loading ? "AI is Painting..." : "Generate Magic"}
          </button>
        </div>

        <div className="lg:col-span-8">
          <div className="aspect-square w-full max-w-2xl mx-auto relative">
            {image ? (
              <div className="h-full w-full rounded-[2rem] overflow-hidden border border-white/10 relative">
                <img src={image} className="w-full h-full object-cover" alt="Result" />
                <a href={image} download className="absolute bottom-6 right-6 bg-white text-black p-3 rounded-xl shadow-2xl hover:scale-110 transition-transform">
                  <Download size={20} />
                </a>
              </div>
            ) : (
              <div className="h-full w-full rounded-[2rem] border-2 border-dashed border-white/5 bg-white/[0.02] flex flex-col items-center justify-center text-slate-600">
                <ImageIcon size={48} className="opacity-20 mb-4" />
                <p className="font-medium">Ready for your prompt</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;