import { useState, useEffect } from 'react';
import { Search as SearchIcon, Image as ImageIcon, Loader2 } from 'lucide-react';
import { useAppStore } from '../store';

export default function Search() {
  const { setSelectedArtifactId } = useAppStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounced Text Search
  useEffect(() => {
    if (!query) {
      setResults([]);
      return;
    }

    const timer = setTimeout(() => {
      handleTextSearch(query);
    }, 500);

    return () => clearTimeout(timer);
  }, [query]);

  const handleTextSearch = async (text: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/search/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, k: 20 })
      });
      if (!res.ok) throw new Error('Search request failed or indices not built.');
      const data = await res.json();
      setResults(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleImageSearch = async (file: File) => {
    setLoading(true);
    setError(null);
    setQuery(''); // Reset text to avoid confusion
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('k', '20');

      const res = await fetch('http://localhost:8000/api/search/image', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Search request failed or indices not built.');
      const data = await res.json();
      setResults(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        handleImageSearch(file);
      } else {
        alert("Please drop an image file.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImageSearch(e.target.files[0]);
    }
  };

  return (
    <div className="p-8 h-full flex flex-col gap-8 max-w-7xl mx-auto" onDragEnter={handleDrag}>
      
      {/* Search Header */}
      <div className="flex flex-col items-center gap-6 mt-8">
        <h1 className="text-3xl font-bold">Cross-Modal Antiquities Retrieval</h1>
        
        <div className={`w-full max-w-2xl relative transition-all duration-300 ${dragActive ? 'scale-105' : ''}`}>
          <div className={`absolute inset-0 z-0 bg-gradient-to-r from-teal-500/20 to-purple-500/20 rounded-2xl blur-xl transition-opacity ${dragActive ? 'opacity-100' : 'opacity-0'}`}></div>
          
          <div 
            className={`relative z-10 bg-slate-900 border-2 rounded-2xl p-2 flex flex-col gap-2 transition-colors ${dragActive ? 'border-teal-400 border-dashed' : 'border-slate-700'}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="flex items-center gap-3 px-4 py-2">
              <SearchIcon className="text-slate-400" size={24} />
              <input 
                type="text" 
                placeholder="Search antiquities by text, or drag an image here..." 
                className="w-full bg-transparent border-none outline-none text-lg text-slate-200 placeholder-slate-500"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <label className="cursor-pointer flex items-center justify-center p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors">
                <ImageIcon size={24} />
                <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
              </label>
            </div>
            
            {dragActive && (
              <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-slate-900/80 backdrop-blur-sm pointer-events-none">
                <p className="text-teal-400 font-medium text-lg">Drop image here to search</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Loading & Error States */}
      <div className="flex justify-center h-8">
        {loading && <Loader2 className="animate-spin text-teal-400" size={24} />}
        {error && <p className="text-red-400">{error}</p>}
      </div>

      {/* Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-20">
        {results.map((item, idx) => (
          <div 
            key={idx} 
            onClick={() => setSelectedArtifactId(item['Object ID'])}
            className="cursor-pointer bg-slate-900 border border-slate-800 rounded-xl overflow-hidden hover:border-teal-500/50 transition-colors group"
          >
            <div className="h-48 w-full relative bg-slate-800 overflow-hidden">
              {item['Primary Image'] ? (
                <img 
                  src={item['Primary Image']} 
                  alt={item['Title'] || item['Object Name']}
                  loading="lazy"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-500">
                  <ImageIcon size={32} />
                </div>
              )}
              <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2 py-1 rounded text-xs font-mono text-teal-400 border border-slate-700">
                {item.similarity_percentage}% match
              </div>
            </div>
            <div className="p-4 flex flex-col gap-2">
              <h3 className="font-semibold text-slate-200 line-clamp-1 truncate" title={item['Title']}>{item['Title'] || 'Untitled'}</h3>
              <p className="text-sm text-slate-400">{item['Object Name']} • {item['Culture'] || 'Unknown Culture'}</p>
              <div className="mt-2 text-xs text-slate-500 line-clamp-3">
                {item['text_serialized']}
              </div>
            </div>
          </div>
        ))}
      </div>
      
    </div>
  );
}
