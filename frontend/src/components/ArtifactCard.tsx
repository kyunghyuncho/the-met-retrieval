import { useEffect, useState } from 'react';
import { useAppStore } from '../store';
import { X, Loader2, Image as ImageIcon, MapPin, Calendar, Fingerprint, Palette } from 'lucide-react';

export default function ArtifactCard() {
  const { selectedArtifactId, setSelectedArtifactId } = useAppStore();
  const [artifact, setArtifact] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedArtifactId) {
      setArtifact(null);
      setError(null);
      return;
    }

    const fetchArtifact = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`http://localhost:8000/api/metadata/item/${selectedArtifactId}`);
        if (!res.ok) throw new Error('Artifact not found');
        const data = await res.json();
        setArtifact(data);
      } catch (err: any) {
        console.error(err);
        setError('Failed to load artifact details.');
      } finally {
        setLoading(false);
      }
    };

    fetchArtifact();
  }, [selectedArtifactId]);

  if (!selectedArtifactId) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="fixed inset-0" 
        onClick={() => setSelectedArtifactId(null)}
      />
      
      <div className="relative z-10 w-full max-w-4xl max-h-[90vh] bg-slate-900 border border-slate-700/60 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md">
          <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <Fingerprint className="text-teal-400" size={20} />
            Artifact Registration
          </h2>
          <button 
            onClick={() => setSelectedArtifactId(null)}
            className="p-2 rounded-full hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-64 gap-4">
              <Loader2 className="w-8 h-8 text-teal-400 animate-spin" />
              <p className="text-slate-400">Retrieving secure historical record...</p>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-64 text-red-400">
              {error}
            </div>
          ) : artifact ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Image Section */}
              <div className="space-y-4">
                <div className="aspect-[4/5] rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center border border-slate-800 relative group">
                  {artifact['Primary Image'] ? (
                    <img 
                      src={artifact['Primary Image']} 
                      alt={artifact['Title'] || artifact['Object Name']}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                    />
                  ) : (
                    <div className="flex flex-col items-center text-slate-600">
                      <ImageIcon size={48} />
                      <span className="mt-2 text-sm">Image Unavailable</span>
                    </div>
                  )}
                  {artifact['Department'] && (
                    <div className="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/50">
                      <span className="text-xs font-medium text-teal-400 tracking-wider uppercase">
                        {artifact['Department']}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Data Section */}
              <div className="flex flex-col gap-6">
                <div>
                  <h1 className="text-3xl font-bold text-slate-100 mb-2 leading-tight">
                    {artifact['Title'] || 'Untitled Artifact'}
                  </h1>
                  <p className="text-lg text-slate-400">
                    {artifact['Culture'] || 'Unknown Culture'}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                      <Calendar size={16} />
                      <span className="text-xs font-semibold uppercase">Date Origin</span>
                    </div>
                    <p className="text-slate-200">{artifact['Object Date'] || 'Unknown'}</p>
                  </div>
                  <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                      <Palette size={16} />
                      <span className="text-xs font-semibold uppercase">Dimensions</span>
                    </div>
                    <p className="text-slate-200 text-sm">{artifact['Dimensions'] || 'N/A'}</p>
                  </div>
                </div>

                <div className="space-y-4 mt-2">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Detailed Medium</h3>
                    <p className="text-slate-300 leading-relaxed bg-slate-800/30 p-4 rounded-lg border border-slate-800">
                      {artifact['Medium'] || 'Unspecified'}
                    </p>
                  </div>
                  
                  {artifact['Credit Line'] && (
                    <div>
                      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Provenance & Acquisition</h3>
                      <p className="text-slate-300 text-sm leading-relaxed">
                        {artifact['Credit Line']}
                      </p>
                    </div>
                  )}
                  
                  {(artifact['City'] || artifact['Country'] || artifact['Region']) && (
                    <div className="flex items-center gap-2 text-blue-400 pt-2">
                      <MapPin size={18} />
                      <span className="text-sm">
                        {[artifact['City'], artifact['Region'], artifact['Country']].filter(Boolean).join(', ')}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
