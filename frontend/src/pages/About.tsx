import { Info, ExternalLink, Library } from 'lucide-react';

export default function About() {
  return (
    <div className="max-w-4xl mx-auto p-12">
      <div className="space-y-8">
        {/* Header */}
        <div className="flex items-center gap-4 border-b border-slate-800 pb-6">
          <div className="p-3 bg-teal-500/10 rounded-xl">
            <Info className="text-teal-400" size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Data Source & Attribution</h1>
            <p className="text-slate-400 mt-1">Provenance of the antiquities dataset used in this system.</p>
          </div>
        </div>

        {/* AIC Section */}
        <div className="bg-slate-900/50 rounded-2xl border border-slate-800 p-8 space-y-6 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <Library className="text-blue-400" size={24} />
            <h2 className="text-xl font-semibold text-slate-100">Art Institute of Chicago (AIC)</h2>
          </div>
          
          <div className="space-y-4 text-slate-300 leading-relaxed">
            <p>
              The primary data source for this retrieval system is the <strong>Art Institute of Chicago (AIC)</strong>. 
              Founded in 1879, the Art Institute is one of the oldest and largest art museums in the United States, 
              housing a permanent collection that spans centuries and continents.
            </p>
            <p>
              This project leverages the <strong>AIC Public API</strong> and their pioneering <strong>Open Access</strong> 
              initiative, which provides high-resolution images and comprehensive metadata for nearly 50,000 public-domain 
              artworks under a Creative Commons Zero (CC0 1.0) license.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
            <a 
              href="https://www.artic.edu/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center justify-between p-4 bg-slate-800/50 hover:bg-slate-800 rounded-xl border border-slate-700/50 transition-all group"
            >
              <span className="text-slate-200">Official Website</span>
              <ExternalLink size={18} className="text-slate-500 group-hover:text-teal-400 transition-colors" />
            </a>
            <a 
              href="https://api.artic.edu/docs/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center justify-between p-4 bg-slate-800/50 hover:bg-slate-800 rounded-xl border border-slate-700/50 transition-all group"
            >
              <span className="text-slate-200">API Documentation</span>
              <ExternalLink size={18} className="text-slate-500 group-hover:text-teal-400 transition-colors" />
            </a>
          </div>
        </div>

        {/* Gratitude Section */}
        <div className="bg-gradient-to-br from-teal-500/5 to-blue-500/5 rounded-2xl border border-teal-500/10 p-8 text-center">
          <h2 className="text-lg font-semibold text-teal-300 mb-3">Acknowledgements</h2>
          <p className="text-slate-400 italic">
            "We extend our deepest gratitude to the team at the Art Institute of Chicago for their commitment 
            to open access and digital scholarship. Their robust technical infrastructure and generous licensing 
            models are fundamental to the existence of academic research platforms like this one."
          </p>
        </div>

        {/* License Footer */}
        <div className="text-center text-slate-600 text-sm">
          <p>© {new Date().getFullYear()} Implementation by Kyunghyun Cho. Data provided by AIC under CC0 1.0.</p>
        </div>
      </div>
    </div>
  );
}
