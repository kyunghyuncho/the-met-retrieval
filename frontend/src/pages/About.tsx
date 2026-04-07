import { Info, ExternalLink, Library, ShieldCheck, Database, Focus } from 'lucide-react';

export default function About() {
  return (
    <div className="h-full w-full overflow-y-auto pb-24">
      <div className="max-w-5xl mx-auto px-8 pt-16">
        
        {/* Premium Header */}
        <div className="relative mb-16">
          <div className="absolute -inset-1 blur-3xl bg-gradient-to-r from-teal-500/20 via-blue-500/20 to-purple-500/20 opacity-50 rounded-full" />
          <div className="relative flex items-center gap-6">
            <div className="p-4 bg-slate-900 shadow-xl shadow-teal-900/20 rounded-2xl border border-slate-700/50">
              <Info className="text-teal-400" size={40} />
            </div>
            <div>
              <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-slate-100 to-slate-400 tracking-tight">
                Data Source & Provenance
              </h1>
              <p className="text-slate-400 mt-2 text-lg">
                Curated museum antiquities driving the cross-modal retrieval pipeline.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Main Content Column */}
          <div className="lg:col-span-2 space-y-8">
            {/* The Provider Card */}
            <div className="group relative rounded-3xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 p-8 overflow-hidden hover:border-slate-600/50 transition-colors">
              <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
                <Library size={120} />
              </div>
              
              <div className="relative z-10 flex items-center gap-4 mb-6">
                <div className="p-2.5 bg-blue-500/10 rounded-xl">
                  <Library className="text-blue-400" size={28} />
                </div>
                <h2 className="text-2xl font-bold text-slate-100">Art Institute of Chicago</h2>
              </div>
              
              <div className="relative z-10 space-y-5 text-slate-300 leading-relaxed text-lg">
                <p>
                  The foundational data representing the structural knowledge of this contrastive model originates from the <strong>Art Institute of Chicago (AIC)</strong>. Founded in 1879, it is among the oldest, largest, and most highly recognized art museums globally.
                </p>
                <p>
                  By leveraging their pioneering <strong>Public API</strong>, we harvest tens of thousands of high-resolution digital facsimiles alongside rigorous curatorial metadata. This allows for semantic embeddings that deeply capture intricate artistic styles, media, and historical context.
                </p>
              </div>

              <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 gap-4 mt-8">
                <a 
                  href="https://www.artic.edu/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center justify-between p-4 bg-slate-950/50 backdrop-blur-md rounded-2xl border border-slate-700/40 hover:bg-slate-800 hover:border-teal-500/50 hover:shadow-[0_0_20px_rgba(20,184,166,0.15)] transition-all group/link"
                >
                  <span className="font-medium text-slate-200">AIC Homepage</span>
                  <ExternalLink size={20} className="text-slate-500 group-hover/link:text-teal-400 transition-colors" />
                </a>
                <a 
                  href="https://api.artic.edu/docs/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center justify-between p-4 bg-slate-950/50 backdrop-blur-md rounded-2xl border border-slate-700/40 hover:bg-slate-800 hover:border-blue-500/50 hover:shadow-[0_0_20px_rgba(59,130,246,0.15)] transition-all group/link"
                >
                  <span className="font-medium text-slate-200">API Documentation</span>
                  <ExternalLink size={20} className="text-slate-500 group-hover/link:text-blue-400 transition-colors" />
                </a>
              </div>
            </div>

            {/* Acknowledgment */}
            <div className="rounded-3xl bg-gradient-to-br from-teal-500/10 via-slate-900 to-blue-500/5 border border-teal-500/20 p-8 shadow-inner">
              <div className="flex items-center gap-3 mb-4">
                <Focus className="text-teal-400" size={24} />
                <h3 className="text-xl font-semibold text-teal-300">Curatorial Gratitude</h3>
              </div>
              <p className="text-slate-300 text-lg italic leading-relaxed">
                "We extend our profound gratitude to the Art Institute of Chicago for their absolute commitment to open access and digital scholarship. Their robust technical infrastructure and generous licensing frameworks are fundamental cornerstones for independent academic research platforms."
              </p>
            </div>
          </div>

          {/* Sidebar Area */}
          <div className="space-y-6">
            
            <div className="rounded-3xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 p-6 flex flex-col gap-4">
              <div className="w-12 h-12 bg-green-500/10 text-green-400 rounded-full flex items-center justify-center">
                <ShieldCheck size={24} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-200 mb-1">CC0 Open Access</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  All images and metadata leveraged in this retrieval index are distributed under a <strong>Creative Commons Zero (CC0 1.0)</strong> license, entirely free from copyright restrictions.
                </p>
              </div>
            </div>

            <div className="rounded-3xl bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 p-6 flex flex-col gap-4">
              <div className="w-12 h-12 bg-purple-500/10 text-purple-400 rounded-full flex items-center justify-center">
                <Database size={24} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-200 mb-1">Vector Index</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  The semantic embeddings are compressed identically to standard dense FAISS vectors, mapping over dimensionality `D=512` utilizing cross-modal joint approximations.
                </p>
              </div>
            </div>

          </div>
        </div>

        {/* Academic Attribution Footer */}
        <div className="mt-20 pt-8 border-t border-slate-800/80 flex flex-col items-center justify-center text-center gap-2">
          <p className="text-slate-300 font-medium tracking-wide">
            Designed & Implemented by <span className="text-teal-400">Kyunghyun Cho</span>
          </p>
          <p className="text-slate-500 text-sm">
            Professor of Computer Science and Data Science<br />
            New York University
          </p>
          <p className="text-slate-600 text-xs mt-4 uppercase tracking-widest font-semibold">
            © {new Date().getFullYear()} — Academic Open Source
          </p>
        </div>

      </div>
    </div>
  );
}
