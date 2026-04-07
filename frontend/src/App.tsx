import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Train from './pages/Train';
import MapView from './pages/Map';
import Search from './pages/Search';
import About from './pages/About';
import { Database, Map as MapIcon, Search as SearchIcon, Info } from 'lucide-react';

function Sidebar() {
  const location = useLocation();
  
  const isActive = (path: string) => {
    return location.pathname === path || (path === '/search' && location.pathname === '/');
  };

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col">
      <div className="p-6">
        <h1 className="text-xl font-bold bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent">Met Retrieval</h1>
      </div>
      <nav className="flex-1 px-4 space-y-2">
        <Link to="/train" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${isActive('/train') ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 text-slate-400'}`}>
          <Database size={20} className="text-teal-400"/>
          <span>Training Config</span>
        </Link>
        <Link to="/map" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${isActive('/map') ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 text-slate-400'}`}>
          <MapIcon size={20} className="text-blue-400"/>
          <span>Geospatial EDA</span>
        </Link>
        <Link to="/search" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${isActive('/search') ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 text-slate-400'}`}>
          <SearchIcon size={20} className="text-purple-400"/>
          <span>Retrieval</span>
        </Link>
        <Link to="/about" className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${isActive('/about') ? 'bg-slate-800 text-white' : 'hover:bg-slate-800 text-slate-400'}`}>
          <Info size={20} className="text-slate-400"/>
          <span>Data Source</span>
        </Link>
      </nav>
    </aside>
  );
}

function App() {
  return (
    <Router>
      <div className="flex h-screen bg-[#0f1115] text-slate-200">
        <Sidebar />
        <main className="flex-1 overflow-auto bg-[#0a0c10]">
          <Routes>
            <Route path="/train" element={<Train />} />
            <Route path="/map" element={<MapView />} />
            <Route path="/search" element={<Search />} />
            <Route path="/about" element={<About />} />
            <Route path="/" element={<Search />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
