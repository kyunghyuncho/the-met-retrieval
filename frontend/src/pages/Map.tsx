import { useState, useEffect, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { MapPin } from 'lucide-react';

// Using a dark base map
const INITIAL_VIEW_STATE = {
  longitude: 0,
  latitude: 20,
  zoom: 1.5,
  pitch: 0,
  bearing: 0
};

export default function MapView() {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/api/metadata/locations')
      .then(res => res.json())
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load map data', err);
        setLoading(false);
      });
  }, []);

  const layers = useMemo(() => [
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data,
      pickable: true,
      opacity: 0.8,
      stroked: false,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 2,
      radiusMaxPixels: 10,
      lineWidthMinPixels: 1,
      getPosition: (d: any) => [d.longitude, d.latitude],
      getFillColor: (d: any) => {
        // Simplified mapping for aesthetics. E.g. randomish distribution or based on age.
        // A full implementation would parse d.age string properly.
        return [45, 212, 191, 150]; // Teal 400
      },
      onClick: (info: any) => {
        if (info.object) {
          console.log('Clicked artifact:', info.object.id);
          // In a full implementation, slide in a side-panel.
        }
      }
    })
  ], [data]);

  return (
    <div className="h-full w-full relative flex flex-col items-center justify-center">
      {loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#0a0c10]/80 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-4 text-slate-400">
            <div className="w-8 h-8 rounded-full border-2 border-teal-500 border-t-transparent animate-spin"></div>
            <p>Loading dataset for geospatial rendering...</p>
          </div>
        </div>
      )}
      
      {!loading && data.length === 0 && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center">
          <MapPin size={48} className="text-slate-600 mb-4" />
          <h2 className="text-xl text-slate-400">No geospatial data available</h2>
          <p className="text-slate-500 text-sm mt-2">Make sure you have run the geocoding pipeline.</p>
        </div>
      )}

      {/* Since we don't have a mapbox token in the pedagogical app, we will use DeckGL without a basemap
          or a simple tile layer wrapper. For simple visualization of coordinates, points will outline the map. */}
      <div className="absolute inset-4 rounded-xl border border-slate-800 overflow-hidden bg-slate-900/50">
        <DeckGL
          initialViewState={INITIAL_VIEW_STATE}
          controller={true}
          layers={layers}
          getTooltip={(info: any) => info.object ? `Artifact ID: ${info.object.id}\nLocation: ${info.object.longitude.toFixed(2)}, ${info.object.latitude.toFixed(2)}` : null}
        >
          {/* TileLayer would go here for basemap if Nominatim raster tiles are used */}
        </DeckGL>
      </div>

    </div>
  );
}
