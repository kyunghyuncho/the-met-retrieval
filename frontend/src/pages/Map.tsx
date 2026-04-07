import { useState, useEffect, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';
import { BitmapLayer } from '@deck.gl/layers';
import { MapPin } from 'lucide-react';
import { useAppStore } from '../store';

// Using a professional dark base map style via OSM
const INITIAL_VIEW_STATE = {
  longitude: 15,
  latitude: 35,
  zoom: 1.8,
  pitch: 0,
  bearing: 0
};

export default function MapView() {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { setSelectedArtifactId } = useAppStore();

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/metadata/locations')
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
    // Add an OpenStreetMap basemap layer
    new TileLayer({
      id: 'tile-layer',
      data: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: (props) => {
        const {
          bbox: { west, south, east, north }
        } = props.tile as any;

        return new BitmapLayer(props as any, {
          data: undefined,
          image: props.data,
          bounds: [west, south, east, north]
        } as any);
      }
    }),
    
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data,
      pickable: true,
      opacity: 0.8,
      stroked: false,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 3,
      radiusMaxPixels: 12,
      lineWidthMinPixels: 1,
      getPosition: (d: any) => [d.longitude, d.latitude],
      getFillColor: () => [45, 212, 191, 150], // Teal 400
      onClick: (info: any) => {
        if (info.object) {
          setSelectedArtifactId(info.object.id);
        }
      }
    })
  ], [data, setSelectedArtifactId]);

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
          initialViewState={INITIAL_VIEW_STATE as any}
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
