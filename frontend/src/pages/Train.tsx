import { useState, useEffect, useRef, useMemo } from 'react';
import { useAppStore } from '../store';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Square, Activity } from 'lucide-react';

export default function Train() {
  const { isTraining, sessionId, telemetryHistory, setTrainingStatus, addTelemetry, resetTelemetry } = useAppStore();
  const [lr, setLr] = useState<number>(0.0001);
  const [batchSize, setBatchSize] = useState<number>(256);
  const [dJoint, setDJoint] = useState<number>(512);
  const [maxEpochs, setMaxEpochs] = useState<number>(50);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (isTraining && sessionId) {
      setWsStatus('connecting');
      const ws = new WebSocket(`ws://127.0.0.1:8000/ws/telemetry/${sessionId}`);
      
      ws.onopen = () => {
        setWsStatus('connected');

        console.log(' [WebSocket] Connected to telemetry for session:', sessionId);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'train_step' || data.type === 'val_epoch') {
            addTelemetry({
              epoch: data.epoch,
              batch: data.batch,
              train_loss: data.train_loss,
              val_loss: data.val_loss,
              val_r_at_5: data.val_r_at_5
            });
          } else if (data.status === 'completed') {
            console.log(' [WebSocket] Training completed');
            setTrainingStatus(false, null);
            setWsStatus('disconnected');
            ws.close();
          }
        } catch (e) {
          console.error(" [WebSocket] Error parsing telemetry", e);
        }
      };
      
      ws.onerror = (e) => {
        setWsStatus('disconnected');
        console.error(' [WebSocket] Error:', e);
      };
      
      ws.onclose = () => {
        setWsStatus('disconnected');
      };

      
      wsRef.current = ws;
      
      return () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.close();
        }
      };
    }
  }, [isTraining, sessionId, addTelemetry, setTrainingStatus]);

  const handleStart = async () => {
    try {
      resetTelemetry();
      const payload = {
        learning_rate: lr,
        batch_size: batchSize,
        d_joint: dJoint,
        max_epochs: maxEpochs,
        temperature_init: 0.07
      };
      
      const res = await fetch('http://127.0.0.1:8000/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) throw new Error('Failed to start training');
      
      const data = await res.json();
      setTrainingStatus(true, data.session_id);
      
    } catch (err) {
      console.error(err);
      alert('Error starting training. Ensure data pipeline has been run.');
    }
  };

  const handleAbort = async () => {
    if (!sessionId) return;
    try {
      await fetch(`http://127.0.0.1:8000/api/train/${sessionId}`, {
        method: 'DELETE'
      });
      setTrainingStatus(false, null);
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (err) {
      console.error('Abort failed', err);
    }
  };

  // Process telemetry for high-resolution batch-level chart
  const chartData = useMemo(() => {
    let lastTrainLoss = undefined;
    let lastValLoss = undefined;
    let lastValR = undefined;
    
    return telemetryHistory.map((curr, idx) => {
      if (curr.train_loss !== undefined) lastTrainLoss = curr.train_loss;
      if (curr.val_loss !== undefined) lastValLoss = curr.val_loss;
      if (curr.val_r_at_5 !== undefined) lastValR = curr.val_r_at_5;
      
      return {
        step: idx,
        epoch: curr.epoch,
        batch: curr.batch,
        train_loss: lastTrainLoss,
        val_loss: lastValLoss,
        val_r_at_5: lastValR
      };
    });
  }, [telemetryHistory]);

  return (
    <div className="p-8 h-full flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="text-teal-400" size={28} />
          <h1 className="text-2xl font-semibold">Contrastive Learning Configuration</h1>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${wsStatus === 'connected' ? 'bg-teal-400 animate-pulse' : wsStatus === 'connecting' ? 'bg-yellow-400' : 'bg-slate-600'}`} />
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{wsStatus}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        
        {/* Config Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col gap-6">
          <h2 className="text-lg font-medium text-slate-300 border-b border-slate-800 pb-2">Hyperparameters</h2>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-400">Learning Rate</label>
                <span className="text-sm text-teal-400">{lr}</span>
              </div>
              <input type="range" min="0.00001" max="0.001" step="0.00001" value={lr} onChange={e => setLr(parseFloat(e.target.value))} className="w-full accent-teal-500" disabled={isTraining} />
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-400">Batch Size</label>
                <span className="text-sm text-teal-400">{batchSize}</span>
              </div>
              <input type="range" min="32" max="1024" step="32" value={batchSize} onChange={e => setBatchSize(parseInt(e.target.value))} className="w-full accent-teal-500" disabled={isTraining} />
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-400">Joint Projection Dim</label>
                <span className="text-sm text-teal-400">{dJoint}</span>
              </div>
              <input type="range" min="128" max="1024" step="64" value={dJoint} onChange={e => setDJoint(parseInt(e.target.value))} className="w-full accent-teal-500" disabled={isTraining} />
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-400">Max Epochs</label>
                <span className="text-sm text-teal-400">{maxEpochs}</span>
              </div>
              <input type="range" min="1" max="200" step="1" value={maxEpochs} onChange={e => setMaxEpochs(parseInt(e.target.value))} className="w-full accent-teal-500" disabled={isTraining} />
            </div>
          </div>
          
          <div className="mt-auto">
            {isTraining ? (
              <button onClick={handleAbort} className="w-full py-3 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors flex items-center justify-center gap-2 font-medium">
                <Square size={18} />
                Abort Training
              </button>
            ) : (
              <button onClick={handleStart} className="w-full py-3 rounded-lg bg-teal-500/20 text-teal-400 hover:bg-teal-500/30 transition-colors flex items-center justify-center gap-2 font-medium">
                <Play size={18} />
                Start Training
              </button>
            )}
          </div>
        </div>


        {/* Telemetry Chart */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col">
          <h2 className="text-lg font-medium text-slate-300 border-b border-slate-800 pb-2 mb-4">Training Telemetry</h2>
          <div className="flex-1 min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis 
                  dataKey="step" 
                  stroke="#64748b" 
                  tick={{fill: '#64748b'}} 
                  label={{ value: 'Training Steps', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 12 }} 
                />
                <YAxis stroke="#64748b" tick={{fill: '#64748b'}} domain={['auto', 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f1115', borderColor: '#1e293b', color: '#f1f5f9' }}
                  itemStyle={{ color: '#f1f5f9' }}
                  labelFormatter={(value, payload) => {
                    const item = payload[0]?.payload;
                    return item ? `Step ${value} (Epoch ${item.epoch}, Batch ${item.batch})` : `Step ${value}`;
                  }}
                />
                <Legend />
                <Line type="monotone" dataKey="train_loss" stroke="#2dd4bf" strokeWidth={2} dot={false} name="Train Loss" animationDuration={300} />
                <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" strokeWidth={2} dot={false} name="Val Loss" animationDuration={300} />
              </LineChart>
            </ResponsiveContainer>

          </div>
        </div>
        
      </div>
    </div>
  );
}
