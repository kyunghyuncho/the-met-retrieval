import { create } from 'zustand';

interface TelemetryData {
  epoch: number;
  batch?: number;
  train_loss?: number;
  val_loss?: number;
  val_r_at_5?: number;
}

interface AppState {
  isTraining: boolean;
  sessionId: string | null;
  telemetryHistory: TelemetryData[];
  setTrainingStatus: (status: boolean, sessionId?: string) => void;
  addTelemetry: (data: TelemetryData) => void;
  resetTelemetry: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  isTraining: false,
  sessionId: null,
  telemetryHistory: [],
  setTrainingStatus: (status, sessionId = null) => 
    set({ isTraining: status, sessionId }),
  addTelemetry: (data) => 
    set((state) => ({ telemetryHistory: [...state.telemetryHistory, data] })),
  resetTelemetry: () => set({ telemetryHistory: [] }),
}));
