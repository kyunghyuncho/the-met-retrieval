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
  selectedArtifactId: string | number | null;
  setTrainingStatus: (status: boolean, sessionId?: string | null) => void;
  addTelemetry: (data: TelemetryData) => void;
  resetTelemetry: () => void;
  setSelectedArtifactId: (id: string | number | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  isTraining: false,
  sessionId: null,
  telemetryHistory: [],
  selectedArtifactId: null,
  setTrainingStatus: (status, sessionId = null) => 
    set({ isTraining: status, sessionId }),
  addTelemetry: (data) => 
    set((state) => ({ telemetryHistory: [...state.telemetryHistory, data] })),
  resetTelemetry: () => set({ telemetryHistory: [] }),
  setSelectedArtifactId: (id) => set({ selectedArtifactId: id }),
}));
