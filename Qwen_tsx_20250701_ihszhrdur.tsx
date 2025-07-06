import React, { useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import NeuralVisualizer from './NeuralVisualizer';
import MetricsPanel from './MetricsPanel';
import ControlPanel from './ControlPanel';

const App = () => {
  const [metrics, setMetrics] = useState({
    intelligence: 0.92,
    ethical: "98%",
    complexity: "42K",
    energy: "33%"
  });

  return (
    <div className="app">
      <h1>CosmicMind: Conscious AI System</h1>
      <NeuralVisualizer />
      <MetricsPanel metrics={metrics} />
      <ControlPanel setMetrics={setMetrics} />
    </div>
  );
};

export default App;