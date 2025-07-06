// CosmicMind.tsx
import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import Chart from 'chart.js/auto';

const CosmicMindDashboard: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize 3D Scene
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0b20);
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);

    const controls = new OrbitControls(camera, renderer.domElement);
    camera.position.z = 25;
    camera.position.y = 10;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x333366, 1.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.5);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Quantum Core
    const geometry = new THREE.DodecahedronGeometry(2.5, 3);
    const material = new THREE.MeshPhysicalMaterial({
      color: 0x00c9ff,
      emissive: 0x0044ff,
      metalness: 0.8,
      roughness: 0.1,
      transmission: 0.9,
      opacity: 0.95,
      transparent: true,
    });
    const core = new THREE.Mesh(geometry, material);
    scene.add(core);

    // Animation Loop
    const animate = () => {
      requestAnimationFrame(animate);
      core.rotation.x += 0.01;
      core.rotation.y += 0.01;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Initialize Charts
  useEffect(() => {
    const ctx = document.getElementById('consensusChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Integrity', 'Ethics', 'Resources', 'Security', 'Efficiency', 'Decentralization'],
        datasets: [{
          label: 'Consensus',
          data: [92, 96, 88, 99, 85, 94],
          backgroundColor: 'rgba(77, 204, 255, 0.2)',
          borderColor: '#4dccff'
        }]
      }
    });
  }, []);

  return (
    <div style={{ fontFamily: 'Segoe UI' }}>
      <h1>CosmicMind: Adaptive Quantum AI</h1>
      <div ref={containerRef} style={{ width: '100%', height: '70vh', position: 'relative' }}>
        <canvas ref={canvasRef}></canvas>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-around', margin: '40px 0' }}>
        <div style={{ background: 'rgba(30,50,100,0.8)', padding: '20px', borderRadius: '15px' }}>
          <h3>Emergent Intelligence</h3>
          <p id="intelligence">0.92</p>
        </div>
        <div style={{ background: 'rgba(30,50,100,0.8)', padding: '20px', borderRadius: '15px' }}>
          <h3>Ethical Compliance</h3>
          <p id="ethical">98%</p>
        </div>
        <div style={{ background: 'rgba(30,50,100,0.8)', padding: '20px', borderRadius: '15px' }}>
          <h3>Neural Connections</h3>
          <p id="complexity">42K</p>
        </div>
        <div style={{ background: 'rgba(30,50,100,0.8)', padding: '20px', borderRadius: '15px' }}>
          <h3>Energy Efficiency</h3>
          <p id="energy">33%</p>
        </div>
      </div>

      <canvas id="consensusChart" width="400" height="200"></canvas>
    </div>
  );
};

export default CosmicMindDashboard;