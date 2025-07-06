#!/bin/bash

# Create root folder
mkdir -p cosmicmind/{frontend,backend,shared}
mkdir -p cosmicmind/frontend/{src,public}
mkdir -p cosmicmind/frontend/src/{components,styles}
mkdir -p cosmicmind/backend/{ai_core,api,data}
mkdir -p cosmicmind/shared/types

# Copy frontend files
cat << 'EOF' > cosmicmind/frontend/public/index.html
<!-- PASTE YOUR FULL HTML FILE CONTENT HERE -->
EOF

cat << 'EOF' > cosmicmind/frontend/src/App.tsx
// Paste App.tsx code
EOF

cat << 'EOF' > cosmicmind/frontend/src/main.tsx
// Paste main.tsx code
EOF

cat << 'EOF' > cosmicmind/frontend/src/components/NeuralVisualizer.tsx
// Paste component code
EOF

# Repeat for all other components...

# Backend files
cat << 'EOF' > cosmicmind/backend/ai_core/quantum_net.py
# Paste quantum_net.py code
EOF

cat << 'EOF' > cosmicmind/backend/main.py
# Paste FastAPI main code
EOF

# And so on for all backend files...

# Shared type
cat << 'EOF' > cosmicmind/shared/types/ai_types.ts
// Paste shared types
EOF

# Top-level files
cat << 'EOF' > cosmicmind/docker-compose.yml
# Paste docker-compose.yml
EOF

cat << 'EOF' > cosmicmind/README.md
# Paste README.md
EOF

cat << 'EOF' > cosmicmind/.gitignore
# Paste .gitignore
EOF

# Package JSON
cat << 'EOF' > cosmicmind/frontend/package.json
{
  "name": "cosmicmind-frontend",
  "private": true,
  "version": "0.1.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "three": "^0.152.2",
    "chart.js": "^4.4.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^3.2.0",
    "typescript": "^5.2.2",
    "vite": "^4.4.9"
  }
}
EOF

# Compress
cd ..
zip -r cosmicmind.zip cosmicmind/