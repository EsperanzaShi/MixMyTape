import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "./App.css";

const cosineSimilarity = (a, b) => {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
  return dot / (magA * magB);
};

const App = () => {
  const [points, setPoints] = useState([]);
  const [genres, setGenres] = useState([]);
  const [selectedGenres, setSelectedGenres] = useState(new Set());
  const [topN, setTopN] = useState(1);
  const [hiphopFile, setHiphopFile] = useState(null);
  const [hiphopLatent, setHiphopLatent] = useState(null);
  const [similarPoints, setSimilarPoints] = useState([]);
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [remixMode, setRemixMode] = useState(null);
  const [status, setStatus] = useState("");
  const [mixtapeUrl, setMixtapeUrl] = useState(null);
  const [camera, setCamera] = useState(undefined);
  
  // Instrument classification states
  const [activeTab, setActiveTab] = useState("remix"); // "remix" or "classify"
  const [classifyFile, setClassifyFile] = useState(null);
  const [classificationResults, setClassificationResults] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [windowDuration, setWindowDuration] = useState(3.0);

  useEffect(() => {
    fetch("/data/umap_embeddings.json")
      .then((res) => res.json())
      .then((data) => {
        setPoints(data);
        setGenres([...new Set(data.map((p) => p.genre))]);
        setSelectedGenres(new Set(data.map((p) => p.genre)));
      });
  }, []);

  // Helper to check if all genres are selected
  const allGenresSelected = genres.length > 0 && genres.every(g => selectedGenres.has(g));

  // When hiphopLatent, topN, selectedGenres, or points change, update similarPoints
  useEffect(() => {
    if (!hiphopLatent) {
      setSimilarPoints([]);
      return;
    }
    // If all genres are selected, consider all points; otherwise, only selected genres
    const candidates = allGenresSelected
      ? points.filter((p) => p.latent)
      : points.filter((p) => p.latent && selectedGenres.has(p.genre));
    const sims = candidates
      .map((p) => ({
        ...p,
        similarity: cosineSimilarity(hiphopLatent, p.latent),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topN);
    setSimilarPoints(sims);
  }, [hiphopLatent, topN, points, selectedGenres, allGenresSelected]);

  // Upload and encode hiphop track
  const handleAudioUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setHiphopFile(file);
    setStatus("Encoding hiphop track...");
    setSelectedChunk(null);
    setRemixMode(null);
    setMixtapeUrl(null);

    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/api/encode", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      setHiphopLatent(data.latent);
      setStatus("Hiphop track encoded. Select a sample to remix.");
    } else {
      setStatus("Encoding failed.");
    }
  };

  // Handle instrument classification file upload
  const handleClassifyFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setClassifyFile(file);
    setClassificationResults(null);
  };

  // Run instrument classification
  const handleClassify = async () => {
    if (!classifyFile) return;
    
    setIsClassifying(true);
    setStatus("Analyzing instruments...");
    
    const formData = new FormData();
    formData.append("file", classifyFile);
    formData.append("window_duration", windowDuration);
    
    try {
      const res = await fetch("http://localhost:8000/api/classify", {
        method: "POST",
        body: formData,
      });
      
      if (res.ok) {
        const data = await res.json();
        setClassificationResults(data);
        setStatus("Instrument analysis complete!");
      } else {
        setStatus("Classification failed.");
      }
    } catch (error) {
      setStatus("Classification failed: " + error.message);
    } finally {
      setIsClassifying(false);
    }
  };

  // Only allow selection from Top-N
  const handleClickPoint = (data) => {
    if (!hiphopLatent) return;
    const path = data.points[0].customdata;
    const clicked = similarPoints.find((p) => p.path === path);
    if (!clicked) return;
    setSelectedChunk(clicked);
    setRemixMode(null);
    setMixtapeUrl(null);
    setStatus(`Selected sample: ${clicked.path}. Choose remix mode.`);
    // Play audio with volume control
    const audio = new Audio(clicked.path);
    audio.volume = 0.3; // Set volume to 30% to match remixed audio loudness
    audio.play();
  };

  // Preserve camera state on user interaction
  const handleRelayout = (event) => {
    if (event['scene.camera']) {
      setCamera(event['scene.camera']);
    }
  };

  // Helper to extract parent track filename from chunk path
  const getParentTrackFilename = (chunkPath) => {
    const match = chunkPath.match(/chunked_wavs\/(\w+)\/(\w+\.\d+)_chunk\d+\.wav/);
    if (match) {
      return `${match[1]}/${match[2]}.wav`;
    }
    return null;
  };

  // Handle remix mode click
  const handleRemix = async (mode) => {
    if (!hiphopFile || !selectedChunk) return;
    setRemixMode(mode);
    setStatus("Generating remix...");
    setMixtapeUrl(null);
    const backendMode = mode === 'blend' ? 'weighted_blend' : mode;
    const formData = new FormData();
    formData.append("file1", hiphopFile);
    formData.append("parent_track_filename", getParentTrackFilename(selectedChunk.path));
    formData.append("remix_mode", backendMode);
    const res = await fetch("http://localhost:8000/api/remix", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setMixtapeUrl(url);
      setStatus("Remix complete! Play or download below.");
    } else {
      setStatus("Remix failed.");
    }
  };

  // Plotly data: always show all points, but highlight Top-N
  const SCATTER_SCALE = 1; // Try 2, 3, or higher for more spread
  const plotData = [
    {
      x: points.map((p) => p.x * SCATTER_SCALE),
      y: points.map((p) => p.y * SCATTER_SCALE),
      z: points.map((p) => p.z * SCATTER_SCALE),
      text: points.map((p) => `Genre: ${p.genre}<br>ID: ${p.id}`),
      customdata: points.map((p) => p.path),
      type: "scatter3d",
      mode: "markers",
      marker: {
        size: 8,
        color: points.map((p) =>
          similarPoints.find((sp) => sp.path === p.path)
            ? "red"
            : "#00bcd4"
        ),
        opacity: 0.8,
      },
      name: "Samples",
    },
  ];

  // Add anchor (hiphop latent) as a big white dot with halo
  if (hiphopLatent) {
    // Find the 3D coordinates for the anchor (projected UMAP, if available)
    // If you have a way to project the hiphopLatent to UMAP, use that here.
    // For now, we'll use the mean of the topN similar points as a placeholder.
    let anchorX = 0, anchorY = 0, anchorZ = 0;
    if (similarPoints.length > 0) {
      anchorX = similarPoints[0].x;
      anchorY = similarPoints[0].y;
      anchorZ = similarPoints[0].z;
    } else if (points.length > 0) {
      anchorX = points[0].x;
      anchorY = points[0].y;
      anchorZ = points[0].z;
    }
    plotData.push({
      x: [anchorX],
      y: [anchorY],
      z: [anchorZ],
      text: ["Hiphop Anchor"],
      customdata: [null],
      type: "scatter3d",
      mode: "markers",
      marker: {
        size: 14,
        color: "#fff",
        opacity: 1,
        line: {
          width: 8,
          color: "rgba(255,255,255,0.4)",
          blur: 4,
        },
      },
      name: "Hiphop Anchor",
      showlegend: false,
    });
  }

  return (
    <div className="app-container">
      {/* Left: Samples */}
      <div className="sidebar">
        <h2>Samples</h2>
        {genres.map((genre) => (
          <label key={genre}>
            <input
              type="checkbox"
              checked={selectedGenres.has(genre)}
              onChange={() => {
                const newSet = new Set(selectedGenres);
                newSet.has(genre) ? newSet.delete(genre) : newSet.add(genre);
                setSelectedGenres(newSet);
              }}
            />
            {genre}
          </label>
        ))}
        {/* Top-N vertical slider */}
        <div className="topn-container">
          <div className="topn-label">
            Top - <span className="topn-value">{topN}</span> <br />recommend
          </div>
          <input
            type="range"
            min={1}
            max={10}
            value={topN}
            className="horizontal-slider"
            onChange={e => setTopN(Number(e.target.value))}
          />
        </div>
      </div>

      {/* Center: Track and Plot */}
      <div className="center-panel">
        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={`tab-button ${activeTab === "remix" ? "active" : ""}`}
            onClick={() => setActiveTab("remix")}
          >
            Remix
          </button>
          <button 
            className={`tab-button ${activeTab === "classify" ? "active" : ""}`}
            onClick={() => setActiveTab("classify")}
          >
            Classify
          </button>
        </div>

        {activeTab === "remix" ? (
          <>
            <h1>Track</h1>
            <input type="file" accept="audio/*" onChange={handleAudioUpload} />
            <Plot
              className="plot-container"
              data={plotData}
              layout={{
                width: undefined,
                height: undefined,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                font: { color: "#fff" },
                scene: {
                  xaxis: { visible: false, showgrid: false, zeroline: false },
                  yaxis: { visible: false, showgrid: false, zeroline: false },
                  zaxis: { visible: false, showgrid: false, zeroline: false },
                  camera: camera || {
                    eye: { x: 1.2, y: 1.2, z: 0.7 } // Move camera closer (smaller values = closer)
                  },
                },
                autosize: true,
                margin: { t: 30, r: 0, l: 0, b: 0 },
              }}
              useResizeHandler={true}
              style={{width: '100%', height: '100%'}}
              onClick={handleClickPoint}
              onRelayout={handleRelayout}
            />
          </>
        ) : (
          <div className="classify-panel">
            <h1>Instrument Classifier</h1>
            
            {/* File Upload */}
            <div className="upload-section">
              <input 
                type="file" 
                accept="audio/*" 
                onChange={handleClassifyFileUpload}
                className="classify-file-input"
              />
              {classifyFile && (
                <div className="file-info">
                  <p>Selected: {classifyFile.name}</p>
                  <audio controls src={URL.createObjectURL(classifyFile)} />
                </div>
              )}
            </div>

            {/* Settings */}
            <div className="settings-section">
              <label>
                Window Duration: {windowDuration}s
                <input
                  type="range"
                  min={1.0}
                  max={5.0}
                  step={0.5}
                  value={windowDuration}
                  onChange={(e) => setWindowDuration(parseFloat(e.target.value))}
                  className="window-slider"
                />
              </label>
            </div>

            {/* Analysis Button */}
            <button 
              onClick={handleClassify}
              disabled={!classifyFile || isClassifying}
              className="classify-button"
            >
              {isClassifying ? "Analyzing..." : "Start Analysis"}
            </button>

            {/* Results */}
            {classificationResults && (
              <div className="results-section">
                <h2>Analysis Results</h2>
                
                {/* Summary */}
                <div className="summary-grid">
                  <div className="summary-item">
                    <h3>Most Common</h3>
                    <p>{classificationResults.summary.most_common_instrument_name}</p>
                  </div>
                  <div className="summary-item">
                    <h3>Avg Confidence</h3>
                    <p>{(classificationResults.summary.average_confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="summary-item">
                    <h3>Total Windows</h3>
                    <p>{classificationResults.summary.total_windows}</p>
                  </div>
                </div>

                {/* Detailed Results */}
                <div className="results-table">
                  <h3>Time-Window Analysis</h3>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Window</th>
                          <th>Time</th>
                          <th>Instrument</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {classificationResults.results.map((result, index) => (
                          <tr key={index}>
                            <td>{result.window}</td>
                            <td>{result.start_time.toFixed(1)}s - {result.end_time.toFixed(1)}s</td>
                            <td>{result.instrument_name}</td>
                            <td>{(result.confidence * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Status/logs overlay */}
        {status && (
          <div className="status-bottom">
            {status}
            {mixtapeUrl && (
              <div>
                <audio controls src={mixtapeUrl} />
                <a href={mixtapeUrl} download="mixtape.wav">Download</a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right: Remix */}
      <div className="sidebar">
        <h2>Remix</h2>
        <div>
          Selected sample:<br />
          {selectedChunk ? getParentTrackFilename(selectedChunk.path) : "None"}
        </div>
        <div>
          Generate mode<br />
          <button onClick={() => handleRemix("interleave")}>interleave</button>
          <button onClick={() => handleRemix("blend")}>blend</button>
          <button onClick={() => handleRemix("average")}>average</button>
        </div>
      </div>
    </div>
  );
};

export default App;