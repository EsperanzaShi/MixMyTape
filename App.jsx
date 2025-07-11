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
  const [selectedClips, setSelectedClips] = useState([]);
  const [topN, setTopN] = useState(5);
  const [anchorClip, setAnchorClip] = useState(null);
  const [similarPoints, setSimilarPoints] = useState([]);

  useEffect(() => {
    fetch("/data/umap_embeddings.json")
      .then((res) => res.json())
      .then((data) => {
        setPoints(data);
        setGenres([...new Set(data.map((p) => p.genre))]);
        setSelectedGenres(new Set(data.map((p) => p.genre)));
      });
  }, []);

  const handleGenreToggle = (genre) => {
    const newSet = new Set(selectedGenres);
    newSet.has(genre) ? newSet.delete(genre) : newSet.add(genre);
    setSelectedGenres(newSet);
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(selectedClips, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "mixtape.json";
    link.click();
  };

  const handleClickPoint = (data) => {
    const path = data.points[0].customdata;
    const clicked = points.find((p) => p.path === path);
    if (!clicked || !clicked.latent) return;

    setAnchorClip(clicked);

    const simScores = points
      .filter(p => p.genre !== "hiphop" && p.latent)
      .map(p => ({
        ...p,
        similarity: cosineSimilarity(clicked.latent, p.latent)
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topN);

    setSimilarPoints(simScores);

    const audio = new Audio(clicked.path);
    audio.play();
    setSelectedClips(prev => [...prev, clicked]);
  };

  const basePoints = anchorClip ? similarPoints : points.filter(p => selectedGenres.has(p.genre));

  const plotData = [];

  if (anchorClip) {
    plotData.push({
      x: [anchorClip.x],
      y: [anchorClip.y],
      z: [anchorClip.z],
      text: [`🎯 Anchor<br>${anchorClip.genre}<br>${anchorClip.id}`],
      customdata: [anchorClip.path],
      type: "scatter3d",
      mode: "markers",
      marker: {
        size: 8,
        color: "red",
      },
      name: "Anchor",
    });
  }

  plotData.push({
    x: basePoints.map((p) => p.x),
    y: basePoints.map((p) => p.y),
    z: basePoints.map((p) => p.z),
    text: basePoints.map((p) => `Genre: ${p.genre}<br>ID: ${p.id}<br>Score: ${p.score}`),
    customdata: basePoints.map((p) => p.path),
    type: "scatter3d",
    mode: "markers",
    marker: {
      size: 5,
      color: basePoints.map((p) => p.similarity || p.score || 0),
      colorscale: "Viridis",
      opacity: 0.8,
    },
    name: "Suggested",
  });

  return (
    <div className="app">
      <div className="sidebar">
        <h2>Genres</h2>
        {genres.map((genre) => (
          <label key={genre}>
            <input
              type="checkbox"
              checked={selectedGenres.has(genre)}
              onChange={() => handleGenreToggle(genre)}
            />
            {genre}
          </label>
        ))}

        <h2>Top-N Similar</h2>
        <input
          type="number"
          min="1"
          value={topN}
          onChange={(e) => setTopN(Number(e.target.value))}
        />
        <button onClick={() => {
          setAnchorClip(null);
          setSimilarPoints([]);
        }}>
          Clear Top-N
        </button>

        <button onClick={handleExport}>Export Selected</button>
      </div>

      <Plot
        data={plotData}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#fff" },
          scene: {
            xaxis: { title: "UMAP-X", gridcolor: "rgba(255,255,255,0.1)" },
            yaxis: { title: "UMAP-Y", gridcolor: "rgba(255,255,255,0.1)" },
            zaxis: { title: "UMAP-Z", gridcolor: "rgba(255,255,255,0.1)" },
          },
          autosize: true,
          height: 600,
          margin: { t: 30, r: 0, l: 0, b: 0 },
        }}
        onClick={handleClickPoint}
      />
    </div>
  );
};

export default App;