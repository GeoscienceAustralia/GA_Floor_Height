import React, { useState, useEffect, useCallback, useRef } from 'react'
import DeckGL from '@deck.gl/react'
import { PointCloudLayer, LineLayer, ScatterplotLayer } from '@deck.gl/layers'
import { OrbitView } from '@deck.gl/core'


const CLASSIFICATION_COLORS = {
  0: [128, 128, 128],  
  1: [128, 128, 128],  
  2: [139, 69, 19],    
  3: [34, 139, 34],    
  4: [0, 255, 0],      
  5: [0, 100, 0],      
  6: [255, 165, 0],    
  7: [255, 0, 0],      
  9: [0, 191, 255],    
  10: [128, 0, 128],   
  11: [64, 64, 64],    
  12: [255, 255, 0],   
}

const CLASSIFICATION_NAMES = {
  0: "Never Classified",
  1: "Unclassified",
  2: "Ground",
  3: "Low Vegetation",
  4: "Medium Vegetation",
  5: "High Vegetation",
  6: "Building",
  7: "Low Point",
  9: "Water",
  10: "Rail",
  11: "Road Surface",
  12: "Overlap"
}

function App() {
  const [loading, setLoading] = useState(false)
  const [pointData, setPointData] = useState([])
  const [viewState, setViewState] = useState({
    target: [0, 0, 0],
    zoom: 5,
    rotationX: 15,
    rotationOrbit: 30,
    minZoom: -2,
    maxZoom: 10
  })
  const [pointSize, setPointSize] = useState(1)
  const [selectedRegion, setSelectedRegion] = useState('wagga')
  const [selectedFile, setSelectedFile] = useState(null)  
  const [measurementMode, setMeasurementMode] = useState(false)
  const [measurementPoints, setMeasurementPoints] = useState([])
  const [stats, setStats] = useState(null)
  const [hoveredPoint, setHoveredPoint] = useState(null)

  
  const [initialClipId, setInitialClipId] = useState(null)
  const [initialStateLoaded, setInitialStateLoaded] = useState(false)
  const initializedRef = useRef(false)
  const [browsingMode, setBrowsingMode] = useState(false)
  const [filesList, setFilesList] = useState([])
  const [currentFileIndex, setCurrentFileIndex] = useState(0)

  
  useEffect(() => {
    console.log('App mounted, fetching initial state...')
    
    const timer = setTimeout(() => {
      fetch('/api/initial-state')
        .then(res => {
          console.log('Initial state response status:', res.status)
          return res.json()
        })
        .then(data => {
          console.log('Initial state from server:', data)
          if (data.region) {
            console.log('Setting region to:', data.region)
            setSelectedRegion(data.region)
          }
          if (data.clip_id) {
            console.log('Setting initial clip ID to:', data.clip_id)
            setInitialClipId(data.clip_id)
            setBrowsingMode(false)
            
            setSelectedFile(null)
            setPointData([])
          } else {
            console.log('No clip ID provided, entering browsing mode')
            setBrowsingMode(true)
          }
          setInitialStateLoaded(true)
          console.log('Initial state loaded successfully')
        })
        .catch(err => {
          console.error('Error fetching initial state:', err)
          setInitialStateLoaded(true)
        })
    }, 1000)
    
    return () => clearTimeout(timer)
  }, [])

  
  useEffect(() => {
    console.log('File fetch effect - initialStateLoaded:', initialStateLoaded, 'region:', selectedRegion, 'clipId:', initialClipId, 'browsingMode:', browsingMode)
    if (initialStateLoaded) {
      fetchAvailableFiles(selectedRegion)
    }
  }, [selectedRegion, initialClipId, initialStateLoaded, browsingMode])

  useEffect(() => {
    if (!browsingMode || filesList.length === 0) return

    const handleKeyPress = (e) => {
      if (e.key === 'ArrowLeft') {
        navigateToFile('prev')
      } else if (e.key === 'ArrowRight') {
        navigateToFile('next')
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [browsingMode, filesList, currentFileIndex])

  const fetchAvailableFiles = async (region) => {
    
    if (!initialClipId && !browsingMode) {
      console.log('No clip ID provided and not in browsing mode')
      setLoading(false)
      return
    }
    
    try {
      if (browsingMode) {
        console.log(`Browsing mode: fetching all files for region: ${region}`)
        setLoading(true)
        
        const response = await fetch(`/api/lidar/files/${region}`)
        const data = await response.json()
        
        if (data.files && data.files.length > 0) {
          console.log(`Found ${data.files.length} files in region`)
          setFilesList(data.files)
          setSelectedFile(data.files[0])
          setCurrentFileIndex(0)
        } else {
          console.log('No files found in region')
          setLoading(false)
        }
      } else {
        console.log(`Looking for clip ID: ${initialClipId} in region: ${region}`)
        setLoading(true)
        
        const response = await fetch(`/api/lidar/file-exists/${region}/${initialClipId}`)
        const data = await response.json()
        
        if (data.exists && data.filename) {
          console.log(`Found file: ${data.filename}`)
          setSelectedFile(data.filename)
        } else {
          console.log(`File not found for clip ID: ${initialClipId}`)
          setLoading(false)
          setSelectedFile(null)
        }
      }
    } catch (error) {
      console.error('Error fetching files:', error)
      setLoading(false)
    }
  }

  
  useEffect(() => {
    console.log(`Load effect triggered - selectedFile: ${selectedFile}, selectedRegion: ${selectedRegion}`)
    if (selectedFile) {
      console.log(`Loading point cloud for: ${selectedFile}`)
      loadPointCloud(selectedRegion, selectedFile)
    }
  }, [selectedFile, selectedRegion])

  const loadPointCloud = async (region, filename) => {
    console.log(`loadPointCloud called with region: ${region}, filename: ${filename}`)
    setLoading(true)
    try {
      const url = `/api/lidar/data/${region}/${filename}`
      console.log(`Fetching from: ${url}`)
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      console.log(`Received data with ${data.points?.length || 0} points`)
      
      
      const processedData = data.points.map(point => {
        
        const baseColor = CLASSIFICATION_COLORS[point.classification] || [128, 128, 128]
        
        
        
        
        const intensityFactor = 0.7 + (point.intensity * 0.6)
        
        
        const color = baseColor.map(c => Math.min(255, Math.round(c * intensityFactor)))
        
        return {
          position: [point.x, point.y, point.z],
          color: color,
          classification: point.classification,
          intensity: point.intensity,
          ...point
        }
      })
      
      setPointData(processedData)
      
      
      const bounds = {
        minX: Math.min(...data.points.map(p => p.x)),
        maxX: Math.max(...data.points.map(p => p.x)),
        minY: Math.min(...data.points.map(p => p.y)),
        maxY: Math.max(...data.points.map(p => p.y)),
        minZ: Math.min(...data.points.map(p => p.z)),
        maxZ: Math.max(...data.points.map(p => p.z))
      }
      
      const center = [
        (bounds.minX + bounds.maxX) / 2,
        (bounds.minY + bounds.maxY) / 2,
        (bounds.minZ + bounds.maxZ) / 2
      ]
      
      setViewState(prev => ({ ...prev, target: center }))
      setStats({
        pointCount: data.points.length,
        bounds,
        classifications: data.classifications
      })
      
    } catch (error) {
      console.error('Error loading point cloud:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleClick = useCallback((info) => {
    if (!measurementMode || !info.object) return
    
    const point = info.object
    const newPoints = [...measurementPoints, point]
    
    if (newPoints.length > 2) {
      newPoints.shift() 
    }
    
    setMeasurementPoints(newPoints)
  }, [measurementMode, measurementPoints])

  const handleHover = useCallback((info) => {
    if (measurementMode && info.object) {
      setHoveredPoint(info.object)
    } else {
      setHoveredPoint(null)
    }
  }, [measurementMode])

  const navigateToFile = (direction) => {
    if (!browsingMode || filesList.length === 0) return
    
    let newIndex = currentFileIndex
    if (direction === 'next') {
      newIndex = (currentFileIndex + 1) % filesList.length
    } else {
      newIndex = currentFileIndex === 0 ? filesList.length - 1 : currentFileIndex - 1
    }
    
    setCurrentFileIndex(newIndex)
    setSelectedFile(filesList[newIndex])
    setMeasurementPoints([])
  }

  const getLayers = () => {
    const layers = [
      new PointCloudLayer({
        id: 'point-cloud',
        data: pointData,
        getPosition: d => d.position,
        getColor: d => {
          
          if (measurementMode && hoveredPoint && 
              d.x === hoveredPoint.x && d.y === hoveredPoint.y && d.z === hoveredPoint.z) {
            return [255, 255, 0] 
          }
          return d.color
        },
        getNormal: [0, 0, 15],
        pointSize: measurementMode ? pointSize * 1.5 : pointSize, 
        pickable: true,
        updateTriggers: {
          getColor: [hoveredPoint, measurementMode],
          pointSize: [measurementMode, pointSize]
        }
      })
    ]

    
    if (measurementPoints.length > 0) {
      layers.push(
        new ScatterplotLayer({
          id: 'measurement-points',
          data: measurementPoints,
          getPosition: d => d.position,
          getFillColor: [255, 0, 0],
          getRadius: 3,
          radiusMinPixels: 5,
          radiusMaxPixels: 10
        })
      )

      if (measurementPoints.length === 2) {
        layers.push(
          new LineLayer({
            id: 'measurement-line',
            data: [{
              sourcePosition: measurementPoints[0].position,
              targetPosition: measurementPoints[1].position
            }],
            getSourcePosition: d => d.sourcePosition,
            getTargetPosition: d => d.targetPosition,
            getColor: [255, 0, 0],
            getWidth: 3
          })
        )
      } else if (measurementPoints.length === 1 && hoveredPoint) {
        
        layers.push(
          new LineLayer({
            id: 'measurement-preview',
            data: [{
              sourcePosition: measurementPoints[0].position,
              targetPosition: hoveredPoint.position
            }],
            getSourcePosition: d => d.sourcePosition,
            getTargetPosition: d => d.targetPosition,
            getColor: [255, 255, 0, 128], 
            getWidth: 2,
            getDashArray: [4, 4] 
          })
        )
      }
    }

    return layers
  }

  const getMeasurementResults = () => {
    if (measurementPoints.length !== 2) return null
    
    const p1 = measurementPoints[0]
    const p2 = measurementPoints[1]
    
    const dx = p2.x - p1.x
    const dy = p2.y - p1.y
    const dz = p2.z - p1.z
    
    const horizontalDist = Math.sqrt(dx * dx + dy * dy)
    const slopeDist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    const heightDiff = Math.abs(dz)
    const angle = Math.atan2(Math.abs(dz), horizontalDist) * 180 / Math.PI
    
    return {
      heightDiff: heightDiff.toFixed(3),
      horizontalDist: horizontalDist.toFixed(3),
      slopeDist: slopeDist.toFixed(3),
      angle: angle.toFixed(1),
      p1Height: p1.z.toFixed(3),
      p2Height: p2.z.toFixed(3)
    }
  }

  return (
    <div style={{ 
      position: 'relative', 
      width: '100vw', 
      height: '100vh',
      cursor: measurementMode ? 'crosshair' : 'grab'
    }}>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller={true}
        views={new OrbitView()}
        layers={getLayers()}
        onClick={handleClick}
        onHover={handleHover}
        getTooltip={({ object }) => object && {
          html: `
            <div style="background: rgba(0,0,0,0.8); padding: 10px; border-radius: 4px;">
              <div><b>X:</b> ${object.x.toFixed(3)} m</div>
              <div><b>Y:</b> ${object.y.toFixed(3)} m</div>
              <div style="background: yellow; color: black; padding: 2px;"><b>Z:</b> ${object.z.toFixed(3)} m</div>
              <div><b>Class:</b> ${CLASSIFICATION_NAMES[object.classification] || 'Unknown'}</div>
              <div><b>Intensity:</b> ${(object.intensity * 100).toFixed(1)}%</div>
            </div>
          `
        }}
      />

      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div>Loading point cloud...</div>
        </div>
      )}

      <div className="control-panel">
        <h2>LiDAR Point Cloud Viewer</h2>
        
        <div className="control-group">
          <label>Viewing Clip</label>
          <div style={{padding: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', fontSize: '14px', fontFamily: 'monospace'}}>
            {selectedFile ? selectedFile.split('/').pop() : (initialClipId ? `Searching for ${initialClipId}...` : 'No clip shown')}
          </div>
        </div>

        {browsingMode && filesList.length > 0 && (
          <div className="control-group">
            <label>Browse Files ({currentFileIndex + 1} of {filesList.length})</label>
            <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
              <button 
                onClick={() => navigateToFile('prev')} 
                style={{flex: 1}}
                title="Previous file (←)"
              >
                ← Previous
              </button>
              <button 
                onClick={() => navigateToFile('next')} 
                style={{flex: 1}}
                title="Next file (→)"
              >
                Next →
              </button>
            </div>
          </div>
        )}

        <div className="control-group">
          <label>Point Size: {pointSize}</label>
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.5"
            value={pointSize}
            onChange={e => setPointSize(parseFloat(e.target.value))}
          />
        </div>

        <div className="control-group">
          <button onClick={() => {
            setMeasurementMode(!measurementMode)
            setMeasurementPoints([])
            setHoveredPoint(null)
          }}>
            {measurementMode ? 'Exit Measurement Mode' : 'Enable Measurement'}
          </button>
        </div>
      </div>

      {measurementMode && (
        <div className="measurement-panel">
          <h3>📏 Measurement Tool</h3>
          {measurementPoints.length === 0 && <p>Click on a point to start measuring</p>}
          {measurementPoints.length === 1 && <p>Click on a second point</p>}
          {measurementPoints.length === 2 && getMeasurementResults() && (
            <div className="measurement-result">
              <div>Point 1 Height: <span className="value">{getMeasurementResults().p1Height} m</span></div>
              <div>Point 2 Height: <span className="value">{getMeasurementResults().p2Height} m</span></div>
              <div>Height Difference: <span className="value">{getMeasurementResults().heightDiff} m</span></div>
              <div>Horizontal Distance: <span className="value">{getMeasurementResults().horizontalDist} m</span></div>
              <div>Slope Distance: <span className="value">{getMeasurementResults().slopeDist} m</span></div>
              <div>Angle: <span className="value">{getMeasurementResults().angle}°</span></div>
            </div>
          )}
        </div>
      )}

      {stats && (
        <div className="stats-panel">
          <div>Points: {stats.pointCount.toLocaleString()}</div>
          <div>X: {stats.bounds.minX.toFixed(2)} to {stats.bounds.maxX.toFixed(2)}</div>
          <div>Y: {stats.bounds.minY.toFixed(2)} to {stats.bounds.maxY.toFixed(2)}</div>
          <div>Z: {stats.bounds.minZ.toFixed(2)} to {stats.bounds.maxZ.toFixed(2)}</div>
        </div>
      )}
    </div>
  )
}

export default App