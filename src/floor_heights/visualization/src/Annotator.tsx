import React, { useState, useEffect, useCallback } from 'react';
import { Stage, Layer, Image as KonvaImage, Line, Group } from 'react-konva';
import useImage from 'use-image';

const CATEGORY_COLORS = {
  0: '#E74C3C',
  1: '#2ECC71',
  2: '#3498DB',
  3: '#F39C12',
  4: '#9B59B6',
};

const CATEGORY_NAMES = {
  0: 'Door',
  1: 'Foundation',
  2: 'Garage',
  3: 'Stairs',
  4: 'Window',
};

interface COCOAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  segmentation: number[][];
  bbox: number[];
  area: number;
}

interface COCOImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}

interface COCOCategory {
  id: number;
  name: string;
}

interface AnnotatorProps {}

const imageCache = new Map<string, HTMLImageElement>();

const preloadImage = (url: string): Promise<HTMLImageElement> => {
  if (imageCache.has(url)) {
    return Promise.resolve(imageCache.get(url)!);
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      imageCache.set(url, img);
      resolve(img);
    };
    img.onerror = reject;
    img.src = url;
  });
};

const ImageWithAnnotations: React.FC<{
  imageUrl: string;
  annotations: COCOAnnotation[];
  imageInfo: COCOImage;
  stageWidth: number;
  stageHeight: number;
  maskOpacity: number;
}> = ({ imageUrl, annotations, imageInfo, stageWidth, stageHeight, maskOpacity }) => {
  const [image, status] = useImage(imageUrl);
  const [showMasks, setShowMasks] = useState(false);

  useEffect(() => {
    if (status === 'loaded') {
      const timer = setTimeout(() => setShowMasks(true), 50);
      return () => clearTimeout(timer);
    } else {
      setShowMasks(false);
    }
  }, [status]);

  const padding = 40;
  const scale = Math.min(
    (stageWidth - padding) / imageInfo.width,
    (stageHeight - padding) / imageInfo.height
  ) * 0.95;

  const scaledWidth = imageInfo.width * scale;
  const scaledHeight = imageInfo.height * scale;
  const offsetX = (stageWidth - scaledWidth) / 2;
  const offsetY = (stageHeight - scaledHeight) / 2;

  return (
    <Group>
      {image && (
        <KonvaImage
          image={image}
          width={scaledWidth}
          height={scaledHeight}
          x={offsetX}
          y={offsetY}
        />
      )}

      {showMasks && image && annotations.map((ann, idx) => {
        const color = CATEGORY_COLORS[ann.category_id] || '#808080';

        return ann.segmentation.map((polygon, polyIdx) => {
          const points = [];
          for (let i = 0; i < polygon.length; i += 2) {
            points.push(
              polygon[i] * scale + offsetX,
              polygon[i + 1] * scale + offsetY
            );
          }

          return (
            <Line
              key={`${idx}-${polyIdx}`}
              points={points}
              fill={color}
              opacity={maskOpacity}
              closed
              stroke={color}
              strokeWidth={1.5}
              strokeOpacity={Math.min(maskOpacity + 0.2, 1)}
            />
          );
        });
      })}
    </Group>
  );
};

export const Annotator: React.FC<AnnotatorProps> = () => {
  const [images, setImages] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [cocoData, setCocoData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [maskOpacity, setMaskOpacity] = useState(0.75);
  const [imageLoading, setImageLoading] = useState(false);

  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight - 120
  });

  useEffect(() => {
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight - 120
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);

        const cachedData = localStorage.getItem('annotatorData');
        const cacheTimestamp = localStorage.getItem('annotatorDataTimestamp');
        const cacheAge = cacheTimestamp ? Date.now() - parseInt(cacheTimestamp) : Infinity;
        const oneHour = 60 * 60 * 1000;

        let imagesData, coco;

        if (cachedData && cacheAge < oneHour) {
          const cached = JSON.parse(cachedData);
          imagesData = cached.images;
          coco = cached.coco;
        } else {
          const [imagesResponse, cocoResponse] = await Promise.all([
            fetch('/api/annotation/images'),
            fetch('/api/annotation/coco')
          ]);

          imagesData = await imagesResponse.json();
          coco = await cocoResponse.json();

          localStorage.setItem('annotatorData', JSON.stringify({ images: imagesData, coco }));
          localStorage.setItem('annotatorDataTimestamp', Date.now().toString());
        }

        setImages(imagesData.images);
        setCocoData(coco);

        if (imagesData.images.length > 0) {
          const preloadCount = Math.min(3, imagesData.images.length);
          const preloadPromises = [];
          for (let i = 0; i < preloadCount; i++) {
            preloadPromises.push(preloadImage(`/api/annotation/image/${imagesData.images[i]}`));
          }
          await Promise.all(preloadPromises);
        }

        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handlePrevious = useCallback(() => {
    setImageLoading(true);
    setCurrentIndex(prev => Math.max(0, prev - 1));
  }, []);

  const handleNext = useCallback(() => {
    setImageLoading(true);
    setCurrentIndex(prev => Math.min(images.length - 1, prev + 1));
  }, [images.length]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') handlePrevious();
      if (e.key === 'ArrowRight') handleNext();
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handlePrevious, handleNext]);

  useEffect(() => {
    if (images.length === 0) return;

    const preloadAdjacent = async () => {
      const preloadPromises = [];

      if (currentIndex > 0) {
        preloadPromises.push(
          preloadImage(`/api/annotation/image/${images[currentIndex - 1]}`)
        );
      }

      preloadPromises.push(
        preloadImage(`/api/annotation/image/${images[currentIndex]}`)
      );

      for (let i = 1; i <= 2; i++) {
        if (currentIndex + i < images.length) {
          preloadPromises.push(
            preloadImage(`/api/annotation/image/${images[currentIndex + i]}`)
          );
        }
      }

      await Promise.all(preloadPromises);
    };

    preloadAdjacent();

    const timer = setTimeout(() => setImageLoading(false), 50);
    return () => clearTimeout(timer);
  }, [currentIndex, images]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p className="mt-4 text-gray-300">Loading annotations...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-8 max-w-md">
          <h2 className="text-xl font-semibold text-red-400 mb-2">Error</h2>
          <p className="text-gray-300">{error}</p>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-yellow-900/20 border border-yellow-500/50 rounded-lg p-8 max-w-md">
          <h2 className="text-xl font-semibold text-yellow-400 mb-2">No images found</h2>
          <p className="text-gray-300">No images found in the dataset directory</p>
        </div>
      </div>
    );
  }

  const currentImage = images[currentIndex];
  const imageId = cocoData?.filename_to_image_id[currentImage];
  const imageInfo = cocoData?.images.find((img: COCOImage) => img.id === imageId);
  const annotations = cocoData?.annotations_by_image[imageId] || [];

  const categoryCounts: Record<number, number> = {};
  annotations.forEach((ann: COCOAnnotation) => {
    categoryCounts[ann.category_id] = (categoryCounts[ann.category_id] || 0) + 1;
  });

  return (
    <div className="h-screen bg-gray-900 flex flex-col overflow-hidden">
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <h1 className="text-xl font-semibold text-white">Annotation Viewer</h1>
            <div className="flex items-center space-x-3 text-sm">
              <span className="text-gray-400">{currentIndex + 1} / {images.length}</span>
              <span className="text-gray-300 font-medium">{currentImage}</span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {Object.entries(CATEGORY_NAMES).map(([id, name]) => {
              const count = categoryCounts[parseInt(id)] || 0;
              return count > 0 ? (
                <div key={id} className="flex items-center space-x-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: CATEGORY_COLORS[parseInt(id)] }}
                  />
                  <span className="text-sm text-gray-300">{name}</span>
                  <span className="text-sm font-medium text-white bg-gray-700 px-2 py-0.5 rounded">
                    {count}
                  </span>
                </div>
              ) : null;
            })}
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <span className="text-sm text-gray-400">Opacity</span>
              <input
                type="range"
                min="0"
                max="100"
                value={maskOpacity * 100}
                onChange={(e) => setMaskOpacity(parseInt(e.target.value) / 100)}
                className="w-24 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <span className="text-sm font-medium text-gray-300 w-12 text-right">
                {Math.round(maskOpacity * 100)}%
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                className="p-2 rounded-md bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white transition-colors"
                onClick={handlePrevious}
                disabled={currentIndex === 0}
                title="Previous (←)"
              >
                <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </button>
              <button
                className="p-2 rounded-md bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 text-white transition-colors"
                onClick={handleNext}
                disabled={currentIndex === images.length - 1}
                title="Next (→)"
              >
                <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 bg-gray-950 relative">
        {imageLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 z-10 pointer-events-none">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        )}
        <Stage width={dimensions.width} height={dimensions.height}>
          <Layer>
            {imageInfo && (
              <ImageWithAnnotations
                key={currentImage}
                imageUrl={`/api/annotation/image/${currentImage}`}
                annotations={annotations}
                imageInfo={imageInfo}
                stageWidth={dimensions.width}
                stageHeight={dimensions.height}
                maskOpacity={maskOpacity}
              />
            )}
          </Layer>
        </Stage>
      </div>

      <div className="bg-gray-800 border-t border-gray-700 px-6 py-2">
        <div className="flex items-center justify-center space-x-4 text-xs text-gray-400">
          <span>Annotations: {annotations.length}</span>
          <span>•</span>
          <span>Resolution: {imageInfo?.width} × {imageInfo?.height}</span>
          <span>•</span>
          <span>Use ← → arrow keys to navigate</span>
        </div>
      </div>
    </div>
  );
};

export default Annotator;
