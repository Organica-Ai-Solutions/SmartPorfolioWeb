import React, { useEffect, useState } from 'react';
import { Cloud } from 'react-icon-cloud';

// Types for simple icons
interface SimpleIcon {
  slug: string;
  title: string;
  svg: string;
  path: string;
  hex: string;
}

interface SimpleIconsResult {
  simpleIcons: Record<string, SimpleIcon>;
}

// Function to fetch icons with the correct version
const fetchSimpleIcons = async (slugs: string[]): Promise<SimpleIconsResult> => {
  try {
    console.debug('Fetching icons for slugs:', slugs);
    
    const simpleIcons: Record<string, SimpleIcon> = {};
    
    await Promise.all(
      slugs.map(async (slug) => {
        try {
          // Using the latest simple-icons version (14.8.0)
          const response = await fetch(`https://cdn.jsdelivr.net/npm/simple-icons@14.8.0/icons/${slug}.svg`);
          if (response.ok) {
            const svgText = await response.text();
            simpleIcons[slug] = {
              slug,
              title: slug,
              svg: svgText,
              path: '',
              hex: 'ffffff',
            };
          } else {
            console.warn(`Failed to fetch icon for ${slug}: ${response.status}`);
          }
        } catch (error) {
          console.error(`Error fetching icon for ${slug}:`, error);
        }
      })
    );
    
    console.debug('Successfully fetched icons:', Object.keys(simpleIcons));
    
    return {
      simpleIcons,
    };
  } catch (error) {
    console.error('Error fetching simple icons:', error);
    return {
      simpleIcons: {},
    };
  }
};

// Function to render a simple icon
const renderSimpleIcon = (icon: SimpleIcon, size = 42): React.ReactElement => {
  return (
    <a
      key={icon.slug}
      href="#"
      className="text-white hover:text-blue-200 transition-colors"
      onClick={(e) => e.preventDefault()}
      title={icon.title}
    >
      <div dangerouslySetInnerHTML={{ __html: icon.svg }} style={{ width: size, height: size }} />
    </a>
  );
};

interface IconCloudProps {
  slugs: string[];
}

export const IconCloud: React.FC<IconCloudProps> = ({ slugs }) => {
  const [icons, setIcons] = useState<React.ReactElement[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadIcons = async () => {
      setLoading(true);
      const result = await fetchSimpleIcons(slugs);
      
      if (result.simpleIcons) {
        const iconElements = Object.values(result.simpleIcons).map((icon) => 
          renderSimpleIcon(icon)
        );
        setIcons(iconElements);
      }
      
      setLoading(false);
    };

    loadIcons();
  }, [slugs]);

  if (loading) {
    return <div className="text-white">Loading icons...</div>;
  }

  return (
    <div className="icon-cloud-container">
      <Cloud 
        containerProps={{
          style: {
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '290px',
          }
        }}
        options={{
          clickToFront: 500,
          depth: 1,
          imageScale: 2,
          initial: [0.1, -0.1],
          outlineColour: '#0000',
          reverse: true,
          tooltip: 'native',
          tooltipDelay: 0,
          wheelZoom: false,
        }}
      >
        {icons}
      </Cloud>
    </div>
  );
};

export default IconCloud; 