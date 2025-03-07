"use client"

import { useEffect, useMemo, useState } from "react"
import { useTheme } from "next-themes"
import {
  Cloud,
  fetchSimpleIcons as originalFetchSimpleIcons,
  ICloud,
  renderSimpleIcon,
  SimpleIcon,
} from "react-icon-cloud"

// Update version in the URL when fetching icons
const SIMPLE_ICONS_VERSION = "14.8.0" // Updated to the latest version

// Define our custom SimpleIcon interface (it might be different from the react-icon-cloud one)
interface CustomSimpleIcon {
  slug: string;
  title: string;
  path: string;
  hex: string;
  svg?: string; // Add the svg property
}

// Define a mapping for icon slug transformations
// Some icons have different naming conventions in newer versions
const TICKER_TO_ICON_MAPPING: Record<string, string> = {
  // Tech companies
  "MSFT": "microsoft",
  "AMZN": "amazon",
  "ADBE": "adobe",
  "ORCL": "oracle",
  "CSCO": "cisco",
  
  // Financial companies
  "JPM": "jpmorgan",
  "GS": "goldmansachs",
  "BAC": "bankofamerica",
  "WFC": "wellsfargo",
  "BLK": "blackrock",
  "MS": "morganstanley",
  "C": "citigroup",
  "AXP": "americanexpress",
  "SCHW": "charlesschwab",
  
  // Consumer companies
  "DIS": "disney",
  "COST": "costco",
  "PG": "pg",
  "KO": "cocacola",
  "PEP": "pepsi",
  
  // Crypto
  "AVAX-USD": "avalanche",
  "UNI-USD": "uniswap",
  "AAVE-USD": "aave",
  "ATOM-USD": "cosmos",
};

// Alternative slugs to try if the first one fails
const ALTERNATIVE_SLUGS: Record<string, string[]> = {
  "microsoft": ["microsoftazure", "microsoftedge", "microsoftoffice", "microsoftwindows"],
  "amazon": ["amazon"],
  "jpmorgan": ["jpmorganchase"],
  "goldmansachs": ["goldman-sachs", "goldman"],
  "bankofamerica": ["bank-of-america"],
  "wellsfargo": ["wells-fargo"],
  "morganstanley": ["morgan-stanley"],
  "americanexpress": ["american-express", "amex"],
  "charlesschwab": ["charles-schwab"],
  "procterandgamble": ["procter-and-gamble", "procter&gamble", "pg"],
  "cocacola": ["coca-cola"],
  "avalanche": ["avalancheavax"],
};

// Create a wrapper for fetchSimpleIcons that uses our version
const fetchSimpleIcons = async (params: { slugs: string[] }) => {
  try {
    // Override the default behavior to use our version
    const { slugs } = params;
    const simpleIcons: Record<string, CustomSimpleIcon> = {};
    
    // Map ticker symbols to icon slugs using our mapping
    const mappedSlugs = slugs.map(slug => {
      const mapped = TICKER_TO_ICON_MAPPING[slug] || slug.toLowerCase();
      console.debug(`Mapping ${slug} to icon: ${mapped}`);
      return mapped;
    });
    
    console.debug('Fetching icons for slugs:', mappedSlugs);
    
    await Promise.all(
      mappedSlugs.map(async (slug) => {
        try {
          // Try to fetch with the primary slug
          let response = await fetch(`https://cdn.jsdelivr.net/npm/simple-icons@${SIMPLE_ICONS_VERSION}/icons/${slug}.svg`);
          
          // If the primary slug fails, try alternatives if available
          if (!response.ok && ALTERNATIVE_SLUGS[slug]) {
            for (const altSlug of ALTERNATIVE_SLUGS[slug]) {
              console.debug(`Trying alternative slug for ${slug}: ${altSlug}`);
              response = await fetch(`https://cdn.jsdelivr.net/npm/simple-icons@${SIMPLE_ICONS_VERSION}/icons/${altSlug}.svg`);
              if (response.ok) {
                console.debug(`Found icon using alternative slug: ${altSlug}`);
                break;
              }
            }
          }
          
          if (response.ok) {
            const svgText = await response.text();
            
            // Make sure we're only using the SVG content
            const cleanedSvg = svgText
              .replace(/<\?xml.*?\?>/, '') // Remove XML declaration
              .replace(/<\!DOCTYPE.*?>/, '') // Remove DOCTYPE
              .trim();
            
            // SVG needs to maintain viewBox and other attributes
            const viewBoxMatch = cleanedSvg.match(/viewBox="([^"]+)"/);
            const viewBox = viewBoxMatch ? viewBoxMatch[1] : '0 0 24 24';
            
            simpleIcons[slug] = {
              slug,
              title: slug,
              svg: cleanedSvg,
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
    
    console.debug('Successfully fetched icons with version:', SIMPLE_ICONS_VERSION);
    console.debug('Successfully fetched icons:', Object.keys(simpleIcons));
    
    return {
      simpleIcons,
    };
  } catch (error) {
    console.error('Error in custom fetchSimpleIcons:', error);
    // Fallback to original function if our custom one fails
    return originalFetchSimpleIcons(params);
  }
};

export const cloudProps: Omit<ICloud, "children"> = {
  containerProps: {
    style: {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      width: "100%",
      height: "100%",
      padding: "20px 0",
    },
  },
  options: {
    reverse: true,
    depth: 1.5,
    wheelZoom: false,
    imageScale: 2,
    activeCursor: "pointer",
    tooltip: "native",
    initial: [0.1, -0.1],
    clickToFront: 500,
    tooltipDelay: 0,
    outlineColour: "#0000",
    textColour: "#fff",
    textHeight: 0, // Hide text, show only icons
    freezeActive: true,
    shuffleTags: true,
    shape: "sphere",
    noSelect: true,
    noMouse: false,
    freezeDecel: true,
    fadeIn: 1000,
    zoom: 1.1
  },
}

// Map company tickers to their SimpleIcon slugs (updated for v14.8.0)
const tickerToSlug: Record<string, string> = {
  // Tech Companies
  AAPL: "apple",
  GOOGL: "google",
  MSFT: "microsoft",
  AMZN: "amazon",
  META: "facebook",
  NFLX: "netflix",
  TSLA: "tesla",
  NVDA: "nvidia",
  ADBE: "adobe",
  ORCL: "oracle",
  CSCO: "cisco",
  IBM: "ibm",
  INTC: "intel",
  AMD: "amd",
  QCOM: "qualcomm",
  CRM: "salesforce",
  AVGO: "broadcom",
  HPQ: "hp",
  DELL: "dell",
  VMW: "vmware",
  SAP: "sap",
  INTU: "intuit",
  WDAY: "workday",
  NOW: "servicenow",
  PLTR: "palantir",
  AMAT: "applied",
  RHT: "redhat",
  PANW: "paloalto",
  SPLK: "splunk",
  ZEN: "zendesk",
  CRWD: "crowdstrike",
  
  // Consumer & Social Media
  DIS: "disney",
  SBUX: "starbucks",
  ABNB: "airbnb",
  PG: "pg",
  KO: "cocacola",
  PEP: "pepsi",
  MCD: "mcdonalds",
  WMT: "walmart",
  TGT: "target",
  COST: "costco",
  NKE: "nike",
  LEVI: "levis",
  UAA: "underarmour",
  CROX: "crocs",
  ROST: "ross",
  VFC: "vans",
  PUMA: "puma",
  ADS: "adidas",
  
  // Financial & Payment
  JPM: "jpmorgan",
  V: "visa",
  MA: "mastercard",
  PYPL: "paypal",
  MS: "morganstanley",
  GS: "goldmansachs",
  BAC: "bankofamerica",
  WFC: "wellsfargo",
  BLK: "blackrock",
  C: "citigroup",
  AXP: "americanexpress",
  SCHW: "charlesschwab",

  // Crypto
  "BTC-USD": "bitcoin",
  "ETH-USD": "ethereum",
  "SOL-USD": "solana",
  "ADA-USD": "cardano",
  "DOT-USD": "polkadot",
  "AVAX-USD": "avalanche",
  "MATIC-USD": "polygon",
  "LINK-USD": "chainlink",
  "XRP-USD": "xrp",
  "DOGE-USD": "dogecoin",
  "UNI-USD": "uniswap",
  "AAVE-USD": "aave",
  "ATOM-USD": "cosmos",
  "BNB-USD": "binance",
  "LTC-USD": "litecoin",
};

export const renderCustomIcon = (icon: SimpleIcon, theme: string, onClick?: () => void) => {
  // Create style for the icon based on theme
  const iconStyle = {
    display: "inline-block",
    width: "100%",
    height: "100%",
    color: theme === "dark" ? "#fff" : "#000",
    fill: "currentColor",
  };

  // Get icon SVG content if it exists from our custom fetching
  const customSvg = (icon as CustomSimpleIcon).svg;
  
  // If we have the SVG content from our custom fetching, render it directly
  if (customSvg) {
    return (
      <a
        key={icon.slug}
        href="#"
        className="cloud-icon"
        onClick={(e) => {
          e.preventDefault();
          onClick?.();
        }}
        title={icon.title}
        style={{
          width: "42px",
          height: "42px",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={iconStyle}
          dangerouslySetInnerHTML={{ __html: customSvg }}
        />
      </a>
    );
  }

  // Fallback to the original renderSimpleIcon behavior
  return renderSimpleIcon({
    icon,
    bgHex: theme === "dark" ? "#000000" : "#ffffff",
    size: 42,
    fallbackHex: theme === "dark" ? "#fff" : "#000",
    aProps: {
      onClick: (e) => {
        e.preventDefault();
        onClick?.();
      },
    },
  });
};

export type DynamicCloudProps = {
  tickers: string[]
  onTickerSelect: (ticker: string) => void
}

type IconData = Awaited<ReturnType<typeof fetchSimpleIcons>>

export function IconCloud({ tickers, onTickerSelect }: DynamicCloudProps) {
  const [data, setData] = useState<IconData | null>(null)
  const { theme = "dark" } = useTheme()
  
  // Track loading state separately to show loading indicator
  const [isLoading, setIsLoading] = useState(true);
  // Track errors to display an error message
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    // Reset states when tickers change
    setIsLoading(true);
    setHasError(false);
    
    const slugs = tickers.map((ticker) => {
      const hasMapping = tickerToSlug[ticker];
      if (!hasMapping) {
        console.debug(`No icon mapping found for ticker: ${ticker}`);
      }
      return hasMapping;
    }).filter(Boolean) as string[];
    
    if (!slugs.length) {
      console.debug('No mappable tickers provided');
      setIsLoading(false);
      return;
    }

    console.debug('Fetching icons for slugs:', slugs);
    
    fetchSimpleIcons({ 
      slugs
    })
    .then(data => {
      console.debug('Successfully fetched icons:', Object.keys(data.simpleIcons));
      setData(data);
      setIsLoading(false);
    })
    .catch(error => {
      console.error('Error fetching icons:', error);
      setHasError(true);
      setIsLoading(false);
    });
  }, [tickers]);

  const filteredIconData = useMemo(() => {
    return filterIconsByTickers(data, tickers, theme, onTickerSelect)
  }, [data, tickers, theme, onTickerSelect])

  // Show a loading indicator while icons are being fetched
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-blue-400">
        <div className="animate-spin h-8 w-8 border-4 border-blue-500 rounded-full border-t-transparent"></div>
        <span className="ml-3">Loading icons...</span>
      </div>
    );
  }
  
  // Show an error message if fetching failed
  if (hasError) {
    return (
      <div className="flex items-center justify-center h-64 text-red-400">
        <p>Error loading icons. Please try refreshing the page.</p>
      </div>
    );
  }

  // If we have no mapped tickers or no data, show a message
  if (!filteredIconData?.length) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-400 p-4 text-center">
        <p className="mb-2">No icons available for the selected tickers.</p>
        <p className="text-sm">Try adding popular tickers like AAPL, GOOGL, MSFT, or crypto like BTC-USD, ETH-USD.</p>
      </div>
    );
  }

  return (
    <div className="cloud-container mt-4" style={{ minHeight: '300px' }}>
      <Cloud {...cloudProps}>{filteredIconData}</Cloud>
    </div>
  );
}

// Function to filter icons by tickers and render them
const filterIconsByTickers = (
  data: IconData | null, 
  tickers: string[], 
  theme: string,
  onTickerSelect: (ticker: string) => void
) => {
  if (!data) return [];

  return Object.values(data.simpleIcons)
    .map((icon) => {
      // Find the ticker for this icon
      const ticker = Object.entries(tickerToSlug).find(([_, slug]) => slug === icon.slug)?.[0];
      if (!ticker) return null;

      return renderCustomIcon(
        icon,
        theme,
        () => onTickerSelect(ticker)
      );
    })
    .filter(Boolean);
}; 