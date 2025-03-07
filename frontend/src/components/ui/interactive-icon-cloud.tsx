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
    depth: 0.8,
    wheelZoom: false,
    imageScale: 1.75,
    activeCursor: "pointer",
    tooltip: "native",
    clickToFront: 500,
    tooltipDelay: 0,
    outlineColour: "#0000",
    maxSpeed: 0.04,
    minSpeed: 0.02,
    radiusX: 0.9,
    radiusY: 0.9,
    radiusZ: 0.9,
    stretchX: 1,
    stretchY: 1,
    offsetX: 0,
    offsetY: 0,
    shuffleTags: true,
    initial: [0.1, -0.1],
    fadeIn: 500,
    zoom: 0.95,
    dragControl: true,
    noSelect: false,
    noMouse: false,
    pinchZoom: true,
    freezeActive: true,
    freezeDecel: true,
    shape: "sphere"
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
  const bgHex = theme === "light" ? "#f3f2ef" : "#080510"
  const fallbackHex = theme === "light" ? "#6e6e73" : "#ffffff"
  const minContrastRatio = theme === "dark" ? 2 : 1.2

  return renderSimpleIcon({
    icon,
    bgHex,
    fallbackHex,
    minContrastRatio,
    size: 36,
    aProps: {
      href: undefined,
      target: undefined,
      rel: undefined,
      onClick: (e: any) => {
        e.preventDefault()
        onClick?.()
      },
      style: {
        cursor: onClick ? 'pointer' : 'default',
      }
    },
  })
}

export type DynamicCloudProps = {
  tickers: string[]
  onTickerSelect: (ticker: string) => void
}

type IconData = Awaited<ReturnType<typeof fetchSimpleIcons>>

export function IconCloud({ tickers, onTickerSelect }: DynamicCloudProps) {
  const [data, setData] = useState<IconData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { theme } = useTheme()

  useEffect(() => {
    const slugs = tickers
      .filter(ticker => {
        // Check if the ticker exists in our mapping
        const hasMapping = tickerToSlug[ticker];
        if (!hasMapping) {
          console.debug(`No icon mapping found for ticker: ${ticker}`);
        }
        return hasMapping;
      })
      .map(ticker => {
        const slug = tickerToSlug[ticker];
        console.debug(`Mapping ${ticker} to icon: ${slug}`);
        return slug;
      });
    
    if (slugs.length === 0) {
      console.debug('No valid icons to display');
      return;
    }

    console.debug('Fetching icons for slugs:', slugs);
    
    fetchSimpleIcons({ 
      slugs
    })
    .then(data => {
      console.debug('Successfully fetched icons:', Object.keys(data.simpleIcons));
      setData(data);
    })
    .catch(err => {
      console.error('Error fetching icons:', err);
      setError('Failed to load icons');
    });
  }, [tickers]);

  const renderedIcons = useMemo(() => {
    if (!data) return null

    return Object.values(data.simpleIcons).map((icon) => {
      // Find the ticker for this icon
      const ticker = Object.entries(tickerToSlug).find(([_, slug]) => slug === icon.slug)?.[0]
      if (!ticker) return null

      return renderCustomIcon(
        icon,
        theme || "light",
        () => onTickerSelect(ticker)
      )
    }).filter(Boolean)
  }, [data, theme, onTickerSelect])

  return (
    <div className="w-full h-[500px] relative bg-black/20 rounded-lg overflow-hidden">
      {/* @ts-ignore */}
      <Cloud {...cloudProps}>
        <>{renderedIcons}</>
      </Cloud>
    </div>
  )
} 